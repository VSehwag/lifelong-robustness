import torch
import torch.nn as nn
import time
import numpy as np
from utils import AverageMeter, ProgressMeter, accuracy
from attack_vectors import get_attack_vector, combine_attack_vectors

def base(
    model,
    device,
    dataloader,
    criterion,
    optimizer,
    attack_params=None,
    lr_scheduler=None,
    epoch=0,
    args=None,
):
    print(
        " ->->->->->->->->->-> One epoch with supervised training <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc1", ":6.2f")
    top5 = AverageMeter("Acc5", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)

        # basic properties of training
        if i == 0:
            print(
                "images :", images.shape,
                "target :", target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [min: {}, max: {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )

        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    result = {"top1": top1.avg, "top5":  top5.avg, "loss": losses.avg}
    return result
            
            
def adv(
    model,
    device,
    dataloader,
    criterion,
    optimizer,
    attack_params=None,
    lr_scheduler=None,
    epoch=0,
    args=None,
):
    print(
        " ->->->->->->->->->-> One epoch with supervised training <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc1", ":6.2f")
    top5 = AverageMeter("Acc5", ":6.2f")
    losses_adv = AverageMeter("Loss-adv", ":.3f")
    top1_adv = AverageMeter("Acc1-adv", ":6.2f")
    top5_adv = AverageMeter("Acc5-adv", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top1, top5, losses_adv, top1_adv, top5_adv],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    
    attack_vector = get_attack_vector(args.train_attack, attack_params)
    freeze = False if args.freeze_block == -1 else True
    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)

        # basic properties of training
        if i == 0:
            print(
                "images :", images.shape,
                "target :", target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [min: {}, max: {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )
        
        
        output = model(images, freeze=freeze)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        #model.eval() # turn off batch-norm in adv. example generation
        images, target = attack_vector(model, images, target, target)
        #model.train() 
        output = model(images, freeze=freeze)
        loss_adv = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_adv.update(loss_adv.item(), images.size(0))
        top1_adv.update(acc1[0], images.size(0))
        top5_adv.update(acc5[0], images.size(0))
        
        loss = (loss + loss_adv) / 2.0 # combine benign and adversarial loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
    result = {"top1": top1.avg, "top5": top5.avg, "loss": losses.avg, "top1_adv": top1_adv.avg, "top5_adv": top5_adv.avg, "loss": losses_adv.avg}
    return result


def adv_ensemble(
    model,
    device,
    dataloader,
    criterion,
    optimizer,
    attack_params=None,
    lr_scheduler=None,
    epoch=0,
    args=None,
):
    print(
        " ->->->->->->->->->-> One epoch with supervised training <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc1", ":6.2f")
    top5 = AverageMeter("Acc5", ":6.2f")
    losses_adv = AverageMeter("Loss-adv", ":.3f")
    top1_adv = AverageMeter("Acc1-adv", ":6.2f")
    top5_adv = AverageMeter("Acc5-adv", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top1, top5, losses_adv, top1_adv, top5_adv],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    
    # ToDo: Merge this in adv trainer itself.
    attack_vector = combine_attack_vectors(args.train_attacks_list, attack_params, args.ensemble_mode)

    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)

        # basic properties of training
        if i == 0:
            print(
                "images :", images.shape,
                "target :", target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [min: {}, max: {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )
        
        
        output = model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        #model.eval() # turn off batch-norm in adv. example generation
        images, target = attack_vector(model, images, target, target)
        #model.train() 
        output = model(images)
        loss_adv = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_adv.update(loss_adv.item(), images.size(0))
        top1_adv.update(acc1[0], images.size(0))
        top5_adv.update(acc5[0], images.size(0))
        
        loss = (loss + loss_adv) / 2.0 # combine benign and adversarial loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
    result = {"top1": top1.avg, "top5": top5.avg, "loss": losses.avg, "top1_adv": top1_adv.avg, "top5_adv": top5_adv.avg, "loss": losses_adv.avg}
    return result
