import torch
import torch.nn as nn
import time
from utils import AverageMeter, ProgressMeter, accuracy
import attacks
from autoattack import AutoAttack

def base(model, device, val_loader, criterion, attack_params=None, epoch=0, args=None):
    """
        Evaluating on validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc1", ":6.2f")
    top5 = AverageMeter("Acc5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

        progress.display(i)  # print final results

    return round(top1.avg.item(), 4), round(top5.avg.item(), 4)



def adv(model, device, val_loader, criterion, attack_params=None, epoch=0, args=None):
    """
        Evaluating on validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("AdvLoss", ":.4f")
    top1 = AverageMeter("Acc1", ":6.2f")
    top5 = AverageMeter("Acc5", ":6.2f")
    adv_top1 = AverageMeter("Adv-Acc1", ":6.2f")
    adv_top5 = AverageMeter("Adv-Acc5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, adv_losses, top1, top5, adv_top1, adv_top5],
        prefix="Test: ",
    )
    
    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            
            # adversarial images
            if args.autoattack:
                if args.eval_attack == "linf":
                    adversary = AutoAttack(model, norm='Linf', eps=args.EvalAttack.linf.epsilon, version='standard')
                    images = adversary.run_standard_evaluation(images, target, bs=len(images))
                elif args.eval_attack == "l2":
                    adversary = AutoAttack(model, norm='L2', eps=args.EvalAttack.l2.epsilon, version='standard')
                    images = adversary.run_standard_evaluation(images, target, bs=len(images))
            else:
                images = getattr(attacks, args.eval_attack)(model, images, target, attack_params)
    
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            adv_losses.update(loss.item(), images.size(0))
            adv_top1.update(acc1[0], images.size(0))
            adv_top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

        progress.display(i)  # print final results

    return round(adv_top1.avg.item(), 4), round(adv_top5.avg.item(), 4)
