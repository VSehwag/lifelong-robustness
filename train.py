# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import argparse
import importlib
import time
import logging
import warnings
from collections import OrderedDict
import pdb
import importlib
import pickle

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import models
import data
import trainers
import evaluators
import attacks
from utils import *
import warnings
warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser(description="Lifelong robustness")
    
    parser.add_argument("--configs", type=str, default="./configs/configs_cifar.yml")
    parser.add_argument(
        "--results_dir", type=str, default="./trained_models/",
    )
    parser.add_argument("--exp-name", type=str, default="temp")

    parser.add_argument("--arch", type=str, default="cnnLarge")
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=10)
    
    parser.add_argument("--trainer", type=str, choices=("base", "adv", "adv_ensemble"), default="base")
    parser.add_argument("--evaluator", type=str, choices=("base", "adv", "adv_ensemble"), default="base")
    
    parser.add_argument("--train-attack", type=str, choices=("none", "linf", "l2", "snow", "gabor", "jpeg"), default="linf")
    parser.add_argument("--eval-attack", type=str, choices=("none", "linf", "l2", "snow", "gabor", "jpeg"), default="linf")
    
    parser.add_argument("--train-attacks-list", nargs="+", default=None)
    parser.add_argument("--eval-attacks-list", nargs="+", default=None)
    parser.add_argument("--ensemble-mode", type=str, default="max") # using same ensemble mode for training and evaluation
    
    parser.add_argument("--freeze-block", type=int, default=-1, help="Layers before this block are frozen. -1: no frozen layer")
    
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--datadir", type=str, default="./datasets/")
    parser.add_argument("--in-channel", type=int, default=3)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--upsample-size", type=int, default=None)
    
    parser.add_argument("--autoattack", action="store_true", default=False, 
                        help="Use AutoAttack instead of PGD in evaluation only")
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--ckpt", type=str, help="checkpoint path")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    
    args = update_args(parser.parse_args())
    assert args.normalize == False, "Presumption for most code is that the pixel range is [0,1]"
    if args.batch_size > 256 and not args.warmup:
        warnings.warn("Use warmup training for larger batch-sizes > 256")
    
    # create resutls dir (for logs, checkpoints, etc.)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    result_main_dir = os.path.join(args.results_dir, args.exp_name)
    result_sub_dir = os.path.join(result_main_dir, f"trial_{args.trial}")
    create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    model = torch.nn.DataParallel(models.__dict__[args.arch](in_channel=args.in_channel, num_classes=args.num_classes, freeze_block=args.freeze_block)).cuda()
    # model = torch.nn.DataParallel(models.__dict__[args.arch](in_channel=args.in_channel, num_classes=args.num_classes, width=args.width, freeze_block=args.freeze_block)).cuda()
    # print(model)
    
    if args.ckpt:
        ckpt_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(ckpt_dict)
        print(f"Checkpoint loaded from {args.ckpt}")
        
    # Dataloader
    if args.upsample_size is None:
        train_loader, test_loader, _ = data.__dict__[args.dataset](
            args.datadir,
            normalize=args.normalize,
            batch_size=args.batch_size,
        )
    else:
        train_loader, test_loader, _ = data.__dict__[args.dataset](
            args.datadir,
            normalize=args.normalize,
            batch_size=args.batch_size,
            size=args.upsample_size,
        )

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    trainer = getattr(trainers, args.trainer)
    val = getattr(evaluators, args.evaluator)
    
    # warmup
    if args.warmup:
        wamrup_epochs = 5
        print(f"Warmup training for {wamrup_epochs} epochs")
        warmup_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.001, max_lr=args.lr, step_size_up=wamrup_epochs*len(train_loader)
        )
        for epoch in range(wamrup_epochs):
            trainer(
                model,
                "cuda:0",
                train_loader,
                criterion,
                optimizer,
                args.TrainAttack,
                warmup_lr_scheduler,
                epoch,
                args,
            )

    best_prec = 0

    for p in optimizer.param_groups:
        p["lr"] = args.lr
        p["initial_lr"] = args.lr
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), 1e-4
    )
    
    epoch_results = {}
    for epoch in range(0, args.epochs):
        results_train = trainer(model, "cuda:0", train_loader, criterion, optimizer, args.TrainAttack, lr_scheduler, epoch, args)
        print("Using TrainAttack parameters for faster evaluation per epoch. Need to perform another eval post training with EvalAttack parameters.")
        results_val = val(model, "cuda:0", test_loader, criterion, args.TrainAttack, epoch, args)
        epoch_results[epoch] = {'train':results_train, 'val':results_val}
        
        if args.evaluator == "base":
            prec = results_val["top1"]
        elif args.evaluator in ["adv", "adv_ensemble"]:
            prec = results_val["top1_adv"]
        else:
            raise ValueError()
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        
        d = {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec,
            "optimizer": optimizer.state_dict(),
        }

        os.makedirs(os.path.join(result_sub_dir, f"checkpoint_epoch_{epoch}"), exist_ok=True)
        save_checkpoint(
            d, is_best, results_dir=os.path.join(result_sub_dir, f"checkpoint_epoch_{epoch}"),
        )
        
        logger.info(f"Epoch {epoch}, " + ", ".join(["{}: {:.3f}".format(k+"_train", v) for (k,v) in results_train.items()]+["{}: {:.3f}".format(k+"_val", v) for (k,v) in results_val.items()]))
        with open(os.path.join(result_sub_dir, "train_logs.pkl"), "wb") as f:
            pickle.dump(epoch_results, f)


if __name__ == "__main__":
    main()
