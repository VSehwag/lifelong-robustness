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


def main():
    parser = argparse.ArgumentParser(description="Lifelong robustness")
    
    parser.add_argument("--configs", type=str, default="./configs/configs_cifar.yml")
    parser.add_argument(
        "--results_dir", type=str, default="./eval_logs/",
    )
    parser.add_argument("--exp-name", type=str, default="temp")

    parser.add_argument("--arch", type=str, default="cnnSmall")
    parser.add_argument("--num-classes", type=int, default=10)
    
    parser.add_argument("--evaluator", type=str, choices=("base", "adv"), default="base")
    parser.add_argument("--eval-attack", type=str, choices=("linf", "l2"), default="linf")
    
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-dir", type=str, default="./datasets/")
    parser.add_argument("--in-channel", type=int, default=3)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=256)
    
    parser.add_argument("--autoattack", action="store_true", default=False, help="Use AutoAttack instead of PGD")
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--ckpt", type=str, help="checkpoint path")
    parser.add_argument("--seed", type=int, default=12345)

    args = update_args(parser.parse_args())
    assert args.ckpt, "Must provide a checkpint for evaluation"
    
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    results_file = os.path.join(args.results_dir, args.exp_name + ".txt")

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(results_file, "a"))
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    model = torch.nn.DataParallel(models.__dict__[args.arch](in_channel=args.in_channel, num_classes=args.num_classes)).cuda()
    
    # load checkpoint
    ckpt_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(ckpt_dict)
    
    # Dataloader
    train_loader, test_loader, _ = data.__dict__[args.dataset](
        args.data_dir,
        normalize=args.normalize,
        batch_size=args.batch_size,
    )
    
    val = getattr(evaluators, args.evaluator)
    prec1, _ = val(model, "cuda:0", test_loader, nn.CrossEntropyLoss(), args.EvalAttack, 0, args)
    logger.info(f"validation accuracy = {prec1}")

if __name__ == "__main__":
    main()
