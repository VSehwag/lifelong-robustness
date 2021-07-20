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
from attack_vectors import get_attack_vector
from utils import *

import sklearn
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

import CKA

def get_images(imgs):
    images = []
    for x,y in enumerate(imgs[1]):
        images.append(y)
    
    image_data = None
    image_labels = None
    for batch in range(len(images)):
        if image_data is None:
            image_data = images[batch][0]
            image_labels = images[batch][1]
        else:
            image_data = torch.cat((image_data, images[batch][0]), 0)
            image_labels = torch.cat((image_labels, images[batch][1]), 0)
    image_data = image_data.to(device)
    image_labels = image_labels.to(device)
    perm = torch.randperm(image_data.shape[0])

    return image_data[perm], image_labels[perm]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, default='./trained_models/cifar10_ResNet18_base/trial_0/checkpoint/checkpoint.pth.tar', help='Path to first model')
    parser.add_argument('--config1', type=str, default='', help='Attack configuration for model 1')
    parser.add_argument('--model2', type=str, default='./trained_models/cifar10_ResNet18_l2_0.5/trial_0/checkpoint/checkpoint.pth.tar', help='Path to second model')
    parser.add_argument('--config2', type=str, default='', help='Attack configuration for model 2')
    parser.add_argument('--num_test', '-n', type=int, default=1000, help='Number of test points to use')
    parser.add_argument('--dataset', '-d', default='cifar', choices=['cifar', 'imagenet'], help='Dataset used to generate activations')
    parser.add_argument('--attack', '-a', type=str, choices=['none', 'l2'], default='none', help='Attack type')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    args = parser.parse_args()

    if args.dataset == 'cifar':
        images = data.cifar10('./datasets', normalize=False)
    elif args.dataset == 'imagenet':
        images = data.imagenet('./datasets/imagenette2', normalize=False)

    in_channel = 3
    num_classes = 10
    device = 'cuda:' + str(args.gpu)

    print('Loading models...')
    m1 = torch.nn.DataParallel(models.ResNet18(in_channel=in_channel, num_classes=num_classes)).cuda().eval()
    ckpt_dict = torch.load(args.model1, map_location="cpu")["state_dict"]
    m1.load_state_dict(ckpt_dict)

    m2 = torch.nn.DataParallel(models.ResNet18(in_channel=in_channel, num_classes=num_classes)).cuda().eval()
    ckpt_dict = torch.load(args.model2, map_location="cpu")["state_dict"]
    m2.load_state_dict(ckpt_dict)

    image_x, image_y = get_images(images)
    image_x = image_x[:args.num_test]
    image_y = image_y[:args.num_test]

    print('Generating images...')
    if args.attack == 'none':
        model_1_image_x = image_x
        model_2_image_x = image_x
    else:
        with open(args.config1, 'r') as f:
            attack_config1 = EasyDict(yaml.load(f))
        
        with open(args.config2, 'r') as f:
            attack_config2 = EasyDict(yaml.load(f))

        attack1 = get_attack_vector(args.attack, attack_config1)
        attack2 = get_attack_vector(args.attack, attack_config2)

        model_1_image_x = attack1(m1, image_x, image_y, torch.ones(args.num_test))[0]
        model_2_image_x = attack2(m2, image_x, image_y, torch.ones(args.num_test))[0]

    print('Computing CKA...')
    preds1 = m1(model_1_image_x, all=True)
    acts1 = preds1[1]['layer4']['block_1']['conv2'].cpu().detach().numpy().reshape((args.num_test,512))
    preds2 = m2(model_2_image_x, all=True)
    acts2 = preds2[1]['layer4']['block_1']['conv2'].cpu().detach().numpy().reshape((args.num_test,512))
    
    print("CKA:", CKA.feature_space_linear_cka(acts1, acts2))