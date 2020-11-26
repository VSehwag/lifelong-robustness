import os
import sys
import numpy as np
import math
import time
import yaml
from easydict import EasyDict
import shutil, errno
from distutils.dir_util import copy_tree
import sklearn.metrics as skm
from sklearn.covariance import ledoit_wolf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets


#### logging ####
def save_checkpoint(state, is_best, results_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(results_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(results_dir, filename),
            os.path.join(results_dir, "model_best.pth.tar"),
        )


def create_subdirs(sub_dir):
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, "checkpoint"))


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)

def update_args(args):
    with open(args.configs) as f:
        new_args = EasyDict(yaml.load(f))
    
    for k, v in vars(args).items():
        new_args[k] = v
    
    return new_args
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


#### evaluation ####
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_features(model, dataloader, max_images=10**10, verbose=False):
    features, labels = [], []
    total = 0

    for index, (img, label) in enumerate(dataloader):

        if total > max_images:
            break
        
        img, label = img.cuda(), label.cuda()

        features += list(model(img).data.cpu().numpy())
        labels += list(label.data.cpu().numpy())

        if verbose and not index % 50:
            print(index)
            
        total += len(img)

    return np.array(features), np.array(labels)


#### Dataloaders ####
def readloader(dataloader):
    images = []
    labels = []
    for img, label in dataloader:
        images.append(img)
        labels.append(label)
    return torch.cat(images), torch.cat(labels)


def unnormalize(x, norm_layer):
    m, s = torch.tensor(norm_layer.mean).view(1, 3, 1, 1), torch.tensor(norm_layer.std).view(1, 3, 1, 1)
    return x * s + m


def display_vectors(images):
    if len(images) > 64:
        images = images[:64]
    if torch.is_tensor(images):
        images = np.transpose(images.cpu().numpy(), (0, 2, 3, 1))

    d = int(math.sqrt(len(images)))
    plt.figure(figsize=(8, 8))
    image = np.concatenate(
        [
            np.concatenate([images[d * i + j] for j in range(d)], axis=0)
            for i in range(d)
        ],
        axis=1,
    )
    if image.shape[-1] == 1:
        plt.imshow(image[:, :, 0], cmap="gray")
    else:
        plt.imshow(image)