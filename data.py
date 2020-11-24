import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler


def mnist(data_dir, batch_size=128, mode="org", normalize=True, norm_layer=None):
    """
        mode: org | base
    """
    transform_train = [transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)), transforms.ToTensor()]
    transform_test = [transforms.ToTensor()]
    
    if mode == "org":
        None
    elif mode == "base":
        transform_train = [transforms.ToTensor()]
    else:
        raise ValueError(f"{mode} mode not supported")
    
    if norm_layer is None:
        norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    if normalize:
            transform_train.append(norm_layer)
            transform_test.append(norm_layer)
    
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
    
    trainset = datasets.MNIST(
            root=os.path.join(data_dir, "mnist"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.MNIST(
            root=os.path.join(data_dir, "mnist"),
            train=False,
            download=True,
            transform=transform_test,
        )
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, norm_layer


def fmnist(data_dir, batch_size=128, mode="org", normalize=True, norm_layer=None):
    """
        mode: org | base
    """
    transform_train = [transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)), transforms.ToTensor()]
    transform_test = [transforms.ToTensor()]
    
    if mode == "org":
        None
    elif mode == "base":
        transform_train = [transforms.ToTensor()]
    else:
        raise ValueError(f"{mode} mode not supported")
    
    if norm_layer is None:
        norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    if normalize:
            transform_train.append(norm_layer)
            transform_test.append(norm_layer)
    
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
    
    trainset = datasets.FashionMNIST(
            root=os.path.join(data_dir, "mnist"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.FashionMNIST(
            root=os.path.join(data_dir, "mnist"),
            train=False,
            download=True,
            transform=transform_test,
        )
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, norm_layer


def cifar10(data_dir, batch_size=128, mode="org", normalize=True, norm_layer=None, size=32):
    """
        mode: org | base
    """
    transform_train = [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    transform_test = [transforms.ToTensor()]
    
    if mode == "org":
        None
    elif mode == "base":
        transform_train = [transforms.ToTensor()]
    else:
        raise ValueError(f"{mode} mode not supported")
    
    if norm_layer is None:
        norm_layer = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    if normalize:
            transform_train.append(norm_layer)
            transform_test.append(norm_layer)
    
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
    
    trainset = datasets.CIFAR10(
            root=os.path.join(data_dir, "cifar10"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.CIFAR10(
            root=os.path.join(data_dir, "cifar10"),
            train=False,
            download=True,
            transform=transform_test,
        )
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, norm_layer