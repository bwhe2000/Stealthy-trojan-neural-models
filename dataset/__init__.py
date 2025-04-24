from .poisoned_dataset import CIFAR10Poison, MNISTPoison, CIFAR10ReversePoison, MNISTReversePoison
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, MNIST
import torch 
import os
from torch.utils.data import Subset

def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data

def build_poisoned_training_set(is_train, dynamic, args):
    transform, detransform = build_transform(args.dataset)
    print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        trainset = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform, dynamic=dynamic)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        trainset = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform, dynamic=dynamic)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes

def build_reverse_poisoned_training_set(is_train, dynamic, args): 
    transform, detransform = build_transform(args.dataset)
    print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        trainset = CIFAR10ReversePoison(args, args.data_path, train=is_train, download=True, transform=transform, dynamic=dynamic)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        trainset = MNISTReversePoison(args, args.data_path, train=is_train, download=True, transform=transform, dynamic=dynamic)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes

def build_clean_training_set(is_train, args):
    transform, detransform = build_transform(args.dataset)
    print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        trainset = CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        trainset = MNIST(args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes

def build_testset(is_train, dynamic, mode, args):
    transform, detransform = build_transform(args.dataset)
    print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        testset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform, dynamic=dynamic)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform, dynamic=dynamic)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes

    if(mode == 'Test'):
        indices = list(range(0, len(testset_clean) // 2))
        testset_clean = Subset(testset_clean, indices)
        testset_poisoned = Subset(testset_poisoned, indices)
    elif(mode == 'Verification'):
        indices = list(range(len(testset_clean) // 2, len(testset_clean)))
        testset_clean = Subset(testset_clean, indices)
        testset_poisoned = Subset(testset_poisoned, indices)
    else:
        return

    print("Number of the class = %d" % args.nb_classes)
    print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned

def build_transform(dataset):
    if dataset == "CIFAR10":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "MNIST":
        mean, std = (0.5,), (0.5,)
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image
    
    return transform, detransform
