import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import pathlib
import re
import time
import datetime
import numpy as np
import torchattacks
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from art.defences.detector.poison import SpectralSignatureDefense, ActivationDefence
from art.estimators.classification.pytorch import PyTorchClassifier
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_poisoned_training_set, build_reverse_poisoned_training_set, build_clean_training_set, build_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch, reverse_train_model
from models import BadNet

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
parser.add_argument('--GPU_mode', action='store_true', help='Using GPUs')

args = parser.parse_args()

def get_activation(model, layer):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.clone().detach())
    
    handle = layer.register_forward_hook(hook_fn)
    return activations, handle

def compute_differences(model, clean_loader, trojan_loader, layer):
    # Capture activations
    clean_activations, handle_clean = get_activation(model, layer)
    trojan_activations, handle_trojan = get_activation(model, layer)

    # Run forward pass for clean data
    model.eval()
    with torch.no_grad():
        for data, _ in clean_loader:
            _ = model(data)
    
    clean_features = torch.cat(clean_activations, dim=0)

    # Run forward pass for Trojan data
    with torch.no_grad():
        for data, _ in trojan_loader:
            _ = model(data)

    trojan_features = torch.cat(trojan_activations, dim=0)

    # Compute differences
    differences = torch.abs(clean_features.mean(dim=0) - trojan_features.mean(dim=0))

    # Remove hooks
    handle_clean.remove()
    handle_trojan.remove()

    return differences

def prune_top_filters(model, layer, contributions, prune_ratio=0.08):

    # Total filters to prune
    num_filters = contributions.size(0)
    num_to_prune = int(num_filters * prune_ratio)

    # Identify filters to prune
    prune_indices = torch.topk(contributions, num_to_prune, largest=True)[1]

    # Zero out the weights and biases of the selected filters
    layer.weight.data[prune_indices] = 0
    if layer.bias is not None:
        layer.bias.data[prune_indices] = 0

    return model

def fine_tune_model(model, train_loader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def train_model():
    print("{}".format(args).replace(', ', ',\n'))

    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"

    print("\n# load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, dynamic=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, dynamic=True, mode="Test", args=args)

    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)

    basic_model_path = "./dynamic_checkpoints/badnet-%s-100ASR.pth" % args.dataset
    print("## Load model from : %s" % basic_model_path)
    model.load_state_dict(torch.load(basic_model_path), strict=True)

    last_conv_layer = model.fc1[0]
    differences = compute_differences(model, data_loader_val_clean, data_loader_val_poisoned, last_conv_layer)
    model = prune_top_filters(model, last_conv_layer, differences, prune_ratio=0.15)

    # clean_dataset_train, _ = build_clean_training_set(is_train=True, args=args)
    # data_loader_clean_train  = DataLoader(clean_dataset_train,   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # fine_tune_model(model, data_loader_clean_train, epochs=1, lr=0.001)
    test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
    print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
    print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")

train_model()