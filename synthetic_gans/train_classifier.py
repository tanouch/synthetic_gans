import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import os
import functools
import time
import argparse
from sklearn.manifold import TSNE
from torch.nn.utils import spectral_norm
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
from torch import Tensor
from torch.nn import Parameter
from torchvision import datasets
import torchvision.transforms as transforms

from generating_data import generate_z, generate_mixture_gaussian_batch, generate_mnist_batch
from plotting_functions import generate_grid_z, plot_gradient_of_the_generator, plot_data_points, plot_mnist_images, plot_precision_recall_curves
from defining_models import ResNet18, Generator, Discriminator, cal_gradient_penalty, Generator_mnist, Discriminator_mnist, Generator_DELI, Classifier_mnist
from getting_pr_score import get_pr_scores
from tools import convert_to_gpu

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type = int,  default = 0)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='initial learning rate for adam')
    parser.add_argument("--use_gpu", action='store_true', help='shuffle input data')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--steps_eval', type=int, default=100)

    parser.add_argument('--name_exp', type=str, default="default_exp")
    parser.add_argument('--dataset', default='mnist', type=str)

    opt = parser.parse_args()
    return opt

config = get_config()
print(config, flush = True)

if not os.path.exists(config.name_exp):
    os.makedirs(config.name_exp)

if config.dataset =='mnist':
    config.transform = transforms.ToTensor()
    config.train_data = datasets.MNIST(root='data', train=True, download=True, transform=config.transform)
    config.train_loader = torch.utils.data.DataLoader(config.train_data, batch_size=config.batch_size,  drop_last=True, shuffle=True)
    config.dataiter = iter(config.train_loader)
    config.gen_type = "conv"
    config.test_data = datasets.MNIST(root='data', train=False, download=True, transform=config.transform)
    config.test_loader = torch.utils.data.DataLoader(config.test_data, batch_size=config.batch_size)
    classifier = convert_to_gpu(Classifier_mnist(config), config)
elif config.dataset =='fashionMNIST':
    config.transform = transforms.ToTensor()
    config.train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=config.transform)
    config.train_loader = torch.utils.data.DataLoader(config.train_data, batch_size=config.batch_size,  drop_last=True, shuffle=True)
    config.dataiter = iter(config.train_loader)
    config.gen_type = "conv"
    config.test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=config.transform)
    config.test_loader = torch.utils.data.DataLoader(config.test_data, batch_size=config.batch_size)
    classifier = convert_to_gpu(Classifier_mnist(config), config)
elif config.dataset =='cifar10':
    config.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transforms.ToTensor()
    config.train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=config.transform_train)
    config.train_loader = torch.utils.data.DataLoader(config.train_data, batch_size=config.batch_size,  drop_last=True, shuffle=True)
    config.dataiter = iter(config.train_loader)
    config.gen_type = "conv"
    config.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    config.test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=config.transform_test)
    config.test_loader = torch.utils.data.DataLoader(config.test_data, batch_size=config.batch_size)
    #classifier = convert_to_gpu(ResNet18(), config)
    classifier = convert_to_gpu(Classifier_mnist(config), config)
else:
    print("Not the right dataset")
    sys.exit()

optimizer = torch.optim.Adam(classifier.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
CE = convert_to_gpu(nn.CrossEntropyLoss(), config)


def test(classifier, config):
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in config.test_loader:
            images = convert_to_gpu(images, config)
            labels = convert_to_gpu(labels, config)
            embedding, preds = classifier(images)
            test_loss += CE(preds, labels).item()  # sum up batch loss
            pred = preds.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(config.test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(config.test_loader.dataset),
        100. * correct / len(config.test_loader.dataset)))
    print('__________', flush = True)

for s in range(config.steps):
    classifier.train()
    images, labels = generate_mnist_batch(num_points=config.batch_size, config=config)
    images, labels = convert_to_gpu(images, config), convert_to_gpu(labels, config)
    embedding, preds = classifier(images)
    loss = CE(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if s % config.steps_eval == 0 and s != 0:
        print('Step: ',s)
        test(classifier, config)
path = './' + config.name_exp + '/' + 'classifier.pth'
torch.save(classifier.state_dict(), path)

print('test classifier reload')
if config.dataset == 'mnist':
    classifier_bis = convert_to_gpu(Classifier_mnist(config), config)
elif config.dataset == 'fashionMNIST':
    classifier_bis = convert_to_gpu(Classifier_mnist(config), config)
elif config.dataset == 'cifar10':
    classifier_bis = convert_to_gpu(Classifier_mnist(config), config)
classifier_bis.load_state_dict(torch.load(path))
test(classifier_bis, config)
