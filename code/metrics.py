import os
import random
import torch
import requests
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from data_wrapper import TrainCIFAR10
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Metrics:

    def accuracy(self, net, loader):
        """Return accuracy on a dataset given by the data loader."""
        correct = 0
        total = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct / total


