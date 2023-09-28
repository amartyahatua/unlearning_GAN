import os
import random
import torch
import requests
from helper import *
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from data_wrapper import TrainCIFAR10

from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet18
from metrics import Metrics

print(torch.__version__)
class LoadData:
    def __init__(self):
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        RNG = torch.Generator().manual_seed(42)
        cudnn.benchmark = True

        # set manual seed to a constant get a consistent output
        manualSeed = random.randint(1, 10000)

        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        # loading the dataset
        dataset = dset.CIFAR10(root="./data", download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

        # train_set = dset.CIFAR10(root="./data", train=True, download=True, transform=normalize)
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        # we split held out data_ into test and validation set
        held_out = dset.CIFAR10(
            root="./data", train=False, download=True, transform=normalize
        )

        local_path = "forget_idx.npy"
        if not os.path.exists(local_path):
            response = requests.get(
                "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/cifar10/" + local_path
            )
            open(local_path, "wb").write(response.content)
        forget_idx = np.load(local_path)

        # construct indices of retain from those of the forget set
        forget_mask = np.zeros(len(dataset.targets), dtype=bool)
        forget_mask[forget_idx] = True
        retain_idx = np.arange(forget_mask.size)[~forget_mask]

        test_set, val_set = torch.utils.data.random_split(held_out, [5000, 5000], generator=RNG)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)

        # split train set into a forget and a retain set
        self.forget_set = torch.utils.data.Subset(dataset, forget_idx)
        self.retain_set = torch.utils.data.Subset(dataset, retain_idx)

        self.forget_loader = torch.utils.data.DataLoader(self.forget_set, batch_size=128, shuffle=True)
        self.retain_loader = torch.utils.data.DataLoader(self.retain_set, batch_size=128, shuffle=True, generator=RNG)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())

local_path = "weights_resnet18_cifar10.pth"
if not os.path.exists(local_path):
    response = requests.get(
        "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth"
    )
    open(local_path, "wb").write(response.content)

weights_pretrained = torch.load(local_path, map_location=DEVICE)

# load model with pre-trained weights
model = resnet18(weights=None, num_classes=10)
model.load_state_dict(weights_pretrained)
model.to(DEVICE)
model.eval()

metrics = Metrics()
data = LoadData()


def unlearning(net, retain, forget, validation):
    """Unlearning by fine-tuning.

    Fine-tuning is a very simple algorithm that trains using only
    the retain set.

    Args:
      net : nn.Module.
        pre-trained model to use as base of unlearning.
      retain : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
      forget : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget. This method doesn't
        make use of the forget set.
      validation : torch.utils.data.DataLoader.
        Dataset loader for access to the validation set. This method doesn't
        make use of the validation set.
    Returns:
      net : updated model
    """
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * 3 + 2)
    net.train()
    # training
    for _ in tqdm(range(epochs)):
        for inputs, targets in forget:
            #targets = 9 - targets  # invert the target in forget set - taking complement
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
    for _ in tqdm(range(epochs)):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
    # finalize
    net.eval()
    return net

synthetic_retain_data(data.retain_loader)
ft_model = resnet18(weights=None, num_classes=10)
ft_model.load_state_dict(weights_pretrained)
ft_model.to(DEVICE)

# Execute the unlearing routine. This might take a few minutes.
# If run on colab, be sure to be running it on  an instance with GPUs

ft_model = unlearning(ft_model, data.retain_loader, data.forget_loader, data.test_loader)
print(f"Retain set accuracy: {100.0 * metrics.accuracy(ft_model, data.retain_loader):0.1f}%")
print(f"Forget set accuracy: {100.0 * metrics.accuracy(ft_model, data.forget_loader):0.1f}%")
print(f"  Test set accuracy: {100.0 * metrics.accuracy(ft_model, data.test_loader):0.1f}%")


for i in range(5):
    ft_model = unlearning(ft_model, synthetic_retain_data(), synthetic_forget_data(), data.test_loader)
    print(f"Retain set accuracy: {100.0 * metrics.accuracy(ft_model, data.retain_loader):0.1f}%")
    print(f"Forget set accuracy: {100.0 * metrics.accuracy(ft_model, data.forget_loader):0.1f}%")
    print(f"  Test set accuracy: {100.0 * metrics.accuracy(ft_model, data.test_loader):0.1f}%")

print(f"Train set accuracy: {100.0 * metrics.accuracy(model, data.train_loader):0.1f}%")
print(f"Test set accuracy: {100.0 * metrics.accuracy(model, data.test_loader):0.1f}%")
