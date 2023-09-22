import os
import numpy as np
import torch
import requests
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from sklearn import decomposition
from matplotlib import pyplot as plt
#from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

RNG = torch.Generator().manual_seed(42)


class TrainSetMNIST(data.Dataset):
    def __init__(self, image_path='./data_', image_size=28, num_images=5000,
                 p_class=1, n_class=0, p_ratio=0.5):
        self.IMAGE_PATH = image_path
        self.IMAGE_SIZE = image_size
        self.NUM_IMAGES = num_images
        self.P_CLASS = p_class
        self.N_CLASS = n_class
        self.P_RATIO = p_ratio
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_images, train_labels = self.select_train_set()
        self.train_images = train_images
        self.train_labels = train_labels

    def __getitem__(self, index):
        train_image = self.train_images[index]
        train_label = self.train_labels[index]
        if self.transform is not None:
            train_image = self.transform(train_image)
        return train_image, train_label

    def __len__(self):
        return len(self.train_images)

    def select_train_set(self):
        train_set = dsets.MNIST(root=self.IMAGE_PATH, train=True, download=True)
        train_images = train_set.train_data.numpy()
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
        train_labels = train_set.train_labels.numpy()

        p_index = train_labels == self.P_CLASS
        p_train_images = train_images[p_index]
        if len(p_train_images) < self.NUM_IMAGES:
            ValueError("p images for training is not enough")
        p_train_images = p_train_images[:self.NUM_IMAGES]
        np.random.shuffle(p_train_images)
        num_p_train_images = int(len(p_train_images) * self.P_RATIO)
        p_train_images = p_train_images[:num_p_train_images].copy()
        p_train_labels = np.ones(num_p_train_images, dtype='int8')

        n_index = train_labels == self.N_CLASS
        n_train_images = train_images[n_index]
        if len(n_train_images) < self.NUM_IMAGES:
            ValueError("n images for training is not enough")
        n_train_images = n_train_images[:self.NUM_IMAGES]
        n_train_labels = np.zeros(n_train_images.shape[0], dtype='int8')
        train_images = np.concatenate((p_train_images, n_train_images), axis=0)
        train_labels = np.concatenate((p_train_labels, n_train_labels), axis=0)
        state = np.random.get_state()
        np.random.shuffle(train_images)
        np.random.set_state(state)
        np.random.shuffle(train_labels)
        return train_images, train_labels


class TestSetMNIST(data.Dataset):
    def __init__(self, image_path='./data_', image_size=28, p_class=0, n_class=3):
        self.IMAGE_PATH = image_path
        self.IMAGE_SIZE = image_size
        self.P_CLASS = p_class
        self.N_CLASS = n_class
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_images, test_labels = self.select_test_set()
        self.test_images = test_images
        self.test_labels = test_labels

    def __getitem__(self, index):
        test_image = self.test_images[index]
        test_label = self.test_labels[index]
        if self.transform is not None:
            test_image = self.transform(test_image)
        return test_image, test_label

    def __len__(self):
        return len(self.test_images)

    def select_test_set(self):
        test_set = dsets.MNIST(root=self.IMAGE_PATH, train=False, download=False)
        test_images = test_set.test_data.numpy()
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
        test_labels = test_set.test_labels.numpy()

        p_index = test_labels == self.P_CLASS
        p_test_images = test_images[p_index]
        p_test_labels = np.ones(p_test_images.shape[0], dtype='int8')
        n_index = test_labels == self.N_CLASS
        n_test_images = test_images[n_index]
        n_test_labels = np.zeros(n_test_images.shape[0], dtype='int8')
        test_images = np.concatenate((p_test_images, n_test_images), axis=0)
        test_labels = np.concatenate((p_test_labels, n_test_labels), axis=0)

        state = np.random.get_state()
        np.random.shuffle(test_images)
        np.random.set_state(state)
        np.random.shuffle(test_labels)
        return test_images, test_labels


class PUTrainSetMNIST(data.Dataset):
    def __init__(self, image_path='./data_', image_size=28, num_images=5000,
                 p_class=1, n_class=0, p_ratio=0.0, verbose=False, sample_path='./samples'):
        self.IMAGE_PATH = image_path
        self.IMAGE_SIZE = image_size
        self.NUM_IMAGES = num_images
        self.P_CLASS = p_class
        self.N_CLASS = n_class
        self.P_RATIO = p_ratio
        self.VERBOSE = verbose
        self.SAMPLE_PATH = sample_path
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        p_train_images, u_train_images = self.select_pu_train_set()
        self.p_train_images = p_train_images
        self.u_train_images = u_train_images

    def __getitem__(self, index):
        p_train_image = self.p_train_images[index]
        u_train_image = self.u_train_images[index]
        if self.transform is not None:
            p_train_image = self.transform(p_train_image)
            u_train_image = self.transform(u_train_image)
        return p_train_image, u_train_image

    def __len__(self):
        return len(self.p_train_images)

    def select_pu_train_set(self):
        train_set = dsets.MNIST(root=self.IMAGE_PATH, train=True, download=True)
        train_images = train_set.train_data.numpy()
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
        train_labels = train_set.train_labels.numpy()

        num_repeats = int(round((2 - self.P_RATIO) / self.P_RATIO))
        p_index = train_labels == self.P_CLASS
        p_images = train_images[p_index]
        if len(p_images) < self.NUM_IMAGES:
            ValueError("number of p images for training is not enough")
        p_images = p_images[:self.NUM_IMAGES]
        np.random.shuffle(p_images)
        num_p_train_images = int(len(p_images) * self.P_RATIO)
        p_train_images = p_images[:num_p_train_images].copy()
        if self.VERBOSE:
            p_samples = p_train_images.copy()
            p_samples = torch.from_numpy(p_samples)
            p_samples = p_samples.permute(0, 3, 1, 2)
            p_samples = p_samples.view(p_samples.size(0), 1, self.IMAGE_SIZE, self.IMAGE_SIZE)
            p_samples = torch.tensor(p_samples, dtype=torch.int32) * 255

            # torchvision.utils.save_image(p_samples,
            #                              os.path.join(self.SAMPLE_PATH, 'p_train_images.png'), nrow=20)
        u_train_images = p_images[num_p_train_images:].copy()
        p_train_images = np.concatenate(([p_train_images for _ in range(num_repeats)]), axis=0)

        n_index = train_labels == self.N_CLASS
        n_images = train_images[n_index]
        if len(n_images) < self.NUM_IMAGES:
            ValueError("number of p images for training is not enough")
        n_images = n_images[:self.NUM_IMAGES]
        u_train_images = np.concatenate((u_train_images, n_images), axis=0)
        return p_train_images, u_train_images


class TrainCIFAR10():
    def __init__(self):
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.normalize
        )
        self.train_loader = DataLoader(self.train_set, batch_size=128, shuffle=True, num_workers=2)

        # we split held out data_ into test and validation set
        self.held_out = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.normalize
        )

        self.test_set, self.val_set = torch.utils.data.random_split(self.held_out, [5000, 5000], generator=RNG)
        self.test_loader = DataLoader(self.test_set, batch_size=128, shuffle=False, num_workers=2)
        self.val_loader = DataLoader(self.val_set, batch_size=128, shuffle=False, num_workers=2)

        # download the forget and retain index split
        local_path = "forget_idx.npy"
        if not os.path.exists(local_path):
            response = requests.get(
                "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/cifar10/" + local_path
            )
            open(local_path, "wb").write(response.content)
        forget_idx = np.load(local_path)

        # construct indices of retain from those of the forget set
        forget_mask = np.zeros(len(self.train_set.targets), dtype=bool)
        forget_mask[forget_idx] = True
        retain_idx = np.arange(forget_mask.size)[~forget_mask]

        # split train set into a forget and a retain set
        self.forget_set = torch.utils.data.Subset(self.train_set, forget_idx)
        self.retain_set = torch.utils.data.Subset(self.train_set, retain_idx)

        self.forget_loader = torch.utils.data.DataLoader(
            self.forget_set, batch_size=128, shuffle=True, num_workers=2
        )
        self.retain_loader = torch.utils.data.DataLoader(
            self.retain_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
        )
