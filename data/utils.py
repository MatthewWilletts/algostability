import os

import numpy as np

# https://www.youtube.com/watch?v=DuSqffoDojM -> 1:29:03
# -> https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4

from typing import Any, Callable, Optional

import torch
from torchvision.datasets import MNIST, SVHN, FashionMNIST, CIFAR10


def to_one_hot(x, m=None):
    "batch one hot"
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
    return xoh


def one_hot_encode(labels, n_labels=10):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels]).astype(np.float32)
    y[range(labels.size), labels] = 1

    return y


def single_one_hot_encode(label, n_labels=10):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert label >= 0 and label < n_labels

    y = np.zeros([n_labels]).astype(np.float32)
    y[label] = 1

    return y


def single_one_hot_encode_rev(label, n_labels=10, start_label=0):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """
    assert label >= start_label and label < n_labels
    y = np.zeros([n_labels - start_label]).astype(np.float32)
    y[label - start_label] = 1
    return y


mnist_one_hot_transform = lambda label: single_one_hot_encode(label, n_labels=10)
contrastive_one_hot_transform = lambda label: single_one_hot_encode(label, n_labels=2)


def make_dir(dir_name):
    if dir_name[-1] != '/':
        dir_name += '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def make_file(file_name):
    if not os.path.exists(file_name):
        open(file_name, 'a').close()
    return file_name


class Fast_MNIST(MNIST):
    """
    Source as modified from: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
    """
    def __init__(self, *args, **kwargs):
        # deleting device key from kwargs (as cannot be passed to MNIST mother class)
        device = kwargs['device']
        del kwargs['device']
        super().__init__(*args, **kwargs)

        # Insert channel dim and scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class Fast_Fashion_MNIST(FashionMNIST):
    """
    Source as modified from: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
    """
    def __init__(self, *args, **kwargs):
        # deleting device key from kwargs (as cannot be passed to MNIST mother class)
        device = kwargs['device']
        del kwargs['device']
        super().__init__(*args, **kwargs)

        # Insert channel dim and scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class Fast_SVHN(SVHN):
    """
    Source as modified from: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
    """
    def __init__(self, *args, **kwargs):
        # deleting device key from kwargs (as cannot be passed to SVHN mother class)
        device = kwargs['device']
        del kwargs['device']
        super().__init__(*args, **kwargs)

        # convert numpy to torch
        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)

        # # scale data to [0,1]
        # self.data = self.data.float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        # attribute is self.labels, see https://pytorch.org/docs/stable/_modules/torchvision/datasets/svhn.html#SVHN
        self.data, self.labels = self.data.to(device), self.labels.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.data[index].float().div(255), self.labels[index]

        return img, label


class Fast_CIFAR10(CIFAR10):
    """
    Source as modified from: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
    """
    def __init__(self, *args, **kwargs):
        # deleting device key from kwargs (as cannot be passed to SVHN mother class)
        device = kwargs['device']
        del kwargs['device']
        super().__init__(*args, **kwargs)

        # convert numpy to torch
        self.data = torch.tensor(self.data).permute(0, 3, 1, 2)
        self.targets = torch.tensor(self.targets)
        # # scale data to [0,1]
        # self.data = self.data.float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        # attribute is self.labels, see https://pytorch.org/docs/stable/_modules/torchvision/datasets/svhn.html#SVHN
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, targets = self.data[index].float().div(255), self.targets[index]

        return img, targets

    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, label = self.data[index], self.targets[index]

    #     return img, label


def get_good_dims(x_train, threshold=0.1):
    stds = np.std(x_train, axis=0)
    good_dims = np.where(stds > threshold)[0]
    return good_dims


class NaNinLoss(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, message="NaN found in loss"):
        self.message = message
        super().__init__(self.message)