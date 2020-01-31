import pickle
from os.path import join, exists
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from scipy.stats import norm
from .utils import download_file_from_google_drive
from sklearn.datasets import make_moons

def generate_1d_data(n, d):
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(n)
    b = 0.8 + 0.05 * rand.randn(n)
    mask = rand.rand(n) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    return np.digitize(samples, np.linspace(0.0, 1.0, d)).astype('float32')


def generate_2d_data(n, dist):
    import itertools
    d1, d2 = dist.shape
    pairs = list(itertools.product(range(d1), range(d2)))
    idxs = np.random.choice(len(pairs), size=n, replace=True, p=dist.reshape(-1))
    samples = [pairs[i] for i in idxs]

    return np.array(samples).astype('float32')

def generate_1d_flow_data(n):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n//2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n//2,))
    return np.concatenate([gaussian1, gaussian2])


def load_flow_demo_1(n_train, n_test, loader_args, visualize=True, train_only=False):
    # 1d distribution, mixture of two gaussians
    train_data, test_data = generate_1d_flow_data(n_train), generate_1d_flow_data(n_test)

    if visualize:
        plt.figure()
        x = np.linspace(-3, 3, num=100)
        densities = 0.5 * norm.pdf(x, loc=-1, scale=0.25) + 0.5 * norm.pdf(x, loc=0.5, scale=0.5)
        plt.figure()
        plt.plot(x, densities)
        plt.show()
        plt.figure()
        plt.hist(train_data, bins=50)
        # plot_hist(train_data, bins=50, title='Train Set')
        plt.show()

    train_dset, test_dset = NumpyDataset(train_data), NumpyDataset(test_data)
    train_loader, test_loader = data.DataLoader(train_dset, **loader_args), data.DataLoader(test_dset, **loader_args)

    if train_only:
        return train_loader
    return train_loader, test_loader

def load_flow_demo_2(n_train, n_test, loader_args, visualize=True, train_only=False, distribution='uniform'):
    if distribution == 'uniform':
        train_data = np.random.uniform(-2, 2, (n_train,))
        test_data = np.random.uniform(-2, 2, (n_test,))
        xs = np.linspace(-2, 2, 250)
        ys = np.ones_like(xs) * 0.25
    elif distribution == 'triangular':
        train_data = np.random.triangular(-2, -1.5, 2, (n_train,))
        test_data = np.random.triangular(-2, -1.5, 2, (n_test,))
        xs = np.linspace(-2, 2, 250)
        ys = np.zeros_like(xs)
        ys[xs < -1.5] = 2.0 + xs[xs < -1.5]
        ys[xs >= -1.5] = (2 - xs[xs >= -1.5]) / 7
        plt.plot(xs, ys)
    elif distribution == 'complex':
        xs = np.linspace(0, 1, 100)
        ys = np.tanh(np.sin(8 * np.pi * xs) + 3 * np.power(xs, 0.5))
        train_data = np.random.choice(np.linspace(-2, 2, 100), size=n_train, p=ys / ys.sum())
        test_data = np.random.choice(np.linspace(-2, 2, 100), size=n_test, p=ys / ys.sum())
        xs = np.linspace(-2, 2, 100)
        ys = ys / ys.sum() / 0.04
    else:
        raise NotImplementedError

    if visualize:
        plt.figure()
        plt.plot(xs, ys)
        plt.hist(train_data, bins=50, density=True)
        plt.title(distribution)
        plt.show()

    train_dset, test_dset = NumpyDataset(train_data), NumpyDataset(test_data)
    train_loader, test_loader = data.DataLoader(train_dset, **loader_args), data.DataLoader(test_dset, **loader_args)

    if train_only:
        return train_loader
    return train_loader, test_loader


def load_smiley_face(n):
    count = n
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
              -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]


def load_cross(n):
    pass

def load_half_moons(n):
    return make_moons(n_samples=n, noise=0.1)


def make_scatterplot(points, title=None, filename=None):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1)
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig("q1_{}.png".format(filename))


def load_flow_demo_3(n_train, n_test, loader_args, visualize=True, train_only=False, distribution='face'):
    if distribution == 'face':
        train_data, train_labels = load_smiley_face(n_train)
        test_data, test_labels = load_smiley_face(n_test)
    elif distribution == 'moons':
        train_data, train_labels = load_half_moons(n_train)
        test_data, test_labels = load_half_moons(n_test)
    else:
        raise NotImplementedError

    if visualize:
        plt.figure()
        plt.scatter(train_data[:, 0], train_data[:, 1], s=1, c=train_labels)
        plt.title(distribution)
        plt.show()

    train_dset, test_dset = NumpyDataset(train_data), NumpyDataset(test_data)
    train_loader, test_loader = data.DataLoader(train_dset, **loader_args), data.DataLoader(test_dset, **loader_args)

    if train_only:
        return train_loader
    return train_loader, test_loader, train_labels, test_labels


class NumpyDataset(data.Dataset):

    def __init__(self, array, transform=None):
        super().__init__()
        self.array = array
        self.transform = transform

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        x = self.array[index]
        if self.transform:
            x = self.transform(x)
        return x


class ColorMNIST(data.Dataset):

    def __init__(self, root, train=True):
        super().__init__()
        if not exists(root):
            os.makedirs(root)

        file_path = join(root, 'color_mnist.pkl')
        if not exists(file_path):
            download_file_from_google_drive(id='1Dquj__fnKLaGKO53jcXJ0R_WeVnHdYhc',
                                            destination=file_path)

        with open(file_path, 'rb') as f:
           data = pickle.load(f)

        if train:
            self.data = data['train']
        else:
            self.data = data['test']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = torch.FloatTensor(img).permute(2, 0, 1).contiguous()
        img /= 3
        return img