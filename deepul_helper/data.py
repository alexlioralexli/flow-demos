import pickle
from os.path import join, exists
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from .utils import download_file_from_google_drive

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
    train_data, test_data = generate_1d_flow_data(n_train), generate_1d_flow_data(n_test)

    if visualize:
        from .visualize import plot_hist

        plot_hist(train_data, bins=50, title='Train Set')
        if not train_only:
            plot_hist(test_data, bins=d, title='Test Set')

    train_dset, test_dset = NumpyDataset(train_data), NumpyDataset(test_data)
    train_loader, test_loader = data.DataLoader(train_dset, **loader_args), data.DataLoader(test_dset, **loader_args)

    if train_only:
        return train_loader
    return train_loader, test_loader



def load_demo_2(n_train, n_test, d, loader_args, visualize=True):
    from PIL import Image
    from urllib.request import urlopen
    import io

    fd = urlopen('https://wilson1yan.github.io/images/smiley.jpg')
    image_file = io.BytesIO(fd.read())
    im = Image.open(image_file).resize((d, d)).convert('L')
    im = np.array(im).astype('float32')
    dist = im / im.sum()

    if visualize:
        plt.figure()
        plt.title('True Distribution')
        plt.imshow(dist)
        plt.show()

    dist = dist[::-1]
    train_data, test_data = generate_2d_data(n_train, dist), generate_2d_data(n_test, dist)

    if visualize:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_title('Train Set')
        ax1.hist2d(train_data[:, 1], train_data[: ,0], bins=d)
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x0')
        plt.show()

    train_dset, test_dset = NumpyDataset(train_data), NumpyDataset(test_data)
    train_loader, test_loader = data.DataLoader(train_dset, **loader_args), data.DataLoader(test_dset, **loader_args)

    return train_loader, test_loader


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