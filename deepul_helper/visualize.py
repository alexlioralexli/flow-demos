import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import make_grid


def plot_hist(data, bins=10, xlabel='x', ylabel='Probability', title='', density=None):
    bins = np.concatenate((np.arange(bins) - 0.5, [bins - 1 + 0.5]))

    plt.figure()
    plt.hist(data, bins=bins, density=True)

    if density:
        plt.plot(density[0], density[1], label='distribution')
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()




def plot_2d_dist(dist, title='Learned Distribution'):
    plt.figure()
    plt.imshow(dist)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x0')
    plt.show()


def plot_train_curves(epochs, train_losses, test_losses, title=''):
    x = np.linspace(0, epochs, len(train_losses))
    plt.figure()
    plt.plot(x, train_losses, label='train_loss')
    if test_losses:
        plt.plot(x, test_losses, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_batch(batch_tensor, nrow=8, title='', figsize=None):
    grid_img = make_grid(batch_tensor, nrow=nrow)
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def plot_1d_continuous_dist(density, xlabel='x', ylabel="Density", title=''):
    plt.figure()
    plt.plot(density[0], density[1], label='distribution')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def visualize_demo1_flow(train_loader, initial_flow, final_flow):
    plt.figure(figsize=(10,5))
    train_data = train_loader.dataset

    # before:
    plt.subplot(231)
    plt.hist(train_data, bins=50)
    plt.title('True Distribution of x')

    plt.subplot(232)
    x = torch.tensor(np.linspace(-3, 3, 200))
    z, _ = initial_flow.flow(x)
    z = z.detach().numpy()
    plt.plot(x, z)
    plt.title('Flow x -> z')

    plt.subplot(233)
    z_data, _ = initial_flow.flow(train_data)
    z_data = z_data.detach().numpy()
    plt.hist(z_data, bins=50)
    plt.title('Empirical Distribution of z')

    # after:
    plt.subplot(234)
    plt.hist(train_data, bins=50)
    plt.title('True Distribution of x')

    plt.subplot(235)
    x = torch.tensor(np.linspace(-3, 3, 200))
    z, _ = final_flow.flow(x)
    z = z.detach().numpy()
    plt.plot(x, z)
    plt.title('Flow x -> z')

    plt.subplot(236)
    z_data, _ = final_flow.flow(train_data)
    z_data = z_data.detach().numpy()
    plt.hist(z_data, bins=50)
    plt.title('Empirical Distribution of z')

    plt.tight_layout()