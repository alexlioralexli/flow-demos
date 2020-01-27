import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from scipy.stats import norm
import copy
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from deepul_helper.data import load_flow_demo_1, generate_1d_flow_data, NumpyDataset
from deepul_helper.visualize import plot_hist, plot_train_curves


SEED = 10
np.random.seed(SEED)
torch.manual_seed(SEED)

n_train, n_test = 2000, 1000
loader_args = dict(batch_size=128, shuffle=True)
train_loader, test_loader = load_flow_demo_1(n_train, n_test, loader_args, visualize=True, train_only=False)


def train(model, train_loader, optimizer):
    model.train()
    for x in train_loader:
        loss = model.nll(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            loss = model.nll(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss.item()

def train_epochs(model, train_loader, test_loader, train_args):
    epochs, lr = train_args['epochs'], train_args['lr']
    plot = train_args.get('plot', True)
    plot_frequency = train_args.get('plot_frequency', 5)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        train(model, train_loader, optimizer)
        train_loss = eval_loss(model, train_loader)
        train_losses.append(train_loss)

        if test_loader is not None:
            test_loss = eval_loss(model, test_loader)
            test_losses.append(test_loss)

        if plot and epoch % plot_frequency == 0:
            density = model.get_density()
            plt.figure()
            plt.plot(density[0], density[1])
            plt.title(f'Epoch {epoch}')

            # plot_hist(train_loader.dataset.array, bins=d,
                      # title=f'Epoch {epoch}', density=model.get_density())
    if plot:
        plot_train_curves(epochs, train_losses, test_losses, title='Training Curve')
    return train_losses, test_losses





class MixtureCDFFlow(nn.Module):
    def __init__(self,
                 base_dist='uniform',
                 mixture_dist='gaussian',
                 n_components=4):
        super().__init__()
        self.composition = False
        if base_dist == 'uniform':
            self.base_dist = Uniform(0.0, 1.0)
        elif base_dist == 'beta':
            self.base_dist = Beta(5, 5)
        else:
            raise NotImplementedError

        self.loc = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_scale = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        if mixture_dist == 'gaussian':
            self.mixture_dist = Normal  # (self.loc, self.log_scale.exp())
        elif mixture_dist == 'logistic':
            raise NotImplementedError
        self.n_components = n_components

    def flow(self, x):
        # z = cdf of x
        weights = F.softmax(self.weight_logits, dim=0).unsqueeze(0).repeat(x.shape[0], 1)
        z = (self.mixture_dist(self.loc, self.log_scale.exp()).cdf(
            x.unsqueeze(1).repeat(1, self.n_components)) * weights).sum(dim=1)

        # log_det = log dz/dx = log pdf(x)
        log_det = (self.mixture_dist(self.loc, self.log_scale.exp()).log_prob(
            x.unsqueeze(1).repeat(1, self.n_components)).exp() * weights).sum(dim=1).log()

        return z, log_det

    def log_prob(self, x):
        z, log_det = self.flow(x)
        return self.base_dist.log_prob(z) + log_det

    # Compute loss as negative log-likelihood
    def nll(self, x):
        return - self.log_prob(x).mean()

    def get_density(self):
        x = np.linspace(-3, 3, 1000)
        with torch.no_grad():
            y = self.log_prob(torch.tensor(x)).exp().numpy()
        return x, y

def visualize_demo1_flow(train_loader, initial_flow, final_flow):
    plt.figure(figsize=(10,5))
    train_data = torch.cat(list(train_loader))

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

cdf_flow_model = MixtureCDFFlow(base_dist='uniform', mixture_dist='gaussian', n_components=5)
cdf_flow_model_old = copy.deepcopy(cdf_flow_model)
train_epochs(cdf_flow_model, train_loader, test_loader, dict(epochs=50, lr=5e-3))
visualize_demo1_flow(train_loader, cdf_flow_model_old, cdf_flow_model)

beta_cdf_flow_model = MixtureCDFFlow(base_dist='beta', mixture_dist='gaussian', n_components=5)
beta_cdf_flow_model_old = copy.deepcopy(beta_cdf_flow_model)
train_epochs(beta_cdf_flow_model, train_loader, test_loader, dict(epochs=250, lr=1e-2))
visualize_demo1_flow(train_loader, beta_cdf_flow_model_old, beta_cdf_flow_model)