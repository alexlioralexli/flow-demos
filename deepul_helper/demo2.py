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
import deepul_helper.pytorch_util as ptu

from demo1 import MixtureCDFFlow, train_epochs

class CompositionOfFlows(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def flow(self, x):
        z, log_det = x, 0
        for flow_module in self.flows:
            z, next_log_det = flow_module.flow(z)
            log_det += next_log_det
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.flow(x)
        return self.flows[-1].base_dist.log_prob(z) + log_det

    # Compute loss as negative log-likelihood
    def nll(self, x):
        return - self.log_prob(x).mean()

    def get_density(self):
        x = np.linspace(-3, 3, 1000)
        with torch.no_grad():
            y = self.log_prob(torch.tensor(x)).exp().numpy()
        return x, y

if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    n_train, n_test = 2000, 1000
    loader_args = dict(batch_size=128, shuffle=True)
    train_loader, test_loader = load_flow_demo_1(n_train, n_test, loader_args, visualize=True, train_only=False)
    losses = np.zeros([5, 5])
    for n_components in range(10, 11):
        for n_layers in range(10, 11):
            print(n_components, n_layers)
            composed_model = CompositionOfFlows([MixtureCDFFlow(n_components=n_components) for _ in range(n_layers)]).to(ptu.device)
            train_losses, test_losses = train_epochs(composed_model, train_loader, test_loader, dict(epochs=250, lr=1e-2, plot=False))
            losses[n_layers - 1, n_components - 1] = test_losses[-1]
    import IPython; IPython.embed()

