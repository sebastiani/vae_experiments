import torch
import torch.nn as nn
from torch.autograd import Variable


class GaussianMLP_VAE(nn.Module):
    def __init__(self, config):
        super(GaussianMLP_VAE, self).__init__()
        n_dims = config['input_dim']
        hidden_dim = config['hidden_dim']
        z_dim = config['z_dim']
        batch_size = config['batch_size']
        self.h1, self.mu1, self.var1 = self.buildEncoder(n_dims, hidden_dim, z_dim)
        self.h2, self.mu2, self.var2 = self.buildDecoder(hidden_dim, n_dims, z_dim)
        self.sample_z, self.sample_x = self.buildDist(z_dim, n_dims, batch_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z, z_mu, z_logvar = self.encode(x)
        x = self.decode(z)
        return x, z_mu, z_logvar

    def encode(self, x):
        h = self.relu(self.h1(x))
        z_mu = self.mu1(h)
        z_logvar = self.var2(h)

        z = self.sample_z(z_mu, z_logvar)
        return z, z_mu, z_logvar

    def decode(self, z):
        h = self.h2(z)
        x_mu = self.mu2(h)
        x_logvar = self.var2(h)
        x = self.sigmoid(self.sample_x(x_mu, x_logvar))

        return x

    def buildEncoder(self, input_dim, hidden_dim, z_dim):
        h = nn.Linear(input_dim, hidden_dim)
        mu = nn.Linear(hidden_dim, z_dim)
        var = nn.Linear(hidden_dim, z_dim)

        return h, mu, var

    def buildDecoder(self, input_dim, output_dim, z_dim):
        h = nn.Linear(z_dim, output_dim)
        mu = nn.Linear(output_dim, input_dim)
        var = nn.Linear(output_dim, input_dim)

        return h, mu, var

    def buildDist(self, z_dim, input_dim, batch_size):
        # reparametrization trick
        def sample_z(mu, logvar):
            eps = Variable(torch.randn(batch_size, z_dim))
            return mu + torch.exp(logvar / 2) * eps

        def sample_x(mu, logvar):
            eps = Variable(torch.randn(batch_size, input_dim))
            return mu + torch.exp(logvar / 2) * eps

        return sample_z, sample_x