import torch
import torch.nn as nn


def VAELoss(target, reconstruction, z_mu, z_logvar):
    recon_loss = nn.binary_cross_entropy(reconstruction, target, size_average=False) / target.size(0)
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1.0 - z_logvar, 1))

    return recon_loss + kl_loss

