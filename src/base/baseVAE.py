import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseVAE(nn.Module):
    def __init__(self):
        raise(NotImplementedError)

    def buildEncoder(self):
        raise(NotImplementedError)

    def buildDecoder(self):
        raise(NotImplementedError)

    def buildDist(self):
        raise(NotImplementedError)

    def forward(self, *input):
        raise(NotImplementedError)

    def encode(self, *input):
        raise(NotImplementedError)

    def decode(self, *input):
        raise(NotImplementedError)