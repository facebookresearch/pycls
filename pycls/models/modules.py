#!/usr/bin/env python3

"""Custom module abstractions."""

import torch.nn as nn


class Flatten(nn.Module):
    """Computes the flattened view of the input tensor."""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
