import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ez_torch.base_module import Module
from ez_torch.utils import get_uv_grid


def leaky(slope=0.2):
    return nn.LeakyReLU(slope, inplace=True)


def conv_block(i, o, ks, s, p, a=leaky(), d=1, bn=True):
    block = [nn.Conv2d(i, o, kernel_size=ks, stride=s, padding=p, dilation=d)]
    if bn:
        block.append(nn.BatchNorm2d(o))
    if a is not None:
        block.append(a)

    return nn.Sequential(*block)


def deconv_block(i, o, ks, s, p, a=leaky(), d=1, bn=True):
    block = [
        nn.ConvTranspose2d(
            i,
            o,
            kernel_size=ks,
            stride=s,
            padding=p,
            dilation=d,
        )
    ]

    if bn:
        block.append(nn.BatchNorm2d(o))
    if a is not None:
        block.append(a)

    if len(block) == 1:
        return block[0]

    return nn.Sequential(*block)


def dense(i, o, a=leaky()):
    l = nn.Linear(i, o)
    return l if a is None else nn.Sequential(l, a)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class Lambda(nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.forward = forward

    def forward(self, *args):
        return self.forward(*args)


class SpatialLinearTransformer(nn.Module):
    def __init__(self, i, num_channels, only_translations=False):
        super().__init__()

        self.only_translations = only_translations
        self.num_channels = num_channels
        self.locator = nn.Sequential(
            nn.Linear(i, num_channels * 2 * 3),
            Reshape(-1, 2, 3),
        )

        self.device = self.locator[0].bias.device
        # Taken from the pytorch spatial transformer tutorial.
        self.locator[0].weight.data.zero_()
        self.locator[0].bias.data.copy_(
            torch.tensor(
                [1, 0, 0, 0, 1, 0] * num_channels,
                dtype=torch.float,
            ).to(self.device)
        )

    def forward(self, x):
        param_features, input_to_transform = x

        theta = self.locator(param_features)
        _, C, H, W = input_to_transform.shape

        if self.only_translations:
            theta[:, :, :-1] = (
                torch.tensor(
                    [[1, 0], [0, 1]],
                    dtype=torch.float,
                )
                .to(self.device)
                .unsqueeze_(0)
            )

        grid = F.affine_grid(
            theta,
            (theta.size(dim=0), 1, H, W),
            align_corners=True,
        )

        input_to_transform = input_to_transform.reshape(-1, 1, H, W)
        input_to_transform = F.grid_sample(
            input_to_transform,
            grid,
            align_corners=True,
        )

        return input_to_transform.reshape(-1, C, H, W)


class SpatialUVTransformer(nn.Module):
    def __init__(self, i, uv_resolution_shape):
        super().__init__()

        self.uv_resolution_shape = uv_resolution_shape
        self.infer_uv = nn.Sequential(
            nn.Linear(i, np.prod(self.uv_resolution_shape) * 2),
            Reshape(-1, 2, *self.uv_resolution_shape),
            nn.Sigmoid(),
            Lambda(lambda x: x * 2 - 1),  # range in [-1, 1]
        )

    def forward(self, x):
        inp, tensor_3d = x
        uv_map = self.infer_uv(inp)
        H, W = tensor_3d.shape[-2:]
        uv_map = uv_map.ez.resize(H, W).raw.permute(0, 2, 3, 1)
        tensor_3d = F.grid_sample(
            tensor_3d,
            uv_map,
            align_corners=True,
        )
        return tensor_3d


class SpatialUVOffsetTransformer(nn.Module):
    def __init__(self, inp, uv_resolution_shape, weight_mult_factor=0.5):
        super().__init__()

        self.uv_resolution_shape = uv_resolution_shape
        self.infer_offset = nn.Sequential(
            nn.Linear(inp, np.prod(self.uv_resolution_shape) * 2),
            Reshape(-1, 2, *self.uv_resolution_shape),
            nn.Sigmoid(),
            Lambda(lambda x: x * 2 - 1),  # range in [-1, 1]
        )
        self.id_uv_map = nn.parameter.Parameter(
            get_uv_grid(*uv_resolution_shape),
            requires_grad=False,
        )

        self.infer_offset[0].weight.data *= weight_mult_factor
        self.infer_offset[0].bias.data.fill_(0)

    def forward(self, x):
        inp, tensor_3d = x
        offset_map = self.infer_offset(inp) + self.id_uv_map

        H, W = tensor_3d.shape[-2:]
        offset_map = offset_map.ez.resize(H, W).raw.permute(0, 2, 3, 1)
        tensor_3d = F.grid_sample(
            tensor_3d,
            offset_map,
            align_corners=True,
        ).clamp(0, 1)
        return tensor_3d


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        shape = input[0].size() if type(input) is list else input.size()
        bs = shape[0]
        seq_len = shape[1]

        if type(input) is list:
            input = [i.reshape(-1, *i.shape[2:]) for i in input]
        else:
            input = input.reshape(-1, *shape[2:])

        out = self.module(input)
        out = out.view(bs, seq_len, *out.shape[1:])

        return out
