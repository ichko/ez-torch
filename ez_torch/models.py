import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ez_torch.utils import count_parameters, get_uv_grid


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def count_parameters(self):
        return count_parameters(self)

    def make_persisted(self, path):
        self.path = path

    def persist(self):
        torch.save(self.state_dict(), self.path)

    def preload_weights(self):
        self.load_state_dict(torch.load(self.path))

    def save(self, path=None):
        path = path if self.path is None else path
        torch.save(self, f"{self.path}_whole.h5")

    def can_be_preloaded(self):
        return os.path.isfile(self.path)

    def configure_optim(self, lr):
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def metrics(self, _loss, _info):
        return {}

    def set_requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def summary(self, input_size=-1):
        try:
            from torchsummary import summary

            summary(self, input_size)
            return
        except Exception:
            pass

        result = f" > {self.name[:38]:<38} | {count_parameters(self):09,}\n"
        for name, module in self.named_children():
            type = module._get_name()
            num_prams = count_parameters(module)
            result += f" >  {name[:20]:>20}: {type[:15]:<15} | {num_prams:9,}\n"

        print(result)

    def optim_forward(self, X):
        return self.forward(X)

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def optim_step(self, batch):
        X, y = batch

        y_pred = self.optim_forward(X)
        loss = self.criterion(y_pred, y)

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        metrics = self.metrics(
            {
                "loss": loss,
                "X": X,
                "y": y,
                "y_pred": y_pred,
            }
        )

        return {
            "metrics": metrics,
            "loss": loss,
            "y_pred": y_pred,
        }


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
        inp, tensor_3d = x

        theta = self.locator(inp)
        _, C, H, W = tensor_3d.shape

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

        tensor_3d = tensor_3d.reshape(-1, 1, H, W)
        tensor_3d = F.grid_sample(
            tensor_3d,
            grid,
            align_corners=True,
        )

        return tensor_3d.reshape(-1, C, H, W)


class SpatialUVTransformer(nn.Module):
    def __init__(self, i, uv_resolution_shape):
        super().__init__()

        self.uv_resolution_shape = uv_resolution_shape
        self.infer_uv = nn.Sequential(
            nn.Linear(i, np.prod(self.uv_resolution_shape) * 2),
            Reshape(-1, 2, *self.uv_resolution_shape),
            nn.Sigmoid(),
        )

    def forward(self, x):
        inp, tensor_3d = x
        uv_map = self.infer_uv(inp)
        H, W = tensor_3d.shape[-2:]
        uv_map = uv_map.wrap.resize(H, W).raw.permute(0, 2, 3, 1)
        tensor_3d = F.grid_sample(
            tensor_3d,
            uv_map,
            align_corners=True,
        )
        return tensor_3d


class SpatialUVOffsetTransformer(nn.Module):
    def __init__(self, i, uv_resolution_shape):
        super().__init__()

        self.uv_resolution_shape = uv_resolution_shape
        self.infer_offset = nn.Sequential(
            nn.Linear(i, np.prod(self.uv_resolution_shape) * 2),
            Reshape(-1, 2, *self.uv_resolution_shape),
            nn.Sigmoid(),
        )
        self.uv_map = get_uv_grid(*uv_resolution_shape)

    def forward(self, x):
        inp, tensor_3d = x
        offset_map = self.infer_offset(inp) + self.uv_map

        H, W = tensor_3d.shape[-2:]
        offset_map = offset_map.wrap.resize(H, W).raw.permute(0, 2, 3, 1)
        tensor_3d = F.grid_sample(
            tensor_3d,
            offset_map,
            align_corners=True,
        )
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
