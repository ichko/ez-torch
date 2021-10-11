from functools import wraps

import torch
import torch.nn.functional as F
import torchvision


def extend(type, is_property=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        name = func.__name__
        if is_property:
            func = property(func)

        setattr(type, name, func)
        return wrapper

    return decorator


class EasyTensor:
    def __init__(self, tensor):
        self.raw = tensor

    def __getattr__(self, key):
        if hasattr(self.raw, key):
            attr = getattr(self.raw, key)
            if callable(attr):

                @wraps(attr)
                def caller(*args, **kwargs):
                    self.raw = attr(*args, **kwargs)
                    return self

                return caller

            return getattr(self.raw, key)

        raise ValueError(f"EasyTensor does not have property {key}")

    def __getitem__(self, *args):
        return EasyTensor(self.raw.__getitem__(*args))

    @property
    def np(self):
        return self.raw.detach().cpu().numpy()

    def resize(self, *size):
        return EasyTensor(
            F.interpolate(self.raw, size, mode="bicubic", align_corners=True)
        )

    def grid(self, nr=None, padding=3):
        if nr == None:
            nr = self.raw.size(0)
        return EasyTensor(
            torchvision.utils.make_grid(
                self.raw, nrow=nr, padding=padding, normalize=True
            )
        )

    def spread_bs(self, *split_shape):
        shape = self.raw.shape
        _bs, rest_dims = shape[0], shape[1:]
        return EasyTensor(self.raw.reshape(*split_shape, *rest_dims))

    @property
    def hwc(self):
        return EasyTensor(self.raw.permute(0, 2, 3, 1))

    def imshow(self, figsize=None):
        import matplotlib.pyplot as plt

        tensor = self.hwc.np
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots(1, 1)
        ax.imshow(tensor)
        plt.close()

        return fig

    def seq_in_batch(self):
        # This function assumes that dim=1 is the seq dim
        bs, seq_size = self.raw.shape[:2]
        return EasyTensor(self.raw.view(bs * seq_size, *self.raw.shape[2:]))

    def batch_in_seq(self, bs=None, seq_size=None):
        bs_new, rest_shape = self.raw.shape[0], self.raw.shape[1:]
        if bs is None and seq_size is None:
            raise Exception("At least one of bs, seq_size, should be given")

        if bs is None:
            bs = bs_new // seq_size
        else:
            seq_size = bs_new // bs

        return EasyTensor(self._easy_tensor.raw.view(bs, seq_size, *rest_shape))


@extend(torch.Tensor, is_property=True)
def ez(tensor: torch.Tensor):
    return EasyTensor(tensor=tensor)
