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


class TensorWrapper:
    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, key):
        if hasattr(self.tensor, key):
            attr = getattr(self.tensor, key)
            if callable(attr):

                @wraps(attr)
                def caller(*args, **kwargs):
                    self.tensor = attr(*args, **kwargs)
                    return self

                return caller

            return getattr(self.tensor, key)

        raise ValueError(f"TensorWrapper does not have property {key}")

    @property
    def raw(self):
        return self.tensor

    @property
    def np(self):
        return self.tensor.detach().cpu().numpy()

    def resize(self, *size):
        self.tensor = F.interpolate(
            self.tensor, size, mode="bicubic", align_corners=True
        )
        return self

    def grid(self, nr=None, padding=3):
        if nr == None:
            nr = self.tensor.size(0)
        self.tensor = torchvision.utils.make_grid(
            self.tensor, nrow=nr, padding=padding, normalize=True
        )
        return self

    def spread_bs(self, *split_shape):
        shape = self.tensor.shape
        _bs, rest_dims = shape[0], shape[1:]
        self.tensor = self.tensor.reshape(*split_shape, *rest_dims)
        return self

    def imshow(self, figsize=(16, 16)):
        import matplotlib.pyplot as plt

        tensor = self.raw
        tensor = tensor.permute(1, 2, 0)
        tensor = TensorWrapper(tensor).np
        fig = plt.figure(figsize=figsize)

        ax = fig.subplots(1, 1)
        ax.imshow(tensor)

        return self


@extend(torch.Tensor, is_property=True)
def ez(tensor: torch.Tensor):
    return TensorWrapper(tensor=tensor)
