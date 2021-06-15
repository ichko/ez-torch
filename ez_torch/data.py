from typing import Tuple, Union

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.mnist import MNIST
from torchvision.transforms.transforms import Compose, Normalize, ToTensor


def get_mnist_dl(bs_train, bs_test, shuffle) -> Tuple[DataLoader, DataLoader]:
    transform = Compose(
        [
            ToTensor(),
            Normalize((0,), (1,)),
        ]
    )

    train_loader = DataLoader(
        MNIST(
            ".tmp",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=bs_train,
        shuffle=shuffle,
    )

    test_loader = DataLoader(
        MNIST(
            ".tmp",
            train=False,
            download=True,
            transform=transform,
        ),
        batch_size=bs_test,
        shuffle=shuffle,
    )

    return train_loader, test_loader


class MapDataset(Dataset):
    def __init__(self, transform, data: Union[Dataset, DataLoader]) -> None:
        super().__init__()
        self.transform = transform
        self.data = data
        self.it = None

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return (self.transform(b, i) for i, b in enumerate(self.data))


class CachedDataset:
    """Used to generate cached version of a datamodule. Yield once
    and then yields from the cache.

    Args:
        dl ([Dataloader]): Dataloader to cache

    Returns:
        Dataloader: The cached dataloader

    Yields:
        Batch: batches of data
    """

    def __init__(self, data: Union[Dataset, DataLoader]) -> None:
        self.buffer = []
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        if len(self.buffer) > 0:
            for b in self.buffer:
                yield b
        else:
            for b in self.data:
                self.buffer.append(b)
                yield b


class ParamCompose(nn.Module):
    """Compose multiple modules that can yield be forwarded to yield the
    parameters of random transformation. This is needed because we might
    want to do this transformation multiple times.

    Used to forward parameters of random transformations
    of `kornia` augmentation modules.
    """

    def __init__(self, functions):
        super().__init__()
        self.functions = nn.ModuleList(functions)

    def forward(self, inp, params=None):
        if params is None:
            params = [None] * len(self.functions)

        for f, p in zip(self.functions, params):
            inp = f(inp, p)

        return inp

    def forward_parameters(self, shape, device="cpu"):
        params = []
        for f in self.functions:
            p = f.forward_parameters(shape)
            pp = {}
            for k, v in p.items():
                pp[k] = v.to(device)
            params.append(pp)

        return params
