from typing import Tuple, Union

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
        self.it = iter(self.data)
        return self

    def __next__(self):
        el = next(self.it)
        el = self.transform(el)
        return el
