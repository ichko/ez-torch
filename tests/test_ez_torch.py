import ez_torch as ez
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ez_torch.data import get_mnist_dl
from ez_torch.models import Module, SpatialLinearTransformer, SpatialUVOffsetTransformer
from ez_torch.vis import Fig
from tqdm.auto import tqdm

# matplotlib.use("TkAgg")


def test_version():
    assert ez.__version__ == "0.1.0"


def test_SpatialLinearTransformer_classifier():
    # TODO: Task - Implement STN Classifier as show in the tutorial
    # <https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html>
    # Visualize training results - image transformation during training
    pass


def test_SpatialUVOffsetTransformer():
    device = "cuda"

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.feature_model = torchvision.models.resnet18(
                pretrained=False, progress=False, num_classes=254
            )
            # self.feature_model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 254))
            self.transform = SpatialUVOffsetTransformer(
                254,
                uv_resolution_shape=(2, 2),
            )
            # self.transform = SpatialLinearTransformer(
            #     1000,
            #     num_channels=1,
            # )

        def criterion(self, y, y_target):
            return F.binary_cross_entropy(y, y_target)

        def forward(self, x):
            X_features = self.feature_model(x.repeat([1, 3, 1, 1]))
            # X_features = self.feature_model(x)
            X_transformed = self.transform([X_features, X])
            return X_transformed

    train, test = get_mnist_dl(bs_train=16, bs_test=10, shuffle=False)
    X, y = next(iter(train))
    X = X.to(device)

    model = Model().to(device)
    model.configure_optim(lr=0.002)

    fig = Fig(nr=1, nc=3, ion=True, figsize=(10, 5))
    fig[0].imshow(X.ez.grid(nr=4).channel_last.np)
    history = []
    for _i in tqdm(range(1000)):
        info = model.optim_step([X, X])
        loss = info["loss"]
        X_transformed = info["y_pred"]
        print(loss)
        history.append(loss)

        # TODO: Fix animation updates
        fig[1].imshow(X_transformed.ez.grid(nr=4).channel_last.np)
        fig[2].plot(history)
        fig.update()

    plt.show()


if __name__ == "__main__":
    pytest.main()
