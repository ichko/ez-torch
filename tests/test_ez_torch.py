import ez_torch as ez
import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch.nn.functional as F
import torchvision
from ez_torch.data import get_mnist_dl
from ez_torch.models import Module, SpatialUVOffsetTransformer
from ez_torch.vis import Fig

# matplotlib.use("TkAgg")


def test_version():
    assert ez.__version__ == "0.1.0"


def test_SpatialUVOffsetTransformer():
    class Model(Module):
        def __init__(self):
            super().__init__()
            self.feature_model = torchvision.models.resnet18(
                pretrained=True, progress=False
            )
            self.transform = SpatialUVOffsetTransformer(
                1000, uv_resolution_shape=(10, 10)
            )

        def criterion(self, y, y_target):
            return F.binary_cross_entropy(y, y_target)

        def forward(self, x):
            X_features = self.feature_model(x.repeat([1, 3, 1, 1]))
            X_transformed = self.transform([X_features, X])
            return X_transformed

    train, test = get_mnist_dl(bs_train=100, bs_test=10, shuffle=False)
    X, y = next(iter(train))

    model = Model()
    model.configure_optim(lr=0.001)
    # feature_model.eval()

    fig = Fig(nr=1, nc=2, figsize=(10, 5))
    fig[0].imshow(X.ez.grid(nr=10).channel_last)

    for _i in range(100):
        info = model.optim_step([X, X])
        loss = info["loss"]
        X_transformed = info["y_pred"]

        # TODO: Fix animation updates
        fig[1].imshow(X_transformed.ez.grid(nr=10).channel_last)
        fig.update()

    plt.show()


if __name__ == "__main__":
    pytest.main()
