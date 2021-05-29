import ez_torch as ez
import matplotlib.pyplot as plt
import pytest
import torchvision
from ez_torch.data import get_mnist_dl
from ez_torch.models import SpatialUVOffsetTransformer
from ez_torch.vis import Fig


def test_version():
    assert ez.__version__ == "0.1.0"


def test_SpatialUVOffsetTransformer():
    train, test = get_mnist_dl(bs_train=100, bs_test=10, shuffle=False)
    X, y = next(iter(train))

    feature_model = torchvision.models.resnet18(pretrained=True, progress=False)
    # feature_model.eval()

    model = SpatialUVOffsetTransformer(1000, uv_resolution_shape=(5, 5))
    X_features = feature_model(X.repeat([1, 3, 1, 1]))
    X_transformed = model([X_features, X])

    fig = Fig(nr=1, nc=2, figsize=(10, 5))

    for i in range(100):
        fig[0].imshow(X.ez.grid(nr=10).channel_last)
        fig[1].imshow(X_transformed.ez.grid(nr=10).channel_last)

    plt.show()


if __name__ == "__main__":
    pytest.main()
