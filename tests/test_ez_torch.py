import ez_torch as ez
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ez_torch.data import get_mnist_dl
from ez_torch.modules import (
    Module,
    SpatialLinearTransformer,
    SpatialUVOffsetTransformer,
)
from ez_torch.vis import Fig
from tqdm.auto import tqdm

# matplotlib.use("TkAgg")


def test_version():
    assert ez.__version__ == "0.1.0"


def test_SpatialUVOffsetTransformer_trainToBeIdentity():
    """
    Important observations for the STN Training on being the identity.

    - Activation of the feature extractor is important
        - it dictates the distribution of the input activations
            => the initial params of the geometric transformation
        - The initial params should be so that the initial transformation is close to id
        - Activating with ReLU seems to be a good thing to do at the end of the feature extractor
    - Using SGD is crucial. Adam seems to be shaky, the momentum is leading to sporadic
    geometric transformations and we often get stuck at local minima
    - LR is important HParam here. Smaller is better with Adam (0.0001). SGD works best,
    even with large LR (0.01) we get convergence to a good solution.
    """
    device = "cuda"

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.feature_model = nn.Sequential(
                torchvision.models.resnet18(
                    pretrained=False,
                    progress=False,
                    num_classes=32,
                ),
                nn.ReLU(),
            )
            # self.feature_model = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(28 * 28, 32),
            #     nn.ReLU(),
            # )
            self.transform = SpatialUVOffsetTransformer(
                32,
                uv_resolution_shape=(10, 10),
            )

        def criterion(self, y, y_target):
            return F.binary_cross_entropy(y, y_target)

        def forward(self, x):
            x_3d = x.repeat([1, 3, 1, 1])
            X_features = self.feature_model(x_3d)
            X_transformed = self.transform([X_features, x])
            return X_transformed

    train, test = get_mnist_dl(bs_train=16, bs_test=10, shuffle=False)
    X, y = next(iter(train))
    X = X.to(device)

    model = Model().to(device)
    model.configure_optim(lr=0.01)

    fig = Fig(
        nr=1,
        nc=4,
        ion=True,
        figsize=(20, 5),
        realtime_render=False,
        vid_path=".tmp/out.mp4",
        fps=10,
    )
    in_np = X.ez.grid(nr=4).channel_last.np
    fig[0].imshow(in_np)

    history = []
    tq = tqdm(range(30))
    for _i in tq:
        info = model.optim_step([X, X])
        loss = info["loss"]
        X_transformed = info["y_pred"]
        history.append(loss)
        tq.set_description(f"Loss: {loss}")

        # TODO: Fix animation updates
        fig[1].imshow(X_transformed.ez.grid(nr=4).channel_last.np)
        out_np = X_transformed.ez.grid(nr=4).channel_last.np
        fig[2].imshow(np.abs(in_np - out_np))
        fig[3].plot(history[-100:])
        fig.update()


if __name__ == "__main__":
    pytest.main()
