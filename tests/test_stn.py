import ez_torch as ez
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from ez_torch.data import get_mnist_dl
from ez_torch.models import Module, SpatialLinearTransformer, SpatialUVOffsetTransformer
from ez_torch.vis import Fig
from tqdm.auto import tqdm

#  Implement STN Classifier as show in the tutorial
# <https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html>
# Visualize training results - image transformation during training


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 254))

        # self.slt = SpatialLinearTransformer(i=254, num_channels=1)
        self.slt = SpatialUVOffsetTransformer(inp=254, uv_resolution_shape=[5, 5])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        x = self.slt([xs, x])
        # theta = self.fc_loc(xs)
        # theta = theta.view(-1, 2, 3)

        # grid = F.affine_grid(theta, x.size())
        # x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test_SpatialLinearTransformer_classifier():
    device = "cuda"

    model = Net().to(device)
    train_loader, test_loader = get_mnist_dl(bs_train=64, bs_test=64, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    def train(epoch):
        model.train()
        batch_idx, (data, target) = next(iter(enumerate(train_loader)))
        for _ in tqdm(range(256)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )

    #
    # A simple test procedure to measure the STN performances on MNIST.
    #

    def test():
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # sum up batch loss
                test_loss += F.nll_loss(output, target, size_average=False).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print(
                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    100.0 * correct / len(test_loader.dataset),
                )
            )

    def convert_image_np(inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # inp = std * inp + mean
        # inp = np.clip(inp, 0, 1)
        return inp

    # We want to visualize the output of the spatial transformers layer
    # after the training, we visualize a batch of input images and
    # the corresponding transformed batch using STN.

    def visualize_stn():
        with torch.no_grad():
            # Get a batch of training data
            data = next(iter(test_loader))[0].to(device)

            input_tensor = data.cpu()
            transformed_input_tensor = model.stn(data).cpu()

            in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))

            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor)
            )

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title("Dataset Images")

            axarr[1].imshow(out_grid)
            axarr[1].set_title("Transformed Images")

    for epoch in range(1, 5 + 1):
        train(epoch)
        test()

    # Visualize the STN transformation on some input batch
    visualize_stn()

    plt.ioff()
    plt.show()
