import os

import torch
import torch.nn as nn

from ez_torch.utils import count_parameters


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def count_parameters(self):
        return count_parameters(self)

    def save(self, path=None):
        path = path if self.path is None else path
        torch.save(self, f"{self.path}.pk")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters())

    def optimizers(self):
        if not hasattr(self, "__optimizers"):
            self.__optimizers = self.configure_optimizers()
        return self.__optimizers

    def set_requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def summary(self):
        result = f" > {self.name[:38]:<38} | {count_parameters(self):09,}\n"
        for name, module in self.named_children():
            type = module._get_name()
            num_prams = count_parameters(module)
            result += f" >  {name[:20]:>20}: {type[:15]:<15} | {num_prams:9,}\n"

        return result

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def training_step(self, batch):
        X, y = batch
        optim = self.optimizers()

        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        if loss.requires_grad:
            optim.zero_grad()
            loss.backward()
            optim.step()

        return {
            "loss": loss.item(),
            "y_pred": y_pred,
        }
