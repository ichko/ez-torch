from typing import Dict

import matplotlib.pyplot as plt
from torch import Tensor

from ez_torch.tensor_wrapper import TensorWrapper


def unwrap(tensor):
    if isinstance(tensor, TensorWrapper):
        return tensor.raw

    return tensor


class Fig:
    def __init__(self, nr, nc, *args, **kwargs):
        plt.ion()

        self.fig = plt.figure(*args, **kwargs)

        self.axs = self.fig.subplots(nr, nc)
        self.called = set()
        # plt.close()

    def __getitem__(self, index):
        ax = self.axs[index] if hasattr(self.axs, "__getitem__") else self.axs

        class Getattr:
            def __getattr__(_, identifier):
                if identifier == "ax":
                    return ax

                key = identifier, index

                class Call:
                    def __call__(_, data, **kwargs):
                        data = unwrap(data)

                        def set_data(child):
                            try:
                                child.set_array(data.ravel())
                                child.autoscale()
                                return True
                            except:
                                try:
                                    child.set_data(data)
                                    return True
                                except Exception as e:
                                    return False

                        def create_plot():
                            self.called.add(key)
                            try:
                                getattr(sns, identifier)(data, **kwargs, ax=ax)
                            except:
                                try:
                                    getattr(ax, identifier)(data, **kwargs)
                                except Exception as e:
                                    raise e

                        if key in self.called:
                            found = False
                            ax.clear()
                            for child in ax.get_children():
                                success = set_data(child)
                                if success:
                                    found = True
                            if not found:
                                create_plot()
                        else:
                            create_plot()

                return Call()

        return Getattr()

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
