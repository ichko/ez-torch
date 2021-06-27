from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import draw, pause
from torch import Tensor

from ez_torch.tensor_wrapper import TensorWrapper


def unwrap(tensor):
    if isinstance(tensor, TensorWrapper):
        return tensor.raw

    return tensor


class Fig:
    def __init__(self, nr, nc, ion=False, *args, **kwargs):
        if ion:
            plt.ion()

        self.fig = plt.figure(*args, **kwargs)

        self.axs = self.fig.subplots(nr, nc)
        self.plots_mem = {}
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

                        def set_data(plot):
                            try:
                                plot.set_array(data.ravel())
                                plot.autoscale()
                                return True
                            except:
                                try:
                                    plot.set_data(data)
                                    draw()
                                    pause()
                                    return True
                                except Exception as e:
                                    return False

                        def create_plot():
                            try:
                                plot = getattr(sns, identifier)(data, **kwargs, ax=ax)
                                self.plots_mem[key] = [ax, plot]
                            except:
                                try:
                                    plot = getattr(ax, identifier)(data, **kwargs)
                                    self.plots_mem[key] = [ax, plot]
                                except Exception as e:
                                    raise e

                        if key in self.plots_mem:
                            _, plot = self.plots_mem[key]
                            ax.clear()
                            success = set_data(plot)
                            if not success:
                                create_plot()
                        else:
                            create_plot()

                return Call()

        return Getattr()

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
