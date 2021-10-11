import os
import uuid
from datetime import datetime
from shutil import copyfile
from typing import Dict

import ffmpeg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import draw, pause
from torch import Tensor

from ez_torch.tensor_wrapper import EasyTensor


def unwrap(tensor):
    if isinstance(tensor, EasyTensor):
        return tensor.raw

    return tensor


IMAGES_BASE_PATH = ".tmp/fig_imgs/"


class Fig:
    def __init__(
        self,
        nr=1,
        nc=1,
        ion=False,
        realtime_render=True,
        vid_path=None,
        fps=30,
        *args,
        **kwargs,
    ):
        if ion:
            plt.ion()

        self.fig = plt.figure(*args, **kwargs)

        self.axs = self.fig.subplots(nr, nc)
        self.plots_mem = {}
        self.id = f"{datetime.now()}-{uuid.uuid4()}"
        self.current_frame = 0
        self.realtime_render = realtime_render

        # plt.close()

        self.vid_args = vid_path, fps
        self.fig_images_path = os.path.join(IMAGES_BASE_PATH, self.id)
        self.current_image_path = os.path.join(self.fig_images_path, "current.png")
        os.makedirs(self.fig_images_path, exist_ok=True)

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
        self._save_fig()

        if self.realtime_render:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _save_fig(self):
        sequential_image_path = os.path.join(
            self.fig_images_path, f"seq_{self.current_frame:06d}.png"
        )
        plt.savefig(self.current_image_path)
        copyfile(self.current_image_path, sequential_image_path)

        self.current_frame += 1

    def __del__(self):
        path, fps = self.vid_args
        if path is not None:
            ffmpeg.input(
                f"{self.fig_images_path}/seq_*.png",
                pattern_type="glob",
                framerate=fps,
                loglevel="error",
            ).output(path).overwrite_output().run()
