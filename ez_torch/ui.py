import time
from argparse import Namespace
from threading import Thread

import bqplot as bq
import ipywidgets as w
from IPython.display import clear_output, display


class TrainableUI(w.Output):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.epoch = 0
        self.hp = {}
        self.restart()

    def restart(self):
        self.slider = w.FloatSlider()
        self.progressIts = w.FloatProgress(description="[000:000]", min=0, max=1)
        self.running = False

        self.params = Namespace(
            its=w.BoundedIntText(
                value=100,
                min=0,
                max=10000,
                step=1,
                description="its:",
            )
        )
        paramsList = [p for _, p in vars(self.params).items()]
        for _, p in vars(self.params).items():
            p.layout.width = "200px"

        self.toggleBtn = w.Button(description="Play Training")
        self.toggleBtn.on_click(lambda _: self.toggle())
        self.it = 0

        restart = w.Button(description="Restart")
        restart.on_click(lambda _: self.restart())

        buttonsContainer = w.HBox([self.toggleBtn, restart])
        progressContainer = w.HBox([self.progressIts])

        paramsBox = w.VBox(paramsList)

        container = w.HBox(
            [
                paramsBox,
                w.VBox([buttonsContainer, progressContainer]),
            ]
        )

        with self:
            clear_output()
            display(container)

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        self.running = True
        self.toggleBtn.description = "Pause Training"
        thread = Thread(target=self.train)
        thread.start()

    def stop(self):
        self.running = False
        self.toggleBtn.description = "Play Training"

    def train(self):
        while True:
            its = self.params.its.value

            if not self.running or self.it >= its:
                if self.it >= its:
                    self.it = 0
                self.stop()
                break

            self.progressIts.value = (self.it + 1) / its
            time.sleep(0.01)
            self.it += 1
