import time
from argparse import Namespace
from contextlib import nullcontext
from threading import Lock, Thread

import ipywidgets as w
import numpy as np
from IPython.display import clear_output, display


class UIView:
    def __init__(self, container, named_widgets) -> None:
        self._named_widgets = named_widgets
        self._container = container

    def __dir__(self):
        return self._named_widgets.keys()

    def _ipython_display_(self):
        display(self._container)

    def __getattr__(self, attr):
        if attr in self._named_widgets:
            return self._named_widgets[attr]

        raise AttributeError()

    def __call__(self, decorated):
        from functools import wraps

        @wraps(decorated)
        def decorator(*args, **kwargs):
            return decorated(self.values, *args, **kwargs)

        return decorator

    @property
    def values(self):
        class Values:
            def __getattr__(_, attr):
                if hasattr(self._named_widgets[attr], "value"):
                    return self._named_widgets[attr].value
                else:
                    return self._named_widgets[attr]

            def __setattr__(_, name: str, value) -> None:
                self._named_widgets[name].value = value

        return Values()


class ui:
    @staticmethod
    def singleton(name, widget):
        return UIView(widget, {name: widget})

    @staticmethod
    def slider(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.FloatSlider(*args, **kwargs))

    @staticmethod
    def progress(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.FloatProgress(min=0, max=1, **kwargs))

    @staticmethod
    def checkbox(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.Checkbox(*args, **kwargs))

    @staticmethod
    def int(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.BoundedIntText(*args, **kwargs))

    @staticmethod
    def float(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.FloatText(*args, **kwargs))

    @staticmethod
    def text(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.Text(*args, **kwargs))

    @staticmethod
    def button(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.Button(*args, **kwargs))

    @staticmethod
    def output(id="unset"):
        class Output(w.Output):
            def display(self, *things):
                with self:
                    clear_output(wait=True)
                    for t in things:
                        display(t)

            def __enter__(self):
                enter_return = super().__enter__()
                clear_output(wait=True)
                return enter_return

        return ui.singleton(name=id, widget=Output())

    @staticmethod
    def label(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.Label(*args, **kwargs))

    @staticmethod
    def line(id="unset", *args, **kwargs):
        import bqplot as bq

        x_sc = bq.LinearScale(min=0, max=10)
        y_sc = bq.LinearScale(min=-1, max=1)

        ax_x = bq.Axis(label="X", scale=x_sc, tick_format="0.0f")
        ax_y = bq.Axis(
            label="Y", scale=y_sc, orientation="vertical", tick_format="0.2f"
        )

        line = bq.Lines(
            x=[0],
            y=[0],
            scales={"x": x_sc, "y": y_sc},
            colors=["blue"],
        )

        fig = bq.Figure(axes=[ax_x, ax_y], marks=[line], **kwargs)

        class LineWidget(w.Output):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                with self:
                    display(fig)

            def plot(_, y):
                x = np.arange(len(y))
                x_sc.max = len(y)
                y_sc.min = y.min()
                y_sc.max = y.max()
                line.x = x
                line.y = y

        return ui.singleton(name=id, widget=LineWidget())

    @staticmethod
    def h(*children, **kwargs):
        new_named_widgets = {
            n: w for u in children for n, w in u._named_widgets.items()
        }
        container_content = [u._container for u in children]
        return UIView(
            container=w.HBox(container_content, **kwargs),
            named_widgets=new_named_widgets,
        )

    @staticmethod
    def v(*children, **kwargs):
        new_named_widgets = {
            n: w for u in children for n, w in u._named_widgets.items()
        }
        container_content = [u._container for u in children]
        return UIView(
            container=w.VBox(container_content, **kwargs),
            named_widgets=new_named_widgets,
        )


def train(model, dataloader, in_background=True, **params):
    params = Namespace(**params)
    running_state = "stopped"
    it = 0
    batch_iterator = iter(dataloader)
    loss_history = []
    lock = nullcontext()

    view = ui.v(
        ui.h(
            ui.v(
                ui.label(value="LR"),
                ui.slider(
                    "lr",
                    min=0.0001,
                    max=1,
                    value=0.001,
                    step=0.0001,
                    readout_format=".4f",
                    layout={"width": "auto"},
                ),
                ui.label(value="ITS"),
                ui.int(
                    "its",
                    min=1,
                    max=999999999,
                    value=params.its,
                    layout={"width": "auto"},
                ),
                layout=w.Layout(**{"width": "25%", "border": "2px solid #ccc"}),
            ),
            ui.v(
                ui.h(
                    ui.progress("progress", description="IT [000:000]"),
                    ui.button("play", description="Start"),
                    ui.button("stop", description="Stop"),
                ),
                ui.line("loss", title="Loss", layout={"height": "350px"}),
                layout={"width": "100%", "border": "2px solid #ccc"},
            ),
            layout={"border": "2px solid #ccc"},
        ),
        ui.output("out"),
        layout={"border": "4px solid #e6e6e6"},
    )

    def step():
        batch = next(batch_iterator)
        loss = model.training_step(batch)
        loss_history.append(loss)

    def train():
        nonlocal it, batch_iterator
        try:
            while True:
                with lock:
                    step()
                    its = view.its.value

                    if running_state != "running":
                        break

                    if it > its:
                        stop()
                        break

                    view.progress.value = (it + 1) / (its + 1)
                    view.progress.description = f"IT [{it:03}:{its:03}]"

                    loss_history_np = np.array(loss_history)
                    view.loss.plot(loss_history_np)
                    it += 1
        except StopIteration:
            stop()

    def play():
        with lock:
            nonlocal running_state
            view.play.description = "Pause"
            view.stop.disabled = False
            running_state = "running"

            if in_background:
                thread = Thread(target=train)
                thread.start()
            else:
                train()

    def pause():
        with lock:
            nonlocal running_state
            view.play.description = "Start"
            running_state = "paused"

    def stop():
        with lock:
            nonlocal it, batch_iterator, running_state, loss_history
            pause()
            it = 0
            loss_history = []
            batch_iterator = iter(dataloader)
            view.stop.disabled = True
            running_state = "stopped"

    def toggle():
        if running_state != "running":
            play()
        else:
            pause()

    view.play.on_click(lambda _: toggle())
    view.stop.on_click(lambda _: stop())

    return view
