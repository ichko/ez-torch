import time
from argparse import Namespace
from threading import Thread

import ipywidgets as w
import numpy as np
from IPython.display import clear_output, display


class ui:
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

    @staticmethod
    def singleton(name, widget):
        return ui(widget, {name: widget})

    @staticmethod
    def slider(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.FloatSlider(*args, **kwargs))

    @staticmethod
    def progress(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.FloatProgress(min=0, max=1, **kwargs))

    @staticmethod
    def int(id="unset", *args, **kwargs):
        return ui.singleton(name=id, widget=w.IntText(*args, **kwargs))

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
        return ui(
            container=w.HBox(container_content, **kwargs),
            named_widgets=new_named_widgets,
        )

    @staticmethod
    def v(*children, **kwargs):
        new_named_widgets = {
            n: w for u in children for n, w in u._named_widgets.items()
        }
        container_content = [u._container for u in children]
        return ui(
            container=w.VBox(container_content, **kwargs),
            named_widgets=new_named_widgets,
        )


class TrainableUI(w.Output):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self._restart_ui()

    def _restart_ui(self):
        view = ui.v(
            ui.h(
                ui.v(
                    ui.label(value="LR"),
                    ui.slider("lr", min=0.001, max=1, layout={"width": "auto"}),
                    ui.label(value="ITS"),
                    ui.int(
                        "its", min=1, max=1000, default=100, layout={"width": "auto"}
                    ),
                    layout=w.Layout(**{"width": "25%", "border": "2px solid #ccc"}),
                ),
                ui.v(
                    ui.h(
                        ui.progress("progress", description="IT [000:000]"),
                        ui.button("start", description="Start"),
                        ui.button("reset", description="Restart"),
                    ),
                    ui.line("loss", title="Loss", layout={"height": "350px"}),
                    layout={"width": "100%", "border": "2px solid #ccc"},
                ),
                layout={"border": "2px solid #ccc"},
            ),
            ui.output("out"),
            layout={"border": "4px solid #e6e6e6"},
        )

        self.view = view
        with self:
            clear_output()
            display(self.view)

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
