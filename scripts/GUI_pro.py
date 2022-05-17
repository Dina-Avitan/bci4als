import edifice as ed
from edifice.components.forms import Form
from edifice import View, Button, alert

button_style = {"margin": 10, "padding": 10, "font-size": 20}

class App(ed.Component):
    def __init__(self):
        super().__init__()
        rec_params = {}
        self.rec_params_state = ed.StateManager(rec_params)


    def render(self):
        return ed.Window()(
            View(layout="row", style={"margin": 20})(
                View(layout="column", style={"margin": 20})(
                    Form(self.rec_params_state),
                    Button("Save Recording Params",  style=button_style),
                ),
                View(layout="column", style={"margin": 20, "align": "top"})(
                    Button("Health Check", style=button_style),
                    Button("Start Recording", style=button_style),
                    Button("Create Pipeline", style=button_style),
                )
            )
        )

if __name__ == "__main__":
    ed.App(App()).start()
