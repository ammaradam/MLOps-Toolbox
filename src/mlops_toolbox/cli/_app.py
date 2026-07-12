from __future__ import annotations

import typer

from mlops_toolbox.cli.card import card
from mlops_toolbox.cli.config import config
from mlops_toolbox.cli.dashboard import dashboard
from mlops_toolbox.cli.doctor import doctor
from mlops_toolbox.cli.drift import drift
from mlops_toolbox.cli.gate import gate
from mlops_toolbox.cli.init import init
from mlops_toolbox.cli.listing import models, projects
from mlops_toolbox.cli.register import edit, register, unregister
from mlops_toolbox.cli.score import score
from mlops_toolbox.cli.serve import serve
from mlops_toolbox.cli.ship import ship

app = typer.Typer(
    help="mlops-toolbox: init, serve, ship, score, gate, drift, card, and audit "
    "registered models.",
    no_args_is_help=True,
)
app.command()(init)
app.command()(serve)
app.command()(ship)
app.command()(score)
app.command()(gate)
app.command()(drift)
app.command()(card)
app.command()(doctor)
app.command()(models)
app.command()(projects)
app.command()(register)
app.command()(edit)
app.command()(unregister)
app.command()(dashboard)
app.command()(config)
