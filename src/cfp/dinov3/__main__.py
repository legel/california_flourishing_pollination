"""Module-level CLI: ``python -m cfp.dinov3 validate-sample``."""

import typer

from .validate_sample import app as validate_app

app = typer.Typer(help="DINOv3 utilities (validation + production embedding).")
app.add_typer(validate_app, name="validate-sample")

if __name__ == "__main__":
    app()
