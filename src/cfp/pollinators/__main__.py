import typer

from .cross_check import app as cc_app

app = typer.Typer(help="Pollinator cross-check.")
app.add_typer(cc_app, name="cross-check")

if __name__ == "__main__":
    app()
