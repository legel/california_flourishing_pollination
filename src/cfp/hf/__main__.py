import typer

from .publish_meta import app as meta_app

app = typer.Typer(help="HF dataset publishing.")
app.add_typer(meta_app, name="publish-meta")

if __name__ == "__main__":
    app()
