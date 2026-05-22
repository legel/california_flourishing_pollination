import typer

from .fetch_and_filter import app as ff_app

app = typer.Typer(help="GloBi ingestion + pollination filter.")
app.add_typer(ff_app)

if __name__ == "__main__":
    app()
