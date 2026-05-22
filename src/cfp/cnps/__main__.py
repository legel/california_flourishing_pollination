import typer

from .fetch import app as fetch_app

app = typer.Typer(help="CNPS / Calflora ingestion.")
app.add_typer(fetch_app)

if __name__ == "__main__":
    app()
