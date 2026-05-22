import typer

from .calscape import app as calscape_app
from .fetch import app as fetch_app
from .match_inat import app as match_app

app = typer.Typer(help="CNPS / Calscape / Calflora ingestion.")
app.add_typer(fetch_app)
app.add_typer(calscape_app, name="calscape")
app.add_typer(match_app, name="match")

if __name__ == "__main__":
    app()
