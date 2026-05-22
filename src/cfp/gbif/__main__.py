import typer

from .build_manifest import app as bm_app
from .batch_download import app as batch_app

app = typer.Typer(help="GBIF / iNat manifest construction.")
app.add_typer(bm_app)
app.add_typer(batch_app, name="batch")

if __name__ == "__main__":
    app()
