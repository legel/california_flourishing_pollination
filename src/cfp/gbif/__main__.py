import typer

from .build_manifest import app as bm_app

app = typer.Typer(help="GBIF / iNat manifest construction.")
app.add_typer(bm_app)

if __name__ == "__main__":
    app()
