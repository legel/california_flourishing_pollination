import typer

from .download import app as download_app
from .embed import app as embed_app
from .upload import app as upload_app

app = typer.Typer(help="Streaming pipeline: download → embed → upload.")
app.add_typer(download_app)
app.add_typer(embed_app)
app.add_typer(upload_app)

if __name__ == "__main__":
    app()
