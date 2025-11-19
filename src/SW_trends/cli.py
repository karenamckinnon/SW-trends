"""Console script for SW_trends."""

import typer
from rich.console import Console

from SW_trends import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for SW_trends."""
    console.print("Replace this message by putting your code into "
               "SW_trends.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
