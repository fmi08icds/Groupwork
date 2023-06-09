import logging
from pathlib import Path

import typer

from regression_comparison import __title__
from regression_comparison import __version__
from regression_comparison import util
from regression_comparison.baseline import run_baseline
from regression_comparison.dataset.download_data import load_data
from regression_comparison.io import save_results

logger = logging.getLogger("regression_comparison")

app = typer.Typer(name="regression_comparison", help="A short summary of the project")


def version_callback(version: bool):
    if version:
        typer.echo(f"{__title__} {__version__}")
        raise typer.Exit()


ConfigOption = typer.Option(
    None, "-c", "--config", metavar="PATH", help="path to the program configuration"
)
VersionOption = typer.Option(
    None,
    "-v",
    "--version",
    callback=version_callback,
    is_eager=True,
    help="print the program version and exit",
)


@app.command()
def main(config_file: str = ConfigOption, version: bool = VersionOption):
    """
    This is the entry point of your command line application. The values of the CLI params that
    are passed to this application will show up als parameters to this function.

    This docstring is where you describe what your command line application does.
    Try running `python -m regression_comparison --help` to see how this shows up in the command line.
    """
    config = util.load_config(config_file)
    util.logging_setup(config)

    paths = config.get("paths")
    datasets = load_data(
        config.get("datasets_to_load"),
        Path(paths["non_generated"]),
        Path(paths["generated"]),
    )

    if config.get("baseline"):
        baseline_results = run_baseline(datasets, config.get("dependent_variables"))
        save_results(baseline_results, Path(paths["results"], "baseline.csv"))


if __name__ == "__main__":
    app()
