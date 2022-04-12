from pathlib import Path

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ogle.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_DATA_POINTS,
    DEFAULT_RELATIVE_ERROR,
)
from ogle.random_data import generate_parabolic_data


@click.group()
def ogle_cli():
    """Evaluate OGLE data."""


@ogle_cli.command("generate-parabolic-data")
@click.option(
    "-o",
    "--output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_DATA_PATH,
)
@click.option("-n", "--data-points", type=int, default=DEFAULT_DATA_POINTS)
@click.option("-x-rel-err", type=float, default=DEFAULT_RELATIVE_ERROR)
@click.option("-y-rel-err", type=float, default=DEFAULT_RELATIVE_ERROR)
@click.option("--show/--no-show", is_flag=True, default=False)
def generate_parabolic_data_cli(output_path, data_points, x_rel_err, y_rel_err, show):
    a, x, y = generate_parabolic_data(
        data_points=data_points, x_rel_err=x_rel_err, y_rel_err=y_rel_err
    )
    click.echo(f"Generated data with {a=}")
    if show:
        plt.scatter(x, y)
        plt.plot(x, np.polyval(a, x))
        plt.show()
    pd.DataFrame({"x": x, "y": y}).to_csv(output_path, index=False)
    click.echo(f"Data was saved to {output_path}")


if __name__ == "__main__":
    ogle_cli()
