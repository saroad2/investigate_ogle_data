from pathlib import Path

import click
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from ogle.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_DATA_POINTS,
    DEFAULT_EXPERIMENTS,
    DEFAULT_RELATIVE_ERROR,
)
from ogle.fit_data import fit_parabolic_data
from ogle.io_util import read_data
from ogle.random_data import generate_parabolic_data, sample_records


@click.group()
def ogle_cli():
    """Evaluate OGLE data."""


@ogle_cli.command("build-parabolic-data")
@click.argument(
    "data_path",
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
)
@click.option("-n", "--data-points", type=int, default=DEFAULT_DATA_POINTS)
@click.option("--show/--no-show", is_flag=True, default=False)
def build_parabolic(data_path, data_points, show):
    with open(data_path) as fd:
        rows = fd.readlines()
    rows = [list(map(float, row.replace("\n", "").split())) for row in rows]
    x, y, _, _, _ = zip(*rows)
    x, y = np.array(x), np.array(y)
    x -= x[0]
    y = np.power(10, (y[0] - y) / 2.5)
    max_index = np.argmax(y)
    delta_index = data_points // 2
    x, y = (
        x[max_index - delta_index : max_index + delta_index],
        y[max_index - delta_index : max_index + delta_index],
    )
    if show:
        plt.plot(x, y)
        plt.show()
        plt.clf()
    output_path = data_path.parent / f"{data_path.stem}.csv"
    pd.DataFrame(dict(x=x, y=y)).to_csv(output_path, index=False, header=True)


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


@ogle_cli.command("fit-data")
@click.option(
    "-d",
    "--data-path",
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
)
@click.option("--random", "is_random", is_flag=True, default=False)
def fit_data_cli(data_path, is_random):
    real_a, x, y = read_data(data_path=data_path, is_random=is_random)
    fit_result = fit_parabolic_data(x=x, y=y)

    plt.title(rf"Parabolic fit ($\chi^2_{{red}} = {fit_result.chi2_reduced:.2e}$)")
    plt.scatter(x, y, label="Data points")
    plt.plot(x, np.polyval(fit_result.a, x), label="Evaluated parabola")
    if real_a is not None:
        plt.plot(x, np.polyval(real_a, x), label="Real parabola")
    plt.legend()
    plt.show()
    plt.clf()


@ogle_cli.command("monte-carlo")
@click.option(
    "-d",
    "--data-path",
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
)
@click.option("--random", "is_random", is_flag=True, default=False)
@click.option("-e", "--experiments", type=int, default=DEFAULT_EXPERIMENTS)
def monte_carlo_cli(data_path, is_random, experiments):
    real_a, x, y = read_data(data_path=data_path, is_random=is_random)
    a_results = []
    click.echo("Running experiments...")
    for _ in tqdm.trange(experiments):
        x_samples, y_samples = sample_records(x, y)
        a_results.append(fit_parabolic_data(x_samples, y_samples).a)
    click.echo("Done!")
    a_results = np.vstack(a_results)
    a, aerr = np.mean(a_results, axis=0), np.std(a_results, axis=0)
    arelerr = np.abs(aerr / a)
    for i in range(a.shape[0]):
        plt.title(
            rf"Values hist for a[{i}] - {a[i]:.2f} $\pm$ {aerr[i]:.2f} "
            f"( {arelerr[i] * 100 :.2f}% )"
        )
        plt.hist(a_results[:, i], bins=20)
        plt.show()
        plt.clf()
    for i in range(a.shape[0] - 1):
        for j in range(i + 1, a.shape[0]):
            x = a_results[:, i]
            y = a_results[:, j]
            covariance = np.cov(x, y)[0, 1]
            plt.title(rf"Covariance for a[{i}] and a[{j}] - {covariance:.2f}")
            plt.scatter(x, y)
            plt.show()
            plt.clf()


if __name__ == "__main__":
    ogle_cli()
