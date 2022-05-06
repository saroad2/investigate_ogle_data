from pathlib import Path

import click
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from ogle.cli.ogle_cli import ogle_cli_group
from ogle.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_DATA_POINTS,
    DEFAULT_EXPERIMENTS,
    DEFAULT_RELATIVE_ERROR,
    MICROLENSING_PROPERTY_NAMES,
)
from ogle.fit_data import fit_parabolic_data
from ogle.io_util import build_data, read_data, search_data_paths
from ogle.ogle_util import extract_microlensing_properties
from ogle.plot_util import plot_monte_carlo_results, plot_parabolic_fit
from ogle.random_data import generate_parabolic_data, sample_records


@ogle_cli_group.group("parabolic")
def parabolic_cli_group():
    """Parabolic fitting related commands"""


@parabolic_cli_group.command("build-parabolic-data")
@click.argument("data_path", type=click.Path(path_type=Path, exists=True))
@click.option("--show/--no-show", is_flag=True, default=False)
def build_parabolic(data_path, show):
    data_paths = search_data_paths(data_path, suffix="dat")
    for i, path in enumerate(data_paths, start=1):
        click.echo(f"Build data for {path} ({i}/{len(data_paths)})")
        x, y, y_err = build_data(path)
        if show:
            plt.errorbar(x, y, yerr=y_err, linestyle="none")
            plt.show()
            plt.clf()
        output_path = path.with_name(f"{path.stem}.csv")
        pd.DataFrame(dict(x=x, y=y, y_err=y_err)).to_csv(
            output_path, index=False, header=True
        )


@parabolic_cli_group.command("generate-parabolic-data")
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


@parabolic_cli_group.command("fit-data")
@click.option("-d", "--data-path", type=click.Path(path_type=Path, exists=True))
@click.option("--random", "is_random", is_flag=True, default=False)
@click.option("-n", "--data-points", type=int, default=DEFAULT_DATA_POINTS)
@click.option("-e", "--experiments", type=int, default=DEFAULT_EXPERIMENTS)
@click.option(
    "--monte-carlo/--no-monte-carlo",
    is_flag=True,
    default=True,
)
def fit_data_cli(data_path, is_random, data_points, monte_carlo, experiments):
    data_paths = search_data_paths(data_path, suffix="csv")
    delta_index = data_points // 2
    for i, path in enumerate(data_paths, start=1):
        click.echo(f"Fit data for {path} ({i}/{len(data_paths)})")

        real_a, x, y, yerr = read_data(data_path=path, is_random=is_random)
        max_index = np.argmax(y)
        x, y, yerr = (
            x[max_index - delta_index : max_index + delta_index],
            y[max_index - delta_index : max_index + delta_index],
            yerr[max_index - delta_index : max_index + delta_index],
        )
        t_start = x[0]
        click.echo("Fitting parabolic...")
        fit_result = fit_parabolic_data(x=x - t_start, y=y, yerr=yerr)
        plot_parabolic_fit(
            x=x,
            y=y,
            yerr=yerr,
            fit_result=fit_result,
            t_start=t_start,
            output_dir=path.with_name(f"{path.stem}_fitting_results"),
            real_a=real_a,
        )
        click.echo("Done!")
        if not monte_carlo:
            continue
        click.echo("Running monte Carlo...")
        results = []
        for _ in tqdm.trange(experiments):
            x_samples, y_samples, yerr_samples = sample_records(x, y, yerr)
            t_start = x_samples[0]
            fit_results = fit_parabolic_data(
                x_samples - t_start, y_samples, yerr_samples
            )
            results.append(
                extract_microlensing_properties(
                    fit_results.a, fit_results.aerr, t_start=t_start
                )
            )
        plot_monte_carlo_results(
            results,
            property_names=MICROLENSING_PROPERTY_NAMES,
            output_dir=path.with_name(f"{path.stem}_monte_carlo_results"),
        )
        click.echo("Done!")
