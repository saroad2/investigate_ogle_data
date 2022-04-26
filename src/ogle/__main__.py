import json
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
from ogle.ogle_util import extract_microlensing_properties
from ogle.random_data import generate_parabolic_data, sample_records

MICROLENSING_PROPERTY_NAMES = ["t0", "f_max", "u_min"]


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
    output_dir = data_path.parent / f"{data_path.stem}_fitting_results"
    output_dir.mkdir(exist_ok=True)
    real_a, x, y = read_data(data_path=data_path, is_random=is_random)
    fit_result = fit_parabolic_data(x=x, y=y)
    microlensing_properties = extract_microlensing_properties(
        a=fit_result.a, aerr=fit_result.aerr
    )

    plt.title(rf"Parabolic fit ($\chi^2_{{red}} = {fit_result.chi2_reduced:.2e}$)")
    plt.scatter(x, y, label="Data points")
    plt.plot(x, np.polyval(fit_result.a, x), label="Evaluated parabola")
    if real_a is not None:
        plt.plot(x, np.polyval(real_a, x), label="Real parabola")
    plt.legend()
    plt.savefig(output_dir / "parabolic_fit.png")
    plt.clf()

    with open(output_dir / "fit_result.json", mode="w") as fd:
        result_as_dict = fit_result.as_dict()
        result_as_dict.update(
            {
                key: [
                    value.nominal_value,
                    value.std_dev,
                    value.std_dev / np.fabs(value.nominal_value) * 100,
                ]
                for key, value in microlensing_properties.items()
            }
        )
        json.dump(result_as_dict, fd, indent=2)


@ogle_cli.command("monte-carlo")
@click.option(
    "-d",
    "--data-path",
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
)
@click.option("--random", "is_random", is_flag=True, default=False)
@click.option("-e", "--experiments", type=int, default=DEFAULT_EXPERIMENTS)
def monte_carlo_cli(data_path, is_random, experiments):
    output_dir = data_path.parent / f"{data_path.stem}_monte_carlo_results"
    output_dir.mkdir(exist_ok=True)
    _, x, y = read_data(data_path=data_path, is_random=is_random)
    results = []
    click.echo("Running experiments...")
    for _ in tqdm.trange(experiments):
        x_samples, y_samples = sample_records(x, y)
        fit_results = fit_parabolic_data(x_samples, y_samples)
        results.append(extract_microlensing_properties(fit_results.a, fit_results.aerr))
    click.echo("Done!")
    for property_name in MICROLENSING_PROPERTY_NAMES:
        a = np.array([result[property_name].nominal_value for result in results])
        aerr = np.array([result[property_name].std_dev for result in results])
        mean_value = np.mean(a)
        sample_error = np.sqrt(np.sum(aerr**2)) / a.shape[0]
        stat_error = np.std(a)
        total_error = np.sqrt(sample_error**2 + stat_error**2)
        percentage_error = total_error / np.fabs(mean_value) * 100
        plt.title(
            f"Values hist for {property_name} - {mean_value:.2f} "
            rf"$\pm$ {total_error:.2f} "
            f"( {percentage_error:.2f}% )"
        )
        plt.hist(a, bins=50)
        plt.savefig(output_dir / f"{property_name}_hist.png")
        plt.clf()
    for i in range(len(MICROLENSING_PROPERTY_NAMES) - 1):
        for j in range(i + 1, len(MICROLENSING_PROPERTY_NAMES)):
            property_name1, property_name2 = (
                MICROLENSING_PROPERTY_NAMES[i],
                MICROLENSING_PROPERTY_NAMES[j],
            )
            x = np.array([result[property_name1].nominal_value for result in results])
            y = np.array([result[property_name2].nominal_value for result in results])
            covariance = np.cov(x, y)[0, 1]
            correlation = covariance / (np.mean(x) * np.mean(y))
            plt.title(
                f"Correlation for {property_name1} and {property_name2} "
                f"- {correlation:.2f}"
            )
            plt.scatter(x, y)
            plt.savefig(
                output_dir / f"{property_name1}_{property_name2}_correlation.png"
            )
            plt.clf()


if __name__ == "__main__":
    ogle_cli()
