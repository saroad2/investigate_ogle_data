from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class FitResult:
    a: np.ndarray
    acov: np.ndarray
    chi2: float
    degrees_of_freedom: int
    chi2_reduced: float = field(init=False)
    aerr: np.ndarray = field(init=False)
    arelerr: np.ndarray = field(init=False)
    p_value: float = field(init=False)

    def __post_init__(self):
        self.chi2_reduced = self.chi2 / self.degrees_of_freedom
        self.aerr = np.sqrt(np.diag(self.acov))
        self.arelerr = np.abs(self.aerr / self.a)
        self.p_value = stats.chi2.sf(self.chi2, self.degrees_of_freedom)

    def as_dict(self):
        self_as_dict = dict(
            degrees_of_freedom=self.degrees_of_freedom,
            chi2=self.chi2,
            chi2_reduced=self.chi2_reduced,
            acov=self.acov.tolist(),
            pvalue=self.p_value,
        )
        for i, (a_val, aerr_val) in enumerate(zip(self.a, self.aerr), start=1):
            self_as_dict[f"a{i}"] = a_val
            self_as_dict[f"a{i}_error"] = aerr_val
            self_as_dict[f"a{i}_percentage_error"] = aerr_val / np.fabs(a_val) * 100
        return self_as_dict


def fit_parabolic_data(x, y, yerr):
    vander = vander_matrix(x, n=2)
    scale = np.sqrt((vander * vander).sum(axis=0))
    vander /= scale
    weight_matrix = np.diag(1 / yerr**2)
    k_matrix = np.dot(vander.T, weight_matrix)
    n_matrix = np.linalg.inv(np.dot(k_matrix, vander))
    a = np.dot(n_matrix, np.dot(k_matrix, y)) / scale
    degrees_of_freedom = x.shape[0] - 3
    chi2 = calculate_chi2(y, np.polyval(a, x), yerr)
    acov = n_matrix / np.outer(scale, scale)
    acov *= chi2 / degrees_of_freedom
    return FitResult(a=a, acov=acov, chi2=chi2, degrees_of_freedom=degrees_of_freedom)


def vander_matrix(x, n):
    return np.hstack([np.power(x.reshape(-1, 1), n - i) for i in range(n + 1)])


def calculate_chi2(y_true, y_pred, yerr, degrees_of_freedom: int = 1):
    return float(np.sum(((y_true - y_pred) / yerr) ** 2)) / degrees_of_freedom
