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


def fit_parabolic_data(x, y):
    vander = vander_matrix(x, n=2)
    scale = np.sqrt((vander * vander).sum(axis=0))
    vander /= scale
    n_matrix = np.linalg.inv(np.dot(vander.T, vander))
    a = np.dot(np.dot(n_matrix, vander.T), y) / scale
    degrees_of_freedom = x.shape[0] - 3
    residuals = float(np.sum(np.power(y - np.polyval(a, x), 2)))
    chi2 = residuals / degrees_of_freedom
    acov = n_matrix / np.outer(scale, scale)
    acov *= chi2
    return FitResult(a=a, acov=acov, chi2=chi2, degrees_of_freedom=degrees_of_freedom)


def vander_matrix(x, n):
    return np.hstack([np.power(x.reshape(-1, 1), n - i) for i in range(n + 1)])
