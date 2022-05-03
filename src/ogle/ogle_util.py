from uncertainties import ufloat
from uncertainties.umath import *  # noqa: F403


def extract_microlensing_properties(a, aerr, t_start):
    a1, a2, a3 = (
        ufloat(a[0], aerr[0]),
        ufloat(a[1], aerr[1]),
        ufloat(a[2], aerr[2]),
    )
    t0 = t_start - a2 / (2 * a1)
    f_max = a3 - (a2**2) / (4 * a1)
    u_min = sqrt(2 * (f_max / sqrt(f_max**2 - 1) - 1))  # noqa: F405
    return dict(t0=t0, f_max=f_max, u_min=u_min)
