import numpy as np


def compute_modulus(e, s, a=0.66, f_yn=345.):
    """ Returns the elastic modulus from the stress-strain data based on the interval [0, a * f_yn].

    :param np.ndarray e: (n, ) Strain data
    :param np.ndarray s: (n, ) Stress data
    :param float a: Ratio of the nominal yield stress for the upper bound
    :param float f_yn: Nominal yield stress
    :return float: Elastic modulus

    Method taken from RLMTP: https://c4science.ch/source/rlmtp.git
    """

    # Consider the data up to 2/3 of f_yn
    s_abs = np.abs(s)
    i_limit = len(s_abs) - 1
    for i, si in enumerate(s_abs):
        if si > a * f_yn:
            i_limit = i - 1
            break
    # Linear fit based on the data up-to i_limit, modulus is first value
    pf = np.polyfit(e[:i_limit], s[:i_limit], 1)
    return pf[0]


def compute_modulus_avg(data, a=0.66, f_yn=345.):
    e_all = []
    for d in data:
        e_all.append(compute_modulus(d['e_true'], d['Sigma_true'], a=a, f_yn=f_yn))
    return np.mean(e_all)
