"""@package sqp_linsearch
Constraints on the original Voce-Chaboche model for limited information optimization.
"""
from numdifftools import nd_algopy as nda
import numpy as np


def g3_vco_upper(x, constants, variables):
    """ Constraint on the maximum ratio of stress at saturation to initial yield stress for the original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_basic_param = constants['n_basic_param']
    c_start = n_basic_param
    g_start = n_basic_param + 1
    max_hardening_to_yield = constants['rho_yield_sup']
    n_backstresses = int((len(x) - n_basic_param) / 2)
    sy0 = x[n_basic_param - 3]
    q_inf = x[n_basic_param - 2]
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = c_start + 2 * i
        gamma_ind = g_start + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return (sy0 + q_inf + sum_ck_gammak) / sy0 - max_hardening_to_yield


def g3_vco_lower(x, constants, variables):
    """ Constraint on the minimum ratio of stress at saturation to initial yield stress for the original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_basic_param = constants['n_basic_param']
    c_start = n_basic_param
    g_start = n_basic_param + 1
    min_hardening_to_yield = constants['rho_yield_inf']
    n_backstresses = int((len(x) - n_basic_param) / 2)
    sy0 = x[n_basic_param - 3]
    q_inf = x[n_basic_param - 2]
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = c_start + 2 * i
        gamma_ind = g_start + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return -(sy0 + q_inf + sum_ck_gammak) / sy0 + min_hardening_to_yield


def g4_vco_upper(x, constants, variables):
    """ Constraint on the maximum ratio of isotropic to combined isotropic/kinematic hardening at saturation for the
    original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_basic_param = constants['n_basic_param']
    c_start = n_basic_param
    g_start = n_basic_param + 1
    iso_kin_ratio_max = constants['rho_iso_sup']
    q_inf = x[n_basic_param - 2]
    n_backstresses = int((len(x) - n_basic_param) / 2)
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = c_start + 2 * i
        gamma_ind = g_start + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return q_inf / (q_inf + sum_ck_gammak) - iso_kin_ratio_max


def g4_vco_lower(x, constants, variables):
    """ Constraint on the minimum ratio of isotropic to combined isotropic/kinematic hardening at saturation for the
    original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_basic_param = constants['n_basic_param']
    c_start = n_basic_param
    g_start = n_basic_param + 1
    iso_kin_ratio_min = constants['rho_iso_inf']
    q_inf = x[n_basic_param - 2]
    n_backstresses = int((len(x) - n_basic_param) / 2)
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = c_start + 2 * i
        gamma_ind = g_start + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return -q_inf / (q_inf + sum_ck_gammak) + iso_kin_ratio_min


def g5_vco_lower(x, constants, variables):
    """ Constraint on the lower bound ratio of gamma_1 to b for the original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_basic_param = constants['n_basic_param']
    g_start = n_basic_param + 1
    b = x[n_basic_param - 1]
    gamma1 = x[g_start]
    gamma_b_ratio_min = constants['rho_gamma_inf']
    return -gamma1 / b + gamma_b_ratio_min


def g5_vco_upper(x, constants, variables):
    """ Constraint on the upper bound ratio of gamma_1 to b for the original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_basic_param = constants['n_basic_param']
    g_start = n_basic_param + 1
    b = x[n_basic_param - 1]
    gamma1 = x[g_start]
    gamma_b_ratio_max = constants['rho_gamma_sup']
    return gamma1 / b - gamma_b_ratio_max


def g6_vco_lower(x, constants, variables):
    """ Constraint on the lower bound ratio of gamma_1 to gamma_2 for the original VC model.

    gamma_1 is always x[5] and gamma_2 is always x[7].

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_basic_param = constants['n_basic_param']
    g_start = n_basic_param + 1
    gamma1 = x[g_start]
    gamma2 = x[g_start + 2]
    gamma_1_2_ratio_min = constants['rho_gamma_12_inf']
    return -gamma1 / gamma2 + gamma_1_2_ratio_min


def g6_vco_upper(x, constants, variables):
    """ Constraint on the upper bound ratio of gamma_1 to gamma_2 for the original VC model.

    gamma_1 is always x[5] and gamma_2 is always x[7].

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_basic_param = constants['n_basic_param']
    g_start = n_basic_param + 1
    gamma1 = x[g_start]
    gamma2 = x[g_start + 2]
    gamma_1_2_ratio_max = constants['rho_gamma_12_sup']
    return gamma1 / gamma2 - gamma_1_2_ratio_max


def g_kin_ratio_vco_lower(x, constants, variables):
    c1 = x[4]
    gamma1 = x[5]
    c2 = x[6]
    gamma2 = x[7]
    gamma_kin_ratio_min = constants['rho_kin_ratio_inf']
    return -(c1 / gamma1) / (c2 / gamma2) + gamma_kin_ratio_min


def g_kin_ratio_vco_upper(x, constants, variables):
    c1 = x[4]
    gamma1 = x[5]
    c2 = x[6]
    gamma2 = x[7]
    gamma_kin_ratio_max = constants['rho_kin_ratio_sup']
    return (c1 / gamma1) / (c2 / gamma2) - gamma_kin_ratio_max


# Gradients and Hessians of all the above constraints
def g3_vco_lower_gradient(x, constants, variables):
    def fun_wrapper(x1): return g3_vco_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g3_vco_upper_gradient(x, constants, variables):
    def fun_wrapper(x1): return g3_vco_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g4_vco_lower_gradient(x, constants, variables):
    def fun_wrapper(x1): return g4_vco_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g4_vco_upper_gradient(x, constants, variables):
    def fun_wrapper(x1): return g4_vco_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g5_vco_lower_gradient(x, constants, variables):
    def fun_wrapper(x1): return g5_vco_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g5_vco_upper_gradient(x, constants, variables):
    def fun_wrapper(x1): return g5_vco_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g6_vco_lower_gradient(x, constants, variables):
    def fun_wrapper(x1): return g6_vco_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


def g6_vco_upper_gradient(x, constants, variables):
    def fun_wrapper(x1): return g6_vco_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


def g_kin_ratio_vco_lower_gradient(x, constants, variables):
    def fun_wrapper(x1): return g_kin_ratio_vco_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


def g_kin_ratio_vco_upper_gradient(x, constants, variables):
    def fun_wrapper(x1): return g_kin_ratio_vco_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


# Hessians

def g3_vco_lower_hessian(x, constants, variables):
    def fun_wrapper(x1): return g3_vco_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g3_vco_upper_hessian(x, constants, variables):
    def fun_wrapper(x1): return g3_vco_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g4_vco_lower_hessian(x, constants, variables):
    def fun_wrapper(x1): return g4_vco_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g4_vco_upper_hessian(x, constants, variables):
    def fun_wrapper(x1): return g4_vco_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g5_vco_lower_hessian(x, constants, variables):
    def fun_wrapper(x1): return g5_vco_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g5_vco_upper_hessian(x, constants, variables):
    def fun_wrapper(x1): return g5_vco_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g6_vco_lower_hessian(x, constants, variables):
    def fun_wrapper(x1): return g6_vco_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g6_vco_upper_hessian(x, constants, variables):
    def fun_wrapper(x1): return g6_vco_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g_kin_ratio_vco_lower_hessian(x, constants, variables):
    def fun_wrapper(x1): return g_kin_ratio_vco_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g_kin_ratio_vco_upper_hessian(x, constants, variables):
    def fun_wrapper(x1): return g_kin_ratio_vco_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess
