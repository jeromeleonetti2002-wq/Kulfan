import os
import numpy as np
import math
import matplotlib.pyplot as plt  
from scipy.optimize import least_squares

from kulfan_cst import cst_with_coeffs, wrapper, residual, compute_camber_thickness_coeffs, build_camber_thickness_distributions, build_original_camber_thickness_distributions 
from kulfan_cst import build_cst_airfoil_from_geometry, build_cst_airfoil_from_camber_thickness, get_max_camber, get_max_thickness

def coeffs_linear_interpolation(fact, coeffs_0, coeffs_1):
    return coeffs_0 + (coeffs_1 - coeffs_0) * fact

def cst_coeffs_interpolation(fact, coeffs_1, coeffs_2, method = 'linear'):
    if method == 'linear':
        coeffs_interp = coeffs_linear_interpolation(fact, coeffs_1, coeffs_2)
    return coeffs_interp

def cst_interpolation_from_geometry(x_data_1, y_data_1, x_data_2, y_data_2, fact):
    x_plus_1, y_plus_1, x_minus_1, y_minus_1, result_1 = build_cst_airfoil_from_geometry(x_data_1, y_data_1)
    x_plus_2, y_plus_2, x_minus_2, y_minus_2, result_2 = build_cst_airfoil_from_geometry(x_data_2, y_data_2)
    coeffs_1 = np.concatenate((result_1.x[:7], result_1.x[7:]))
    coeffs_2 = np.concatenate((result_2.x[:7], result_2.x[7:]))
    coeffs_interp = cst_coeffs_interpolation(fact, coeffs_1, coeffs_2, method='linear')

    x_int_plus, x_int_minus,  = x_plus_1, x_minus_1
    y_int_plus = cst_with_coeffs(x_int_plus, coeffs_interp[:7])
    y_int_minus = cst_with_coeffs(x_int_minus, coeffs_interp[7:])

    x_int = np.concatenate((x_int_plus, x_int_minus))
    y_int = np.concatenate((y_int_plus, y_int_minus))
    
    return x_int, y_int