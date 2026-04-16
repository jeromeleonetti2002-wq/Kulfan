from kulfan_cst import build_cst_airfoil_from_camber_thickness, build_cst_airfoil_from_geometry, compute_camber_thickness_coeffs, compute_camber_thickness_coeffs, cst_with_coeffs, get_max_camber, wrapper, residual
from kulfan_cst import compute_camber_thickness_coeffs, build_camber_thickness_distributions, build_original_camber_thickness_distributions 
from kulfan_cst import get_max_thickness
from cst_interpolation import cst_interpolation_from_geometry
from kulfan_cst import read_airfoil_data

from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import os

def test_cst_with_coeffs(x_data, y_data):

    x_plus, y_plus, x_minus, y_minus, result = build_cst_airfoil_from_geometry(x_data, y_data)

    plt.figure(figsize=(10,5))
    plt.plot(x_plus, y_plus, label='Fitted Upper Surface', color='red', linestyle='solid')
    plt.plot(x_minus, y_minus, label='Fitted Lower Surface', color='blue', linestyle='solid')
    plt.plot(x_data, y_data, label='Original Airfoil', color='black', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('CST Airfoil Fitting')
    plt.savefig('cst_airfoil_fitting.png')  # Save the figure as a PNG file
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(x_data, residual(result.x, x_data, y_data), label='Residuals', color='green')
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.legend()
    plt.title('Residuals of CST Airfoil Fitting')
    plt.savefig('cst_airfoil_residuals.png')  # Save the figure as a PNG file
    plt.show()

def test_cst_camber_thickness(x_data, y_data):
    x_plus, y_camber, x_minus, y_thickness, airfoil_camber, airfoil_thickness = build_cst_airfoil_from_camber_thickness(x_data, y_data)

    plt.subplot(2, 1, 1)
    plt.plot(x_minus, y_camber, label='Fitted Camber', color='red', linestyle='solid')
    plt.plot(x_plus, airfoil_camber, label='Original Camber', color='orange', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('Camber')
    plt.legend()
    plt.title('Camber Distributions')

    plt.subplot(2, 1, 2)
    plt.plot(x_minus, y_thickness, label='Fitted Thickness', color='blue', linestyle='solid')
    plt.plot(x_plus, airfoil_thickness, label='Original Thickness', color='cyan', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('Thickness')
    plt.legend()
    plt.title('Thickness Distribution')
    plt.show()

def test_cst_interpolation(x_data_1, y_data_1, x_data_2, y_data_2, fact):
    x_int, y_int = cst_interpolation_from_geometry(x_data_1, y_data_1, x_data_2, y_data_2, fact)

    plt.figure(figsize=(10,5))
    plt.plot(x_int, y_int, label='Interpolated Airfoil', color='purple', linestyle='solid')
    plt.plot(x_data_1, y_data_1, label='Airfoil 1', color='red', linestyle='dashed')
    plt.plot(x_data_2, y_data_2, label='Airfoil 2', color='blue', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('CST Airfoil Interpolation')
    plt.savefig('cst_airfoil_interpolation.png')  # Save the figure as a PNG file
    plt.show()

if __name__ == "__main__":
    x_airfoil1, y_airfoil1 = read_airfoil_data('naca4412_100_points.dat')
    x_airfoil2, y_airfoil2 = read_airfoil_data('2032c.dat')

    ############ approximation with CST #############
    #test_cst_with_coeffs(x_airfoil1, y_airfoil1)
    #test_cst_camber_thickness(x_airfoil1, y_airfoil1)

    ############ Camber and thickness distributions #############
    #max_thickness, max_thickness_x = get_max_thickness(x_airfoil1, y_airfoil1, method='original')
    #max_camber, max_camber_x = get_max_camber(x_airfoil1, y_airfoil1, method='original')

    ############ interpolation between two airfoils #############
    test_cst_interpolation(x_airfoil1, y_airfoil1, x_airfoil2, y_airfoil2, 0.5)

