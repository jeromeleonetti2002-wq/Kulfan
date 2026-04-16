import os
import numpy as np
import math
import matplotlib.pyplot as plt  
from scipy.optimize import least_squares

def read_airfoil_data(filename):
    ########## Reading airfoil data from file ##########
    x_airfoil = []
    y_airfoil = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, filename), 'r') as f:
        text = f.read()
        words = text.split()
        for i in range(1, len(words), 2):
            x_airfoil.append(float(words[i]))
            y_airfoil.append(float(words[i+1]))
    return x_airfoil, y_airfoil

######### Functions to build Bernstein polynomials and compute CST coefficients #########

def build_pascal_matrix(n):
    A = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1):
            A[i][j] = math.comb(i, j)
    return A

#A = build_pascal_matrix(6)
#print(A)

def build_berstein_polynomial(n):
    A = build_pascal_matrix(n)
    f1 = lambda x : x
    f2 = lambda x : 1-x
    BP = []
    for i in range(n+1):
        BP.append(lambda x, i=i: A[n][i]*f1(x)**i*f2(x)**(n-i))
        # x = np.linspace(0,1,10)
        # y = BP[i](x)
        # print(y)
    return BP
    
# bp = build_berstein_polynomial(6)
# print(bp)
# print(len(bp))

########### First test ##############

def cst(x, n, method):
    bp = build_berstein_polynomial(n)
    y = []
    for i in range(len(x)):
        c = 0
        for j in range(n+1):
            if method == 'plus':
                c += (1/2 + j)/(3/2 + n)*bp[j](x[i])
            elif method == 'minus':
                c -= (1/2 + j)/(3/2 + n)*bp[j](x[i])
        c *= x[i]**(1/2)*(1-x[i])
        y.append(c)
    return y

############## Optimization of coefficients using least squares ################

from scipy.optimize import least_squares

def cst_with_coeffs(x, coeffs):
    n = len(coeffs) - 1
    bp = build_berstein_polynomial(n)
    y = []
    for xi in x:
        s = sum(coeffs[j] * bp[j](xi) for j in range(n+1)) #Shape function with custom coefficients
        y.append(math.sqrt(xi) * (1 - xi) * s) #Class function multiplied by shape function
    return y

def residual(coeffs, x_data, y_data):
    coeffs_upper = coeffs[:7]
    coeffs_lower = coeffs[7:]
    x_plus = x_data[:len(x_data)//2]
    x_minus = x_data[len(x_data)//2:]
    y_airfoil_upper = y_data[:len(y_data)//2]
    y_airfoil_lower = y_data[len(y_data)//2:] 
    y_upper_pred = cst_with_coeffs(x_plus, coeffs_upper)
    y_lower_pred = cst_with_coeffs(x_minus, coeffs_lower)
    res_upper = np.array(y_upper_pred) - np.array(y_airfoil_upper)
    res_lower = np.array(y_lower_pred) - np.array(y_airfoil_lower)
    return np.concatenate([res_upper, res_lower])

def wrapper(coeffs, x_data, y_data):
    def residual_func(c):
        return residual(c, x_data, y_data)
    result = least_squares(residual_func, coeffs)
    return result
  
def build_cst_airfoil_from_geometry(x_data, y_data):
    y_airfoil_upper = y_data[:len(x_data)//2]
    y_airfoil_lower = y_data[len(x_data)//2:]

    initial_coeffs_upper = [(0.5 + j)/(1.5 + 6) for j in range(7)]
    initial_coeffs_lower = [-(0.5 + j)/(1.5 + 6) for j in range(7)]
    initial_guess = np.array(initial_coeffs_upper + initial_coeffs_lower)

    result = wrapper(initial_guess, x_data, y_data)
    optimized_coeffs_upper = result.x[:7]
    optimized_coeffs_lower = result.x[7:]

    print("Optimized coefficients for upper surface:", optimized_coeffs_upper)
    print("Optimized coefficients for lower surface:", optimized_coeffs_lower)
 
    x_plus = x_data[:len(x_data)//2]
    x_minus = x_data[len(x_data)//2:]
    y_plus = cst_with_coeffs(x_plus, optimized_coeffs_upper)
    y_minus = cst_with_coeffs(x_minus, optimized_coeffs_lower)

    return x_plus, y_plus, x_minus, y_minus, result
# The CST does not include a trailing edge gap term which explain the large residuals at the trailing edge. The CST is not able to capture the sharp trailing edge of the airfoil, which leads to significant discrepancies between the fitted curve and the original airfoil data in that region.


############### Thickness and Camber distributions ##############


def compute_camber_thickness_coeffs(x_data, y_data):
    initial_coeffs_upper = [(0.5 + j)/(1.5 + 6) for j in range(7)]
    initial_coeffs_lower = [-(0.5 + j)/(1.5 + 6) for j in range(7)]
    initial_guess = np.array(initial_coeffs_upper + initial_coeffs_lower)

    result = wrapper(initial_guess, x_data, y_data)

    optimized_coeffs_upper = result.x[:7]
    optimized_coeffs_lower = result.x[7:]

    Camber_coeffs = (optimized_coeffs_upper + optimized_coeffs_lower) / 2
    Thickness_coeffs = (optimized_coeffs_upper - optimized_coeffs_lower) / 2
    
    return Camber_coeffs, Thickness_coeffs

def build_camber_thickness_distributions(x_data, y_data):
    Camber_coeffs, Thickness_coeffs = compute_camber_thickness_coeffs(x_data, y_data) 
    x_minus = x_data[len(x_data)//2:]  
    y_camber = cst_with_coeffs(x_minus, Camber_coeffs)
    y_thickness = cst_with_coeffs(x_minus, Thickness_coeffs)
    return y_camber, y_thickness

def build_original_camber_thickness_distributions(x_data, y_data):
    # original camber and thickness distributions from the airfoil data
    y_airfoil_upper = y_data[:len(x_data)//2]
    y_airfoil_lower = y_data[len(x_data)//2:]
    airfoil_camber = []
    airfoil_thickness = []
    for i in range(len(x_data)//2):
        camber = (y_airfoil_upper[i] + y_airfoil_lower[-i]) / 2
        thickness = (y_airfoil_upper[i] - y_airfoil_lower[-i]) / 2
        airfoil_camber.append(camber)
        airfoil_thickness.append(thickness)
    return airfoil_camber, airfoil_thickness

def build_cst_airfoil_from_camber_thickness(x_data, y_data):
    Camber_coeffs, Thickness_coeffs = compute_camber_thickness_coeffs(x_data, y_data) 
    x_minus = x_data[len(x_data)//2:]
    x_plus = x_data[:len(x_data)//2]
    y_camber = cst_with_coeffs(x_minus, Camber_coeffs)
    y_thickness = cst_with_coeffs(x_minus, Thickness_coeffs)

    y_airfoil_upper = y_data[:len(x_data)//2]
    y_airfoil_lower = y_data[len(x_data)//2:]
    airfoil_camber = []
    airfoil_thickness = []
    for i in range(len(x_data)//2):
        camber = (y_airfoil_upper[i] + y_airfoil_lower[-i]) / 2
        thickness = (y_airfoil_upper[i] - y_airfoil_lower[-i]) / 2
        airfoil_camber.append(camber)
        airfoil_thickness.append(thickness)

    return x_plus, y_camber, x_minus, y_thickness, airfoil_camber, airfoil_thickness


def get_max_thickness(x_data, y_data, method='cst'):
    x_minus = x_data[len(x_data)//2:]
    if method == 'cst':
        _, y_thickness = build_camber_thickness_distributions(x_data, y_data)
        peak_index = int(np.argmax(y_thickness))
        max_thickness = 2 * y_thickness[peak_index]
        max_thickness_x = x_minus[peak_index]
        print(f"CST max thickness: {max_thickness} at x = {max_thickness_x}")

    elif method == 'original':
        _, y_thickness = build_original_camber_thickness_distributions(x_data, y_data)
        peak_index = int(np.argmax(y_thickness))
        max_thickness = 2 * y_thickness[peak_index]
        max_thickness_x = x_minus[-peak_index]
        print(f"Original max thickness: {max_thickness} at x = {max_thickness_x}")

    return max_thickness, max_thickness_x

def get_max_camber(x_data, y_data, method='cst'):
    x_minus = x_data[len(x_data)//2:]
    if method == 'cst':
        y_camber, _ = build_camber_thickness_distributions(x_data, y_data)
        peak_index = int(np.argmax(y_camber))
        max_camber = y_camber[peak_index]
        max_camber_x = x_minus[-peak_index]
        print(f"CST max camber: {max_camber} at x = {max_camber_x}")

    elif method == 'original':
        y_camber, _ = build_original_camber_thickness_distributions(x_data, y_data)
        peak_index = int(np.argmax(y_camber))
        max_camber = y_camber[peak_index]
        max_camber_x = x_minus[-peak_index]
        print(f"Original max camber: {max_camber} at x = {max_camber_x}")

    return max_camber, max_camber_x

    
    
    
