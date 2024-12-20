#Homework 5

#Part 2:  Graduate Supplement

#2.1.1 Define Hydrogen 1s Orbtial and its Laplacian

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

L = 7

#define a function that computes the normalized hydrogen 1s orbital

def psi_1s(x, y, z, Z=1, a0=1):
    """
    Computes the normalized hydrogen 1s orbital at a given point

    Parameters:
        x: float
            x coordinate
        y: float
            y coordinate
        z: float
            z coordinate
        Z: int
            atomic number of the nucleus
        a0: int
            Bohr radius
    
    Returns:
        float: Normalized hydrogen 1s orbital at (x, y, z)
    """

    coeff = 1 / np.sqrt(np.pi)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    return coeff * np.exp(-r)



#define the expression for the laplacian of the 1s orbitial

def laplacian_psi_1s(x, y, z, Z=1, a0=1):
    """
    Computes the laplacian of the 1s orbtial at (x, y, z)
    
    Parameters:
        x: float
            x coordinate
        y: float
            y coordinate
        z: float
            z coordinate
        Z: int
            atomic number of the nucleus
        a0: int
            Bohr radius
    
    Returns:
        float: Normalized hydrogen 1s orbital at (x, y, z)
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) 
    first_derivative = -psi_1s(x, y, z)           #First derivative of psi_1s
    second_derivative = psi_1s(x, y, z)           #Second derivative of psi_1s

    sum = second_derivative + ((2 / r) * first_derivative)

    return sum


#2.1.2 Compute the Diagonal Kinetic Energy Matrix Element Using Random Sampling


#Set the random seed for reproducibility
np.random.seed(42)

#Create a list with of number of points
n_list = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]



def rando_sample_kii(n_points):
    """
    Uses random sampling to calculate the diagonal kinetic energy matrix element (Kii)

    Parameters:
        n_points: list[ints]
            A list that contains the number of points used during the randoming sampling calculation

    Returns:
        list[float]
            A list that contains Kii as a function of the number of points used during the random sampling.
    """

    integral = []

    for points in n_points:
        x = np.random.uniform(-L, L, points)
        y = np.random.uniform(-L, L, points)
        z = np.random.uniform(-L, L, points)

        integrand = (-1 / 2) * psi_1s(x, y, z) * laplacian_psi_1s(x, y, - z)
        integral.append(np.mean(integrand) * 8 * (L ** 3))
    
    return integral

rando_diagonal_ke_matrix = rando_sample_kii(n_list)

plt.plot(n_list, rando_diagonal_ke_matrix)
plt.xlabel('Number of points sampled')
plt.ylabel('Integral Value')
plt.xscale('log')
plt.title('Diagonal Kinetic Energy Matrix Element (Random Sampling)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(True)
plt.show()

#2.1.3 Compute the Diagonal Kinetic Energy Matrix Element Using Importance Sampling


def important_sample_kii(n_points):
    """
    Uses importance sampling to calculate the diagonal kinetic energy matrix element (Kii)

    Parameters:
        n_points: list[ints]
            A list that contains the number of points used during the importance sampling calculation

    Returns:
        list[float]
            A list that contains Kii as a function of the number of points used during the importance sampling.
    """

    integral = []

    for points in n_points:
        x = norm.rvs(size=points)
        y = norm.rvs(size=points)
        z = norm.rvs(size=points)
        
        numer = (-1 / 2) * psi_1s(x, y, z) * laplacian_psi_1s(x, y, z)
        denom = norm.pdf(x) * norm.pdf(y) * norm.pdf(z)
        integrand = numer / denom

        integral.append(np.mean(integrand))


    return integral



#Compute Kii with importance sampling
important_diag_ke_matrix = important_sample_kii(n_list)
# print(important_diag_ke_matrix[-1])

plt.plot(n_list, important_diag_ke_matrix)
plt.xlabel('Number of points sampled')
plt.ylabel('Integral Value')
plt.xscale('log')
plt.title('Diagonal Kinetic Energy Matrix Element (Importance Sampling)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(True)
plt.show()

#2.1.4 Compute the Off-Diagonal Kinetic Energy Matrix Element Using Random Sampling


def rando_off_diag(n_list):
    """
    Uses random sampling to calculate the off-diagonal kinetic energy matrix element (Kij)

    Parameters:
        n_points: list[ints]
            A list that contains the number of points used during the random sampling calculation

    Returns:
        list[float]
            A list that contains Kij as a function of the number of points used during the random sampling.
    """

    integral = []

    for point in n_list:
        x = np.random.uniform(-L, L, point)
        y = np.random.uniform(-L, L, point)
        z = np.random.uniform(-L, L, point)

        integrand =  (-1 / 2) * psi_1s(x, y, z + 0.7) * laplacian_psi_1s(x, y, z - 0.7)
        integral.append(np.mean(integrand) * 8 * (L ** 3))

    return integral


#Compute Kij with random sampling
rando_off_diag_ke_matrix = rando_off_diag(n_list)
# print(rando_off_diag_ke_matrix[-1])

plt.plot(n_list, rando_off_diag_ke_matrix)
plt.xlabel('Number of points sampled')
plt.ylabel('Integral Value')
plt.xscale('log')
plt.title('Off-Diagonal Kinetic Energy Matrix Element (Random Sampling)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(True)
plt.show()


#2.1.5 Compute the Off-Diagonal Kinetic Energy Matrix Element Using Importance Sampling

def important_off_diag(n_points):
    """
    Uses importance sampling to calculate the diagonal kinetic energy matrix element (Kij)

    Parameters:
        n_points: list[ints]
            A list that contains the number of points used during the importance sampling calculation

    Returns:
        list[float]
            A list that contains Kij as a function of the number of points used during the importance sampling.
    """
    
    integral = []

    for points in n_points:
        x = norm.rvs(size=points)
        y = norm.rvs(size=points)
        z = norm.rvs(size=points)
        
        numer = (-1 / 2) * psi_1s(x, y, z + 0.7) * laplacian_psi_1s(x, y, z - 0.7)
        denom = norm.pdf(x) * norm.pdf(y) * norm.pdf(z)
        integrand = numer / denom

        integral.append(np.mean(integrand))


    return integral


#Compute Kij with importance sampling
important_off_diag_ke_matrix = important_off_diag(n_list)


plt.plot(n_list, important_off_diag_ke_matrix)
plt.xlabel('Number of points sampled')
plt.ylabel('Integral Value')
plt.xscale('log')
plt.title('Off-Diagonal Kinetic Energy Matrix Element (Important Sampling)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(True)
plt.show()