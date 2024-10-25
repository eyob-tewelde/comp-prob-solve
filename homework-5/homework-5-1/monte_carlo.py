#Homework 5

#Part 1: Monte Carlo Integration of the Overlap Integral between Two Hydrogen 2p Orbitals

#Define the wavefunction of hydrogen's 2pz orbital in Cartesian coordinates

import numpy as np
import matplotlib.pyplot as plt


#1.1.1 Create a function that computes the wavefunction of the 2pz orbital at (x, y, z)

def psi_2p_z(x, y, z):
    """
    Computes the wavefunction of the 2pz orbital of hydrogen at (x, y, z)

    Parameters:
        x: float or int
            X coordinate
        y: float or int
            Y coordinate
        z: float or int
            Z coordinate

    Returns:
        float or int:
            The value of the wavefunction at (x, y, z)
    """

    orbital_coeff = 1 / (4 * np.sqrt(2 * np.pi))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)           #Convert r to Cartesian coordinates
    cos = z / r                                     #Convert cos(theta) to cartesian coords

    return orbital_coeff * r * cos * np.exp(- r / 2)


#1.1.2 Use random sampling to calculate the overlap integral

# Set the random seed for reproducibility
np.random.seed(42)

def rando_sample(n_points_list, a, b):
    """
    Uses random sampling to calculate the overlap integral between two 2pz orbtials

    Parameters:
        n_points_list: list[ints]
            A list that contains the number of points used during the randoming sampling calculation
        a: int
            Integral bound 1
        b: int
            Integral bound 2
    Returns:
        list[float]
            A list that contains the overlap integrals as a function of the number of points used during the random sampling.
    
    """
    # Create lists to store the results
    integral = []

    # Loop over the number of points to sample
    for n_points in n_points_list:
        x = np.random.uniform(a, b, n_points)
        y = np.random.uniform(a, b, n_points)
        z = np.random.uniform(a, b, n_points)
        integrand = psi_2p_z(x, y, z + 1) * psi_2p_z(x, y, z - 1)
        integral.append(8 * np.mean(integrand) * (b - a)**3)

    return integral
    
#Create a list with of number of points
n_list = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

#Compute the overlap integral with random sampling
two_pz_overlap = rando_sample(n_list, 0, 20)


#1.1.3 Use importance sampling to compute the overlap integral

from scipy.stats import expon
x = 0
y = 0
z = np.linspace(0, 7, 1000)

def importance_sample(n_points_list, sep_dist=1):
    """
    Uses importance sampling to calculate the overlap integral between two 2pz orbtials

    Parameters:
        n_points_list: list[ints]
            A list that contains the number of points used during the randoming sampling calculation
        sep_dist: float
            The seperation between the two 2pz orbtials
    Returns:
        list[float]
            A list that contains the overlap integrals as a function of the number of points used during the random sampling.
    
    """
    # Create lists to store the results
    integral = []

    # Loop over the number of points to sample
    for n_points in n_points_list:
        x = expon.rvs(size=n_points, scale=1)
        y = expon.rvs(size=n_points, scale=1)
        z = expon.rvs(size=n_points, scale=1)
        numer = psi_2p_z(x, y, z + sep_dist) * psi_2p_z(x, y, z - sep_dist)
        denom = expon.pdf(x) * expon.pdf(y) * expon.pdf(z)
        integrand = numer / denom
        integral.append(8 * np.mean(integrand))

    return integral

#Compute the overlap integral with importance sampling
important_sample_overlap = importance_sample(n_list)



#1.1.4 Plot the overlap integral as a function of separation distance

#Create a list of seperate distances
sep_dist = np.arange(0.5, 20, 0.5)

#Number of points used to calculate overlap integral vs seperation distance
n_point = [1000000]

overlap_vs_sep_dist = []
for r in sep_dist:
    overlap = importance_sample(n_point, r)
    overlap_vs_sep_dist.append(overlap)
