#Homework 5

#Part 1: Monte Carlo Integration of the Overlap Integral between Two Hydrogen 2p Orbitals

#Define the wavefunction of hydrogen's 2pz orbital in Cartesian coordinates

import numpy as np
import matplotlib.pyplot as plt


#Create a function that computes the wavefunction of the 2pz orbital at (x, y, z)

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
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    cos = z / r

    return orbital_coeff * r * cos * np.exp(- r / 2)



#Use random sampling to calculate the overlap integral

# Set the random seed for reproducibility
np.random.seed(42)

def rando_sample(n_points_list, a, b):
    """
    
    
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
    
n_list = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
#two_pz_overlap = rando_sample(n_list, 0, 20)

# Plot the results
#plt.plot(n_list, two_pz_overlap)
#plt.axhline(0.74, color="black", linestyle="--")
#plt.xlabel('Number of points sampled')
#plt.ylabel('Integral Value')
#plt.xscale('log')
#plt.title("Overlap integral of Two H atom 2pz orbitals")
#plt.show()


#Use importance sampling to compute the overlap integral

from scipy.stats import expon
x = 0
y = 0
z = np.linspace(0, 7, 1000)


integrand_value = psi_2p_z(x, y, z + 1) * psi_2p_z(x, y, z - 1)
importance_sampling = expon.pdf(z)
plt.plot(z, integrand_value, label = "integrand")
plt.plot(z, importance_sampling, label = "importance sampling")
plt.legend()