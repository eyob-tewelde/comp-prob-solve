#homework-2-1

#Part 1: Argon Dimer


#Create a function that represents the Lennard-Jones formula

def lennard_jones(r, epsilon=0.01, sigma=3.4):
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

result = minimize(
    lennard_jones,
    x0=4,
    method='Nelder-Mead',
    tol=1e-6
)

print(result) #Lennard-Jones potential for two Ar atoms is minimized at 3.816 Angstroms


#Plot the Lennard-Jones potential and equilibrium distance
x_axis = np.linspace(3, 6, 100)
plt.plot(x_axis, lennard_jones(x_axis))
plt.axvline(x=3.816, color ='r', linestyle="--", linewidth='1')
plt.axhline(y=0, color="gray", linestyle="--", linewidth='1')
plt.xlim(right=6)
plt.xlabel("$r\ (Å)$")
plt.ylabel("$V(r)$")
plt.legend()
plt.title("Lennard-Jones Potential for $Ar_2$")
plt.show()





