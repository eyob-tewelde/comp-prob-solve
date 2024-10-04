#homework-2-2

#Numerical Integration of the Second Virial Coefficient

from optimize_argon_dimer import lennard_jones
import numpy as np
from scipy.integrate import trapezoid
import scipy.constants as const
import matplotlib.pyplot as plt
import pandas as pd

#Define constants and distances
diam = 3.4
r = np.linspace(0.001, 5 * diam, 1000)
k_b = 8.617 * (10 ** -5)


#define function for hard-sphere potential

def hard_sphere(r, diam=3.4):
    """
    Computes the hard-sphere potential.

    Parameters:
        r: float or int
            the distance between two particles
        diam: float or int
            the diameter of the hard-sphere

    Returns:
        int: the hard-sphere potential between the two particles
    """
    if r < diam:
        return 1000
    else:
        return 0


#define function for square-well potential

def square_well(r, diam=3.4, depth=0.01, range=1.5):
    """
    Computes the square-well potential

    Parameters:
        r: float or int
            the distance between two particles
        diam: float or int
            the particle diameter
        depth: float or int
            the well-depth
        range: float or int
            the range of the well

    Returns:
        float or int: the square-well potential between the two particles
    """

    if r < diam:
        return 1000
    elif diam <= r and r < diam * range:
        return -1 * depth
    else:
        return 0

#Compute the second virial coefficient

def second_virial(poten, t):
    """
    Computes the second virial coefficient

    Computes the square-well potential

    Parameters:
        poten: function
            the pair-wise interaction potential used to approximate non-ideal gas interactions
        t: float or int
            the temperature of the system
    Returns:
        float or int: the second virial coefficient
    
    """

    integrand_coeff = -2 * np.pi * const.Avogadro
    integrand = (np.exp(-poten / (k_b * t)) - 1) * (r ** 2)
    

    return integrand_coeff * trapezoid(integrand * r ** 2, r)


#Create a list of potentials for each pair-wise interaction potential
hard_sphere_potential_list = []
for i in r:
    hard_sphere_potential_list.append(hard_sphere(i))

square_well_potential_list = []
for i in r:
    square_well_potential_list.append(square_well(i))

lennard_jones_potential_list = []
for i in r:
    lennard_jones_potential_list.append(lennard_jones(i))


#Convert lists to numpy array
hard_sphere_potential = np.array(hard_sphere_potential_list)
square_well_potential = np.array(square_well_potential_list)
lennard_jones_potential = np.array(lennard_jones_potential_list)


#
b2v_hard_sphere_100k = second_virial(hard_sphere_potential, 100)
b2v_square_well_100k = second_virial(square_well_potential, 100)
b2v_lennard_jones_100k = second_virial(lennard_jones_potential, 100)

# print(hard_sphere_100k)
# print(square_well_100k)
# print(lennard_jones_100k)

def b2v_vary_t(poten, t_min=100, t_max=800, num_points=100):
    """
    Computes the B2V for a range of temperatures

    Parameters:
        poten: function
            The potential function for a given pair-wise interaction
        t_min: int
            The minimum temperature of the range
        t_max: int
            The maximum temperature of the range
        num_points: int
            The number of points in the temperature range
    
    Returns:
        np.array: numpy array that contains the B2Vs calculated
                  over the temperature range
        temps: numpy array that contains the temperature points used for the calculation

 
    """
    b2vs = []
    temps = np.linspace(t_min, t_max, num_points, endpoint=True)
    for t in temps:
        b2vs.append(second_virial(poten, t))
    
    return np.array(b2vs), temps

#Calculates the B2V for each pair-wise interaction from T = 100K to T = 800K
hard_sphere_range, temps = b2v_vary_t(hard_sphere_potential)
square_well_range, _ = b2v_vary_t(square_well_potential)
lennard_jones_range, _ = b2v_vary_t(lennard_jones_potential)

#Produces a csv file for B2V
data = {
    'Temperature (K)': temps,
    'B2V Hard Sphere': hard_sphere_range,
    'B2V Square Well': square_well_range,
    'B2V Lennard-Jones': lennard_jones_range,
}

b2v_df = pd.DataFrame(data)
csv_file_path = "homework-2-2/B2V_vs_Temperature.csv"
b2v_df.to_csv(csv_file_path, index=False)


#Plot the B2V for each pair-wise interaction as a function of temperature
plt.plot(temps, hard_sphere_range, color = "blue", label="Hard-Sphere Potential")
plt.plot(temps, square_well_range, color = "orange", label="Square-Well Potential")
plt.plot(temps, lennard_jones_range, color = "green", label="Lennard-Jones Potential")

plt.xlabel("Temperature (K)")
plt.ylabel("Second Virial Coefficient ($Å^3/mol$)")
plt.title("Second Virial Coefficient vs T ")
plt.axhline(y=0, color="gray", linestyle="--", linewidth='1', label = "")
plt.xlim(100, 800)
plt.legend()
plt.show()
