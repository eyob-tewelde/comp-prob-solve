from comp_part_func import two_part_partition
from ce_thermo import internal_energy
import numpy as np
import pandas as pd

#Define temperature range
T = np.linspace(10, 1000, 1000)

#Compute internal energy for an ensemble of two lennard jones particles
ie = internal_energy(two_part_partition(1000), T)

#Define function that computes the constant volume heat capacity
def cv(inter_energy, temperature):
    """
    Compute the constant volume heat capacity

    Parameters:
        inter_energy: numpy.ndarray
            Contains the internal energy as a function of temperature
        temperature: numpy.ndarray
            Contains a range of temperature
    
    Returns:
        numpy.ndarray: heat capacities as a function of temperature
    """

    return np.gradient(inter_energy, temperature)

#Compute the heat capacity
heat_cap = cv(ie, T)

#Define dictionary containing the Temperature, Internal energy, and heat capacity
dic = {
    "Temperature": T,
    "Internal energy": ie,
    "Cv": heat_cap
}

#Write the dictionary to a csv file
df = pd.DataFrame(dic)
df.to_csv("cv_ie_vs_temp.csv", index=False)