#Homework 4

#Part 2: Thermodynamic Properties of Ce(3) in Different Environments

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, eV
import pandas as pd

#Set constants and temperature grid
boltz = k / eV
temp = np.linspace(300, 2000, 1000)

#Internal energy
def internal_energy(Z, T):
    """
    Compute the internal energy

    Parameters:
        Z: np.array
            Partition function
        T: np.array
            Temperature
    Returns
        np.array:
            Array that contains the internal energy as a function of temperature.

    """
    return -np.gradient(np.log(Z), 1 / (boltz * T))

#Free energy
def free_energy(Z, T):
    """
    Compute the free energy

    Parameters:
        Z: np.array
            Partition function
        T: np.array
            Temperature
    Returns
        np.array:
            Array that contains the free energy as a function of temperature.
    """

    return -boltz * T * np.log(Z)

#Entropy
def entropy(A, T):
    """
    Compute the entropy

    Parameters:
        A: np.array
            Free energy
        T: np.array
            Temperature
    Returns
        np.array:
            Array that contains the entropy as a function of temperature.
    """

    return -np.gradient(A, T)

#Case 1: Isolated Ce(3)

#Partition function
def iso_part(T):
    """
    Compute the partition function of an isolated Ce3+ ion w/ 14 degenerate states of E = 0

    Parameters:
        T: np.array
            Temperature
    Returns
        np.array:
            Array that contains the partition function as a function of temperature.
    """

    return 14 * np.exp(0 / (boltz * T))


#Compute the partiton function, internal energy, free energy, and entropy for an isolated Ce3+ ion
iso_partition = iso_part(temp)
iso_internal_energy = internal_energy(iso_partition, temp)
iso_free_energy = free_energy(iso_partition, temp)
iso_entropy = entropy(iso_free_energy, temp)



#Case 2: Ce(3) with spin orbit coupling (SOC)

#Partition function
def soc_part(T):
    """
    Compute the partition function of a Ce3+ ion w/ SOC
    6 degenerate states of E = 0
    8 degenerate states of E = 0.28

    Parameters:
        T: np.array
            Temperature
    Returns
        np.array:
            Array that contains the partition function as a function of temperature.
    """
    ground = 6
    first = 8 * np.exp(-0.28 / (boltz * T))

    return ground + first

#Compute the partiton function, internal energy, free energy, and entropy for a Ce3+ ion w/ SOC
soc_partition = soc_part(temp)
soc_internal_energy = internal_energy(soc_partition, temp)
soc_free_energy = free_energy(soc_partition, temp)
soc_entropy = entropy(soc_free_energy, temp)




#Case 3: Ce(3) with SOC and crystal field splitting (CFS)

#Partition function
def soc_cfs_part(T):
    """
    Compute the partition function of a Ce3+ ion w/ SOC and CFS.
    4 degenerate states of E = 0
    2 degenerate states of E = 0.12
    2 degenerate states of E = 0.25
    4 degenerate states of E = 0.32
    2 degenerate states of E = 0.46

    Parameters:
        T: np.array
            Temperature
    Returns
        np.array:
            Array that contains the partition function as a function of temperature.

    """
    kbt = boltz * T

    ground = 4
    first = 2 * np.exp(-0.12 / (kbt))
    second = 2 * np.exp(-0.25 / (kbt))
    third = 4 * np.exp(-0.32 / (kbt))
    fourth = 2 * np.exp(-0.46 / (kbt))

    return ground + first + second + third + fourth


#Compute the partiton function, internal energy, free energy, and entropy for a Ce3+ ion w/ SOC and CFS
soc_cfs_partition = soc_cfs_part(temp)
soc_cfs_internal_energy = internal_energy(soc_cfs_partition, temp)
soc_cfs_free_energy = free_energy(soc_cfs_internal_energy, temp)
soc_cfs_entropy = entropy(soc_cfs_free_energy, temp)


#Write a csv file containing internal energy, free energy, and entropy vs. temperature for all three cases
index_labels = ["Isolated Ce(3)", "Ce(3) w/ SOC", "Ce(3) w/ SOC and CFS"]

all_cases = {
    "E (iso)": iso_internal_energy,
    "E (SOC)": soc_internal_energy,
    "E (SOC+CFS)":soc_cfs_internal_energy,
    "A (iso)": iso_free_energy, 
    "A (SOC)": soc_free_energy, 
    "A (SOC+CFS": soc_cfs_free_energy,
    "S (iso)": iso_entropy, 
    "S (SOC)": soc_entropy, 
    "S (SOC+CFS":soc_cfs_entropy
}

df = pd.DataFrame(all_cases, index=temp)
df.to_csv("thermo_prop_ce.csv")




#Plot the thermodynamic properties vs T for each case
x_axis = range(300, 2201, 200)
fig, axs = plt.subplots(3, 1, figsize=(6,8))

axs[0].plot(temp, iso_internal_energy, label="Isolated Ce(3)")
axs[0].plot(temp, soc_internal_energy, label="Ce(3) w/ SOC")
axs[0].plot(temp, soc_cfs_internal_energy, label="Ce(3) w/ SOC and CFS")
axs[0].set_title("Internal energy vs Temperature")
axs[0].set_ylabel("Internal energy (E) (eV)")
axs[0].set_xticks(x_axis)
axs[0].set_xlim(300, 2000)
axs[0].set_xlabel("Temperature (K)")



axs[1].plot(temp, iso_free_energy)
axs[1].plot(temp, soc_free_energy)
axs[1].plot(temp, soc_cfs_free_energy)
axs[1].set_title("Free energy vs Temperature")
axs[1].set_ylabel("Free energy (A) (eV)")
axs[1].set_xticks(x_axis)
axs[1].set_xlim(300, 2000)
axs[1].set_xlabel("Temperature (K)")



axs[2].plot(temp, iso_entropy)
axs[2].plot(temp, soc_entropy)
axs[2].plot(temp, soc_cfs_entropy)
axs[2].set_title("Entropy vs Temperature")
axs[2].set_ylabel("Entropy (S) (eV/K)")
axs[2].set_xticks(x_axis)
axs[2].set_xlim(300, 2000)
axs[2].set_xlabel("Temperature (K)")

plt.tight_layout()
fig.legend(loc="outside center left", frameon=False, fontsize=11)
plt.show()
