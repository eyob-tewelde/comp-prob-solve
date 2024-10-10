#Homework 4

#Part 1: Numerical Computation of Work in Thermodynamic Processes

from scipy.integrate import trapezoid
import numpy as np
import pandas as pd

#Define constants
R = 8.314 #J/mol * K


#Define a function to calculate the work done on an ideal gas during an adiabatic expansion.

def compute_work_adi(v_i, v_f, n, T, y):
    """
    Compute the work done on an ideal gas during an adiabatic expansion.

    Parameters:
        v_i: float or int
            the initial volume of the gas (m^3)
        v_f: float or int
            the final volume of the gas (m^3)
        n: float or int
            the number of moles of gas
        T: float or int
            The temperature of the gas (K)
        y: float or int
            The adiabatic index

    Returns:
        list[(float, float)]: 
            A list of tuples that contain the final volume of the expansion and the 
            associated work done on the ideal gas.
    """
    
    #Calculate the constant that satisfies the initial conditions of the gas so 
    #the pressure can be calculated as a function of volume

    P_i = (n * R * T) / v_i
    constant = P_i * (v_i ** y)

    #Calculate the work done as a function of the final volume.

    work_list = []
    for vol in np.arange(v_i, v_f, .0001):
        volume = np.linspace(v_i, vol, 1000)
        dv = volume[0] - volume[1]

        integrand = constant / (volume ** y)
    
        work_adi = -1 * trapezoid(integrand, volume, dv)
        work_list.append((work_adi, vol))

    return work_list

#Retrieve the work done during adiabatic expansion
work = compute_work_adi(0.1, 0.3, 1, 300, 1.4)

#Set the list[(work, final volume)] to a dataframe
work_adi_df = pd.DataFrame(work)

#Convert dataframe to csv file
work_adi_df.to_csv("work_adi.csv", header=["Work", "Final Volume"], index=False)
