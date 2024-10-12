#Homework 4

#Part 3: Graduate Supplement

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.constants import h, k
from scipy.integrate import trapezoid

from optimize_argon_dimer import lennard_jones

#constants

T = np.linspace(10, 1000, 1000)
eps = 1.650242e-21
sig = 3.4e-10

beta = 1 / (k * T) 
m = 6.633 * 10 ** -26               
lam = np.sqrt((beta * (h ** 2)) / (2 * np.pi * m))
coeff = (1 / (h ** 6)) * (1 / (lam ** 6))

vol = 1 * 10 ** -28

r_min = 8.8 * 10 ** -11
r_max = np.cbrt(vol)


#Define partition functions
def two_part_partition(grid_points):
    """
    
    """

    r = np.linspace(r_min, r_max, grid_points)
    integrand = np.exp(-beta * lennard_jones(r, eps, sig))

    z = trapezoid(integrand, r)

    return coeff * z

#Compute partition function
part_func = two_part_partition(1000)
dic = {
    "Temperature": T,
    "Partition Functions": part_func
}

df = pd.DataFrame(dic)
df.to_csv("partition_function_v_temp.csv", index=False)
# plt.plot(T, part_func)
# plt.show()