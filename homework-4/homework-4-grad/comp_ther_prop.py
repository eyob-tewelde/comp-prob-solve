from comp_part_func import two_part_partition
from ce_thermo import internal_energy
import numpy as np
import pandas as pd

T = np.linspace(10, 1000, 1000)

ie = internal_energy(two_part_partition(1000), T)

def cv(inter_energy, temperature):
    """
    
    """
    return np.gradient(inter_energy, temperature)

heat_cap = cv(ie, T)
dic = {
    "Temperature": T,
    "Internal energy": ie,
    "Cv": heat_cap
}

df = pd.DataFrame(dic)

df.to_csv("cv_ie_vs_temp.csv", index=False)