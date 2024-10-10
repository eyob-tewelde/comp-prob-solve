#Homework 4

#Part 1: Numerical Computation of Work in Thermodynamic Processes

import compute_work_adiabatic
import compute_work_isothermal
import matplotlib.pyplot as plt


plt.plot(compute_work_adiabatic.work_adi_df[1], compute_work_adiabatic.work_adi_df[0], label='Adiabatic Expansion')
plt.plot(compute_work_isothermal.work_iso_df[1], compute_work_isothermal.work_iso_df[0], label='Isothermal Expansion')
plt.xlabel("Final Volume ($m^3$)", labelpad=20)
plt.ylabel("Work (J)", labelpad=20)
plt.title("Work done during an ideal gas expansion", pad=20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(frameon=False, fontsize=13)

plt.show()