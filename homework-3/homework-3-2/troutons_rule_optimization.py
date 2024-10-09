import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pandas as pd

#Import trouton.csv
trouton = pd.read_csv("trouton.csv")

#Sort the data in ascending order boiling points
sorted_trouton = trouton.sort_values(by="T_B (K)")

#Extract the boiling points
sorted_boil = sorted_trouton["T_B (K)"]

#Extract the enthalpies of vaporization and convert to J/mol
sorted_enth_vapor = (sorted_trouton["H_v (kcal/mol)"].values) * 4184

#Define the objective function
def objective(param, sorted_trouton):
    a, b = param
    sqr_residual = (sorted_enth_vapor - (a * sorted_boil + b)) ** 2

    return np.sum(sqr_residual)

#minimize the objective function to find a (entropy of vaporization) and b (the intercept).
result = minimize(objective, [1,1], method="Nelder-Mead", args=(sorted_trouton,))
a_opt, b_opt = result.x

# a = 103.9 b = -4845

#Create dictionary to map colors to substance class
colors = {
    "Perfect liquids":"r",
    "Liquids subject to quantum effects":"b",
    "Imperfect liquids":"g",
    "Metals":'tab:purple'
}

#Create fitted line with calculated parameters
line = a_opt * sorted_boil + b_opt


#Make the plot
plt.plot(sorted_boil, line, label="Linear fit")

added_classes = set()

for index, row in sorted_trouton.iterrows():
    bp = row["T_B (K)"]
    cls = row["Class"]
    enth = row["H_v (kcal/mol)"] * 4184
    clr = colors[cls]

    if cls not in added_classes:
        plt.plot(bp, enth, color=clr, label=cls, marker='o', alpha=0.5)
        added_classes.add(cls)
    else:
        plt.plot(bp, enth, color=clr, marker='o', alpha=0.5)

plt.xlabel("Boiling point (K)", labelpad=20, fontsize=13)
plt.ylabel(r"Enthalpies of Vaporization $(J / mol \cdot K)$", labelpad=20, fontsize=13)
plt.title("Trouton’s Rule Optimization", fontsize=14)


plt.text(100, 280000, r'$H_v = a \cdot T_b + b$', fontsize=13)
plt.text(100, 250000, r'a = $103.9 $ J/mol $\cdot K$', fontsize=13)
plt.text(100, 220000, r'b = $-4.85 $ kJ/mol', fontsize=13)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


plt.legend(fontsize=12, bbox_to_anchor=(.975,.3))
plt.savefig('troutons_rule_optimization.png', format='png', dpi=300)
plt.show()

#The slopes calculated using both linear regression and numerical optimization are 
#very similar, suggesting that the calculated entropy of vaporization is accurate. 
#Furthermore, this implies that the two methods should be compared in terms of efficiency, 
#rather than accuracy. The linear regression method requires significantly more lines of 
#code compared to numerical optimization, indicating that the optimization method is 
#better suited for this scenario. However, the calculation speeds cannot be 
#compared given the small size of the data set. Both methods effectively ran instantaneously.



