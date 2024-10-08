#Homework-3

#Part 1: Trouton's Rule

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import t


#Import trouton.csv
trouton = pd.read_csv("trouton.csv")

#Sort the data in ascending order boiling points
sorted_trouton = trouton.sort_values(by="T_B (K)")

#Extract the boiling points
sorted_boil = sorted_trouton["T_B (K)"]

#Extract the enthalpies of vaporization and convert to J/mol
sorted_enth_vapor = (sorted_trouton["H_v (kcal/mol)"].values) * 4184


#Use ordinary least squares to get entropy and intercept parameters
def ols_slope(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator

def ols_intercept(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = ols_slope(x, y)
    return y_mean - slope * x_mean

def ols(x, y):
    slope = ols_slope(x, y)
    intercept = ols_intercept(x, y)
    return slope, intercept

entropy, intercept = ols(sorted_boil, sorted_enth_vapor)
#Entropy of vaporization = 103.85486 J/mol * K
#Enthalpy of vaporization = -4844.6 J/mol * K when the boiling point is 0K (this is the intercept).



line = entropy * sorted_boil + intercept
residuals = sorted_enth_vapor - line


#calculate sum of squared errors
def sse(residuals):
    return np.sum(residuals ** 2)

#Use sum of squared errors to calculate the variance
def variance(residuals):
    return sse(residuals) / (len(residuals) - 2)

#calculate the standard error of the slope
def se_slope(x, residuals):
    numerator = variance(residuals)
    x_mean = np.mean(x)
    denominator = np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)

#calculate the standard error of the intercept
def se_intercept(x, residuals):
    numerator = variance(residuals)
    x_mean = np.mean(x)
    denominator = len(x) * np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)

#calculate the confidence interval of the slope
def confidence_interval_slope(x, residuals, confidence_level):
    se = se_slope(x, residuals)
    n_data_points = len(x)
    df = n_data_points - 2
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, df)
    return critical_t_value * se


#calculate the confidence interval of the intercept

def confidence_interval_intercept(x, residuals, confidence_level):
    # Calculate the standard error of the intercept
    se = se_intercept(x, residuals)

    # Calculate the critical t-value
    n_data_points = len(x)
    df = n_data_points - 2  # degrees of freedom
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, df)

    # Calculate the confidence interval
    return critical_t_value * se


#Calculate the confidence interval of the slope and intercept with 95% confidence
ci_slope = confidence_interval_slope(sorted_boil, residuals, 0.95)
ci_intercept = confidence_interval_intercept(sorted_boil, residuals, 0.95)
#print(ci_slope)
#print(ci_intercept)

#Create dictionary to map colors to substance class
colors = {
    "Perfect liquids":"r",
    "Liquids subject to quantum effects":"b",
    "Imperfect liquids":"g",
    "Metals":'tab:purple'
}

#Plot the data points and the fitted line
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
plt.title("Trouton's Rule", fontsize=14)


plt.text(100, 280000, r'$H_v = a \cdot T_b + b$', fontsize=13)
plt.text(100, 250000, r'a = $103.85 \pm 6.40$ J/mol $\cdot K$', fontsize=13)
plt.text(100, 220000, r'b = $-4.84 \pm 1.31$ kJ/mol', fontsize=13)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


plt.legend(fontsize=12, bbox_to_anchor=(.975,.3))
plt.show()

#The entropy of vaporization was calculated to be 103.85 +/- 6.39 J/mol*K. 
#This is 15.85 J/mol*K larger than the entropy of vaporization predicted by Trouton's rule. 
#As shown in the plot, this difference is largely a result of compounds with extremely high boiling points. 
#These compounds deviate from the relationship the data was plotted to, 
#whereas compounds with low boiling points correlate strongly to the fitted equation. 
#Therefore, it can be concluded the Trouton's Rule holds true for lower boiling points.

plt.savefig('troutons_rule.png', format='png', dpi=300)