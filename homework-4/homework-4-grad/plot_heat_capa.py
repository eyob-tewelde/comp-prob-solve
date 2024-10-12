import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cv_ie_vs_temp.csv', index_col=False)
heat_capacity = df["Cv"].to_numpy()
temperature = df["Temperature"].to_numpy()


plt.plot(temperature, heat_capacity)
plt.xlabel('Temperature (K)')
plt.ylabel('Cv (J/K)')
plt.title('Heat Capacity (Cv) vs Temperature')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()