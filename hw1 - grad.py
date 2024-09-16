#hw1 - grad

#define constants

import numpy as np
h_bar, m, L = 1, 1, 1


grid = np.linspace(-L / 2, L / 2, 2000, endpoint=True)
delt_x = grid[1] - grid[0]

#construct the laplacian matrix

iden_matrix = np.identity(2000)
off_diagonal = np.eye(2000, 2000, 1) + np.eye(2000, 2000, -1)

lap_matrix = (1 / (delt_x ** 2)) * (-2 * iden_matrix + off_diagonal)

#construct the hamiltonian matrix

ham_matrix = (-1 / 2) * lap_matrix

#solve for eigenvalues and eigenfunctions

eigenvalue, eigenvector = np.linalg.eig(ham_matrix)
sorted_eigenval = np.sort(eigenvalue)

first_seven_eigvals = (sorted_eigenval[:7])


first_seven_eigvecs = []
for val in first_seven_eigvals:
    index = np.where(eigenvalue == val)
    first_seven_eigvecs.append(eigenvector[:,index])

squeezed_eigvecs = np.squeeze(first_seven_eigvecs[:5])

import matplotlib.pyplot as plt

for i in range(squeezed_eigvecs.shape[0]):
    plt.plot(grid, squeezed_eigvecs[i], label=str(i))

plt.title("Wavefunctions in a 1-D potential well")
plt.xlabel("L (a0)")
plt.ylabel("Wavefunction")
plt.xlim(-L / 2, L / 2)
plt.legend(fontsize='small', title="Energy Levels", title_fontsize='medium', bbox_to_anchor=(1.05, 1))
plt.grid()
plt.show()