#Homework 2: Graduate Supplement

import numpy as np
import matplotlib.pyplot as plt

#Set constants
h_red = 1
m = 1
w = 1                        #a.u
D = 10                       #a.u
B = np.sqrt(1 / (2 * D))     #a.u
L = 40                       #a.u

x = np.linspace((-L / 2), (L / 2), 2000)    #x values used to generate oscillator potentials
delt_x = x[1] - x[0]

#Computes the potential of a harmonic oscillator
def harmonic_oscillator(x):
    """
    Calculates the potential of a harmonic oscillator

    Parameters:
        x: numpy.ndarray(float)
            array of floats from -L/2 to L/2 where L = 40 a.u.

    Returns:
        float: the potential of the harmonic oscillator at x.
    """

    return (1/2) * m * (w ** 2) * (x ** 2)


#Computes the potential of a anharmonic oscillator
def anharmonic_oscillator(x, x0=0):
    """
    Calculates the potential of a anharmonic oscillator

    Parameters:
        x: numpy.ndarray(float)
            array of floats from -L/2 to L/2 where L = 40 a.u.

    Returns:
        float: the potential of the anharmonic oscillator at x.
    """

    return D * ((1 - np.exp(-B * (x-x0))) ** 2)


#Generates a potential matrix of size x by x containing the oscillator potential on the diagonals
def osc_pot_matrix(oscillator, space):
    """
    Constructs a potential matrix such that the diagonals are the oscillator potentials.

    Parameters:
        oscillator: function
            Function that calculates the potential for given oscillator model.
        space: numpy.ndarray(float)
            array of floats from -L/2 to L/2 where L = 40 a.u.

    Returns:
        numpy.ndarray(float): A matrix of size len(space) x len(space) where the diagnoals are
                              the potentials for a given oscillator model.
    """

    potentials = oscillator(space)
    pot_matrix = np.diag(potentials)

    return pot_matrix


#Create a potential matrix for the harmonic and anharmonic oscillator models

harmonic_potential_matrix = osc_pot_matrix(harmonic_oscillator, x)
anharmonic_potential_matrix = osc_pot_matrix(anharmonic_oscillator, x)


#Generates a laplacian matrix of size x by x
def laplace(size):
    """
    Constructs a laplacian matrix

    Parameters:
        size (int): 
            The desired dimensions of the laplacian matrix

    Returns:
        numpy.ndarray(float): A laplacian matrix of size len(size) x len(size).
    
    """

    iden_matrix = np.identity(size)
    off_diagonal = np.eye(size, k=1) + np.eye(size, k=-1)
    lap_matrix = (1 / (delt_x ** 2)) * (-2 * iden_matrix + off_diagonal)

    return lap_matrix


#Create the laplacian matrix
laplacian = laplace(len(x))


#Create function that constructs a Hamiltonian matrix
def hamiltonian(lap_matrix, pot_matrix):
    """
    Constructs a Hamiltonian matrix

    Parameters:
        lap_matrix (numpy.ndarray(float)): 
            A laplacian matrix
        
        pot_matrix (numpy.ndarray(float)):
            A potential matrix for a given oscillator modal 

    Returns:
        numpy.ndarray(float): The Hamiltonian matrix for a given oscillator modal.
    """

    ham_matrix = (-1 / 2) * lap_matrix + pot_matrix
    return ham_matrix


#Generate the Hamiltonian matrix for the harmonic and anharmonic oscillator
harm_hamiltonian = hamiltonian(laplacian, harmonic_potential_matrix)
anharm_hamiltonian = hamiltonian(laplacian, anharmonic_potential_matrix)



#eigenvalues and eigenvectors for the harmonic oscillator
harm_eigenvalue, harm_eigenvector = np.linalg.eig(harm_hamiltonian)
sorted_harm_eigenval = np.sort(harm_eigenvalue)
first_ten_harm_eigvals = (sorted_harm_eigenval[:10])


#eigenvalues and eigenvectors for the anharmonic oscillator
anharm_eigenvalue, anharm_eigenvector = np.linalg.eig(anharm_hamiltonian)
sorted_anharm_eigenval = np.sort(anharm_eigenvalue)
first_ten_anharm_eigvals = (sorted_anharm_eigenval[:10])


#Extract the eigenvectors that correspond to the first 10 energy levels
first_ten_harm_eigvecs = []
first_ten_anharm_eigvecs = []

# Sort the indices based on the eigenvalues
sorted_harm_indices = np.argsort(harm_eigenvalue)[:10]
sorted_anharm_indices = np.argsort(anharm_eigenvalue)[:10]

# Extract the eigenvectors that correspond to the first 10 energy levels
first_ten_harm_eigvecs = harm_eigenvector[:, sorted_harm_indices]
first_ten_anharm_eigvecs = anharm_eigenvector[:, sorted_anharm_indices]


#Reduce unneccesary dimensions that arise during calculation
squeezed_harm_eigvecs = np.squeeze(first_ten_harm_eigvecs[:10])
squeezed_anharm_eigvecs = np.squeeze(first_ten_anharm_eigvecs[:10])



#Setup plot for harmonic and anharmonic wavefunctions
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# Plot the first 10 harmonic oscillator eigenvectors
for i in range(first_ten_harm_eigvecs.shape[1]):  # Iterate over eigenvector columns
    axs[0].plot(x, first_ten_harm_eigvecs[:, i], label=str(i), linewidth=2, alpha=0.8)

axs[0].set_title("Harmonic Oscillator Wavefunctions")
axs[0].set_xlabel("L (a.u.)")
axs[0].set_ylabel("Wavefunction")
axs[0].legend(fontsize='small', title="Energy Levels", title_fontsize='medium', loc="upper left")
axs[0].set_xlim(-13, 13)
axs[0].set_ylim(-0.12, .12)
axs[0].grid()

# Plot the first 10 anharmonic oscillator eigenvectors
for i in range(first_ten_anharm_eigvecs.shape[1]):  # Iterate over eigenvector columns
    axs[1].plot(x, first_ten_anharm_eigvecs[:, i], label=str(i), linewidth=2, alpha=0.8)

axs[1].set_title("Anharmonic Oscillator Wavefunctions")
axs[1].set_xlabel("L (a.u.)")
axs[1].set_ylabel("Wavefunction")
axs[1].legend(fontsize='small', title="Energy Levels", title_fontsize='medium', loc="upper left")
axs[1].set_xlim(-13, 13)
axs[1].set_ylim(-0.12, .12)
axs[1].grid()

plt.tight_layout()
plt.show()
