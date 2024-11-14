# Grand canonical monte carlo simulations of competitive adsorption


import numpy as np
import matplotlib.pyplot as plt


#Define a function that initializes an empty lattice
def initialize_lattice(size):
    """
    Initialzies an empty lattice of dimensions size x size

    Parameters:
        size: int
            Dimension of the lattice

    Returns:
        ndarray:
            An empty array of dimensions size x size
    """

    lattice = np.zeros([size, size])
    return lattice


#Compute neighbor indices with periodic boundary conditions
def compute_neighbor_indices(size):
    """
    Computes the coordinates of the neighboring sites for all sites in an array with dimensions size x size

    Parameters:
        size: int
            Dimension of the lattice

    Returns:
        dict{(int, int): list[(int, int)]}:
            A dictionary where the key is the site of interest and the value is a list containing the coordinates of all neighboring indices
    """

    neighbor_indices = {}
    for x in range(size):
        for y in range(size):
            neighbors = [
                ((x - 1) % size, y),    #Compute neighboring indices
                ((x + 1) % size, y),
                (x, (y - 1) % size),
                (x, (y + 1) % size)
            ]
            neighbor_indices[(x, y)] = neighbors
    return neighbor_indices


#Calculate interaction energy
def calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB):
    """
    Calculates the interaction energy of a particle with neighboring particles

    Parameters:
        lattice: ndarray
            Represents the simulated lattice. Contains particle A, B or an empty site at all positions within the array
        site: ndarray
            Contains the coordinates of the particle of interest
        particle: int
            Particle of interest
        neighbor_indices: dict{(int, int): list[(int, int)]}
            Contains the coordinates of all neighboring sites
        epsilon_AA: float
            Interaction energy between particle A and A
        epsilon_BB: float
            Interaction energy between particle B and B
        epsilon_AB: float
            Interaction energy between particle A and B

    Returns:
        float:
            The total interaction energy felt by the particle of interest
    """

    x, y = site # Retrieve site coordinates
    interaction_energy = 0  # Initialize interaction energy
    for neighbor in neighbor_indices[(x, y)]:
        neighbor_particle = lattice[neighbor]
        if neighbor_particle != 0:
            if particle == 1:   # Particle A
                if neighbor_particle == 1:
                    interaction_energy += epsilon_AA
                else:   # Neighbor is Particle B
                    interaction_energy += epsilon_AB
            else:  # Particle B
                if neighbor_particle == 2:
                    interaction_energy += epsilon_BB
                else:  # Neighbor is Particle A
                    interaction_energy += epsilon_AB
    return interaction_energy


#Attempt to add or remove a particle
def attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params):
    """
    Accepts or rejects a move in the simulation. The possible moves are the adsorption or desorption of particle A or B.

    Parameters:
        lattice: ndarray
            Represents the simulated lattice. Contains particle A, B or an empty site at all positions within the array
        N_A: int
            Number of particle A's adsorbed to the lattice
        N_A: int
            Number of particle B's adsorbed to the lattice
        N_empty: int
            Number of empty sites
        neighbor_indices: dict{(int, int): list[(int, int)]}
            Contains the coordinates of all neighboring sites
        params: dict{str: float}
            Contains the values for the interaction energies, adsorption energies, and temperature

    Returns:
        N_A: int
            Number of particle A's adsorbed to the lattice
        N_B: int
            Number of particle B's adsorbed to the lattice
        N_empty: int
            Number of empty sites
    """

    size = np.shape(lattice)[0] #Extract lattice size
    N_sites = size * size   #Calculate number of sites
    beta  = 1 / params['T'] #Extract parameter values
    epsilon_A = params['epsilon_A']
    epsilon_B = params['epsilon_B']
    epsilon_AA = params['epsilon_AA']
    epsilon_BB = params['epsilon_BB']
    epsilon_AB = params['epsilon_AB']
    mu_A = params['mu_A']
    mu_B = params['mu_B']


    # Decide whether to add or remove a particle (50% chance each)
    if np.random.rand() < 0.5: # Adding a particle
        if N_empty == 0:
            return N_A, N_B, N_empty  # No empty sites available
        
        # Select a random empty site
        empty_site = np.argwhere(lattice == 0)
        rand_idx = np.random.randint(len(empty_site))
        random_empty_site = empty_site[rand_idx]

        if np.random.rand() < 0.5:  # Adding particle A
            particle = 1
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A
        else:  # Adding Particle B
            particle = 2
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B
    
        #C alculate change in energy for adding particle
        delta_E = epsilon + calculate_interaction_energy(lattice, random_empty_site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB)

        # Calculate acceptance probability for adding particle
        acc_prob = min(1, (N_empty) / (N_s + 1) * np.exp(-beta * (delta_E - mu)))

        r = np.random.rand()
        if r < acc_prob:    # Decide to accept the addition of the particle
            add_x, add_y = random_empty_site
            lattice[add_x, add_y] = particle
            if particle == 1:
                N_A += 1    # Increment site containing particle A count
            else:
                N_B += 1    # Increment site containing particle B count
            N_empty -= 1        # Decrement empty site count


    else:  # Removing a particle
        if N_sites - N_empty == 0:
            return N_A, N_B, N_empty  # No particles to remove
        

        # Select a random filled site
        filled_site = np.argwhere(lattice != 0)     
        rand_idx = np.random.randint(len(filled_site))
        random_filled_site = filled_site[rand_idx]
        rem_x, rem_y = random_filled_site
        particle = lattice[rem_x, rem_y]


        if particle == 1:   # Particle A
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A
        else:  # Particle B
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B

        # Calculate change in energy for removing particle
        delta_E = -epsilon - calculate_interaction_energy(lattice, random_filled_site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB)

        # Calculate acceptance probability for removing particle
        acc_prob = min(1, N_s / (N_empty + 1) * np.exp(-beta * (delta_E + mu)))

        r = np.random.rand()
        if r < acc_prob:    # Decide to accept the removal of a particle
            lattice[rem_x, rem_y] = 0  # Remove particle
            if particle == 1:
                N_A -= 1
            else:
                N_B -= 1
            N_empty += 1

    return N_A, N_B, N_empty


#Run the GCMC Simulation
def run_simulation(size, n_steps, params):
    """
    Runs a grand canonical monte carlo simulation of competitive adsorption

    Parameters:
        size: int
            Used to initalize a lattice of dimensions size x size
        n_steps: int
            Number of steps (moves) the simulation will traverse
        params: dict{str: float}
            Contains the values for the interaction energies, adsorption energies, and temperature
    
    Returns:
        lattice: ndarray
            Represents the simulated lattice. Contains particle A, B or an empty site at all positions within the array
        coverage_A: ndarray
            Number of adsorbed particle A / total sites for each step in the simulation
        coverage_B: ndarray
            Number of adsorbed particle B / total sites for each step in the simulation
    """

    lattice = initialize_lattice(size)  # Initialize lattice
    neighbor_indices = compute_neighbor_indices(size)   # Compute neighbor indices
    N_sites = size * size   # Compute number of empty sites
    N_A = 0     # Initialize number of adsorped A particles
    N_B = 0     # Initialize number of adsorped A particles
    N_empty = N_sites   # Initialize number of empty sites

    coverage_A = np.zeros(n_steps)  # Initalize  array
    coverage_B = np.zeros(n_steps)  # Initialze array

    # Compute a move in the simulation
    for step in range(n_steps):
        N_A, N_B, N_empty = attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params)
        coverage_A[step] = N_A / N_sites    # Update the coverage of particle A
        coverage_B[step] = N_B / N_sites    # Update the coverage of particle B

    return lattice, coverage_A, coverage_B


#Plot lattice configuration
def plot_lattice(lattice, ax, title):
    """
    Plots particles A and B as well as empty sites on a grid representing the lattice

    Parameters:
        lattice: ndarray
            Represents the simulated lattice. Contains particle A, B or an empty site at all positions within the array
        ax: Axes
            An empty subplot
        title: str
            Title of the subplot
    
    Returns:
        ax: Axes
            A subplot containing adsorped A and B particles and empty sites for a given chemical potential of particle A
    """

    size = np.shape(lattice)[0]     # Get dimensions of the lattice
    
    for x in range(size):
        for y in range(size):
            if lattice[x, y] == 1:  # Plot particle A
                ax.plot(x + 0.5, y+ 0.5, 'o', color='red', markersize=10)
            elif lattice[x, y] == 2:    # Plot particle B
                ax.plot(x + 0.5, y+ 0.5, 'o', color='blue', markersize=10)
    
    # Set plot configuration
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(np.arange(0, size + 1, 1))  
    ax.set_yticks(np.arange(0, size + 1, 1))  
    ax.set_xticklabels(['']*(size+1))  
    ax.set_yticklabels(['']*(size+1))  
    ax.grid()
    ax.set_title(title, fontsize=8)

    return ax