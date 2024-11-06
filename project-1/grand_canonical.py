#GCMC simulation


import numpy as np
import matplotlib.pyplot as plt


#Define a function that initializes an empty lattice
def initialize_lattice(size):
    """
    
    
    """
    lattice = np.zeros([size, size])
    return lattice


#Compute neighbor indices with periodic boundary conditions
def compute_neighbor_indices(size):
    """
    
    
    """
    neighbor_indices = {}
    for x in range(size):
        for y in range(size):
            neighbors = [
                ((x - 1) % size, y),
                ((x + 1) % size, y),
                (x, (y - 1) % size),
                (x, (y + 1) % size)
            ]
            neighbor_indices[(x, y)] = neighbors
    return neighbor_indices


#Calculate interaction energy
def calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB):
    """
    
    
    """
    x, y = site
    interaction_energy = 0
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
def attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params, count):
    """
    
    
    
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


    # DECIDE whether to add or remove a particle (50% chance each)
    if np.random.rand() < 0.5: #Adding a particle
        if N_empty == 0:
            return N_A, N_B, N_empty  # No empty sites available
        
        #Select a random empty site
        empty_site = np.argwhere(lattice == 0)
        rand_idx = np.random.randint(len(empty_site))
        random_empty_site = empty_site[rand_idx]

        if np.random.rand() < 0.5:  #Adding particle A
            particle = 1
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A
        else:  # Adding Particle B
            particle = 2
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B
    
        #Calculate change in energy for adding particle
        delta_E = epsilon + calculate_interaction_energy(lattice, random_empty_site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB)

        #Calculate acceptance probability for adding particle
        acc_prob = min(1, (N_empty) / (N_s + 1) * np.exp(-beta * (delta_E - mu)))

        r = np.random.rand()
        if r < acc_prob:    #Decide to accept the addition of the particle
            lattice[random_empty_site] = particle
            if particle == 1:
                N_A += 1    #Increment site containing particle A count
            else:
                N_B += 1    #Increment site containing particle B count
            N_empty -= 1        #Decrement empty site count


    else:  # Removing a particle
        if N_sites - N_empty == 0:
            return N_A, N_B, N_empty  # No particles to remove
        

        #Select a random filled site
        filled_site = np.argwhere(lattice != 0)     
        rand_idx = np.random.randint(len(filled_site))
        random_filled_site = filled_site[rand_idx]
        x, y = random_filled_site
        particle = lattice[x, y]


        if particle == 1:   # Particle A
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A
        else:  # Particle B
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B

        #Calculate change in energy for removing particle
        delta_E = -epsilon - calculate_interaction_energy(lattice, random_filled_site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB)

        #Calculate acceptance probability for removing particle
        acc_prob = min(1, N_s / (N_empty + 1) * np.exp(-beta * (delta_E + mu)))

        r = np.random.rand()
        if r < acc_prob:    #Decide to accept the removal of a particle
            lattice[random_filled_site] = 0  # Remove particle
            if particle == 1:
                N_A -= 1
            else:
                N_B -= 1
            N_empty += 1

    return N_A, N_B, N_empty


#Run the GCMC Simulation
def run_simulation(size, n_steps, params):
    """
    
    
    
    """
    lattice = initialize_lattice(size)  #Initialize lattice
    neighbor_indices = compute_neighbor_indices(size)   #Compute neighbor indices
    N_sites = size * size
    N_A = 0
    N_B = 0
    N_empty = N_sites
    coverage_A = np.zeros(n_steps)
    coverage_B = np.zeros(n_steps)
    for step in range(n_steps):
        N_A, N_B, N_empty = attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params, step)
        coverage_A[step] = N_A / N_sites
        coverage_B[step] = N_B / N_sites

    return lattice, coverage_A, coverage_B


#Plot lattice configuration
def plot_lattice(lattice, ax, title):
    """
    
    
    
    """
    size = np.shape(lattice)[0]
    for x in range(size):
        for y in range(size):
            if lattice[x, y] == 1:
                ax.plot(x + 0.5, y+ 0.5, 'o', color='red', markersize=20)
            elif lattice[x, y] == 2:
                ax.plot(x + 0.5, y+ 0.5, 'o', color='blue', markersize=20)

    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.tick_params(which='both')
    plt.xticks([])
    plt.yticks([])
    plt.grid(which='minor')
    plt.title(title)

    return ax