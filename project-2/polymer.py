#MD simulation of a polymer chain at constant T and V
import numpy as np
np.random.seed(42)

k_B = 1  # Boltzman constant
e_attract = 0.5 # Depthh of attractive LJ potential
n_particles = 20  # Number of particles
sigma = 1.0  # LJ potential parameter
cutoff = 2 ** (1/6) * sigma # Cutoff distance
box_size = 100  # Size of the cubic box
r0 = 1.0  # Equilibrium bond length
mass = 1.0  # Particle mass
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
rescale_interval = 100  # Steps between velocity rescaling

def initialize_chain(n_particles, box_size, r0):
    """
    Initialzies a polymer and simulation box

    Parameters:
        n_particles: int
            Number of particles in the chain
        box_size: int or float
            Length of simulation box
        r0: int or float
            Distance between particles

    Returns:
        ndarray:
            An array containing the intialized positions of the particles
    """

    positions = np.zeros((n_particles, 3))  #Initialize empty array
    current_position = [box_size/2, box_size/2, box_size/2] #Store coordinates of the center of the box
    positions[0] = current_position #Set the intial position to the center of the box
    for i in range(n_particles):
        v = np.random.rand(3)   #Generate random unit vector
        direction = v / np.linalg.norm(v)   #Calculate the direction of the random unit vector
        next_position = current_position + (r0 * direction) #Calculate the position of new particle
        positions[i] = apply_pbc(next_position, box_size)   #Update the position of new particle
        current_position = positions[i] #Update current position
    return positions

def initialize_velocities(n_particles, target_temp, mass):
    """
    Initializes the particle velocities.

    Parameters:
        n_particles: int
            Number of particles in the chain
        target_temp: int or float
            Initial temperature of the simulation
        mass: int or float
            Mass of the particles

    Returns:
        ndarray:
            Contains the initial particle velocities
    """
    velocities = np.random.normal(0, np.sqrt((k_B * target_temp) / mass), (n_particles, 3)) # Generate random velocities from Maxwell-Boltzmann distribution
    velocities -= np.mean(velocities, axis=0)   # Remove net momentum
    return velocities

def apply_pbc(position, box_size):
    """
    Applies periodic boundary conditions to the simulation box

    Parameters:
        position: (float, float)
            Position of the particle
        box_size: int or float
            Length of simulation box
    
    Returns:
        ndarray:
            Updated coordinates of the particle     
    """
    return position % box_size

def compute_harmonic_forces(positions, k, r0, box_size):
    """
    Computes the force felt by bonded particles in the polymer.

    Parameters:
        positions: list of (float, float)
            Positions of the particles
        k: int or float
            Spring constant
        r0: int or float
            Distance between particles
        box_size: int or float
            Length of simulation box

    Returns:
        ndarray:
            Contains the force felt between all particles and their neighbors
    """

    forces = np.zeros_like(positions)   #Intialize empty array
    for i in range(n_particles - 1):
        displacement = positions[i+1] - positions[i]    #Calculate minimum image distance between two neighboring particles
        displacement -= box_size * np.round(displacement / box_size)
        distance = np.linalg.norm(displacement)
        force_magnitude = -k * (distance - r0) #Calculate force between two neighboring particles
        force = force_magnitude * (displacement / distance)
        forces[i] -= force  #update force array
        forces[i+1] += force
    return forces

def compute_lennard_jones_forces(positions, eps, sigma, box_size, interaction_type):
    """
    Computes a repulsive or attractive force between particles if they are two
    or greater than two spacers apart, respectively. 

    Parameters:
        positions: list of (float, float)
            Positions of the particles
        epsilon: int or float
            The depth of the Lennard-Jones potential
        sigma: int or float
            Lennard-Jones parameter
        box_size: int or float
            Length of simulation box
        interaction_type: str
            Labels the interaction as repulsive or attractive

    Returns:
        ndarray:
            Contains the repulsive or attractive forces felt between all non-bonded particles.
    """

    forces = np.zeros_like(positions) #Initialize empty array
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            if interaction_type == 'repulsive' and np.abs(i - j) == 2:  #check if interaction type is repulsive and particles are two spacers apart
                epsilon = eps
            elif interaction_type == 'attractive' and np.abs(i - j) > 2:    #check if interaction type is attractive and particles are more than two spacers apart
                epsilon = eps
            else:
                continue
            displacement = positions[j] - positions[i]  #calculate distance between two particles
            displacement -= box_size * np.round(displacement / box_size)
            distance = np.linalg.norm(displacement)
            if distance < cutoff: #check if the distance between two particles is below cutoff length
                force_magnitude = 24 * epsilon * ((sigma / distance) ** 12 - 0.5 * (sigma / distance) ** 6) / distance
                force = force_magnitude * (displacement / distance) #Calculate force between two particles
                forces[i] -= force #Update force array
                forces[j] += force
    return forces

def velocity_verlet(positions, velocities, forces, dt, mass, k, e_repulsive):
    """
    Updates the positions of particles in a polymer after an applied force

    Parameters:
        positions: list of (float, float)
            Positions of the particles
        velocities: list of (float, float)
            Velocities of the particles
        forces: list of (int or float)
            Forces between the particles
        dt: int or float
            Time step size
        mass: int or float
            Mass of each particle

    Returns:
        ndarray:
            Contains the new position of each particle after a time step
    """
    velocities += 0.5 * forces / mass * dt  #Update the velocity of each particle
    positions += velocities * dt    #Update the positions of each particle
    positions = apply_pbc(positions, box_size)  #Apply periodic boundary conditions to new positions
    new_forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)   #Compute new harmonic forces
    new_forces_repulsive = compute_lennard_jones_forces(positions, e_repulsive, sigma, box_size, 'repulsive')    #Compute new repulsive forces
    new_forces_attractive = compute_lennard_jones_forces(positions, e_attract, sigma, box_size, 'attractive') #Compute new attractive forces
    forces_new = new_forces_harmonic + new_forces_repulsive + new_forces_attractive #Sum the new forces
    velocities += 0.5 * forces_new / mass * dt #Update the velocity of each particle
    return positions, velocities, forces_new

def rescale_velocities(velocities, target_temperature, mass):
    """
    Adjusts the velocities of the particles so that they fit the Maxwell-Boltzman
    distribution of a given temperature

    Parameters:
        velocities: list of (float, float)
            Velocities of the particles
        target_temperature: int or float
            Desired constant temperature of the simulation
        mass: int or float
            Mass of each particle

    Returns:
        ndarray:
            Contains the adjusted velocities of each particle
    """
    kinetic_energy = 0.5 * mass * np.sum(np.linalg.norm(velocities, axis=1) ** 2) #Calculate the overall kinetic energy
    current_temperature = (2/3) * kinetic_energy / (n_particles * k_B)  #Calculate the current temperature
    scaling_factor = np.sqrt(target_temperature / current_temperature) #Calculate scaling factor
    velocities *= scaling_factor    #Adjust current velocities by scaling factor

    return velocities

def calc_gyration_radius(positions):
    """
    Computes the radius of gyration for a polymer

    Parameters:
        positions: list of (float, float)
            Positions of the particles

    Returns:
        Rg: int or float
            Radius of gyration
    """

    com = np.mean(positions, axis=0)    #Calculate the center of mass of the polymer

    Rg_squared = np.mean(np.sum((positions - com) ** 2, axis=1))    #Calculate radius of gyration squared
    
    Rg = np.sqrt(Rg_squared)    #Calculate the radius of gyration
    return Rg

def calc_end_to_end_dist(positions):
    """
    Computes the end-to-end distance for a polymer

    Parameters:
        positions: list of (float, float)
            Positions of the particles

    Returns:
        Ree: int or float
            End-to-end distance
    """

    Ree = np.linalg.norm(positions[-1] - positions[0])  #Calculate the end to end distance
    # Ree -= box_size * np.round(Ree / box_size)
    return Ree

def calc_harmonic_potential_energy(positions, k):
    """
    Computes the overall harmonic bond potential energy of the polymer.

    Parameters:
        positions: list of (float, float)
            Positions of the particles
        k: int or float
            Spring constant

    Returns:
        int or float:
            Overall harmonic bond potential energy
    """
    potential_energy = 0    #Initialize potential energy
    for i in range(n_particles - 1):
        displacement = positions[i+1] - positions[i]
        displacement -= box_size * np.round(displacement / box_size)
        distance = np.linalg.norm(displacement) #Calculate distance between two particles
        potential_energy += k * ((distance - r0) ** 2) / 2   #Calculate harmonic bond potential between two particles and add to overall harmonic bond potential
    return potential_energy

def calc_LJ_potential_energy(positions, eps, interaction_type):
    """
    Computes the overall Lennard-Jones potential energy of the polymer.

    Parameters:
        positions: list of (float, float)
            Positions of the particles
        interaction_type: str
            Labels the interaction as repulsive or attractive

    Returns:
        int or float:
            Overall Lennard-Jones potential energy
    
    """
    potential_energy = 0    #Initialize potential energy
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            displacement = positions[j] - positions[i]
            displacement -= box_size * np.round(displacement / box_size)
            distance = np.linalg.norm(displacement) #Calculate distance between two particles

            if interaction_type == 'repulsive' and np.abs(i - j) == 2:  #Check if the particles are repulsively interacting and are two spacers apart
                epsilon = eps
                if distance < cutoff:
                    potential_energy += 4 * epsilon * ((sigma / distance)**12 - (sigma / distance)**6 + 0.25)   #Calculate the repulsive potential energy and update overall energy
            elif interaction_type == 'attractive' and np.abs(i - j) > 2:    #Check if the particles are attractively interacting and are more than two spacers apart
                epsilon = eps 
                potential_energy += 4 * epsilon * ((sigma / distance)**12 - (sigma / distance)**6) #Calculate the attractice potential energy and update overall energy
            else:
                continue
    return potential_energy
