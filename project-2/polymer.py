#MD simulation of a polymer chain at constant T and V
import numpy as np
np.random.seed(42)

from scipy import constants

k_B = constants.k
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
k = 1.0  # Spring constant
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
target_temperature = 0.1  # Target temperature
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_repulsive = 1.0  # Depth of repulsive LJ potential
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter
cutoff = (2 ** (1 /6)) * sigma

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

    positions = np.zeros((n_particles, 3))
    current_position = [box_size/2, box_size/2, box_size/2]
    positions[0] = current_position
    for i in range(n_particles):
        v = np.random.rand(2)
        direction = v / np.linalg.norm(v)
        next_position = current_position + r0 * direction
        positions[1] = apply_pbc(next_position, box_size)
        current_position = positions[i]
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

    velocities = np.random.normal(0, np.sqrt((k_B * target_temp)) / mass, (n_particles, 2))
    velocities -= np.mean(velocities)
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

    forces = np.zeros_like(positions)
    for i in range(n_particles - 1):
        displacement = positions[i+1] - positions[i]
        displacement -= box_size * np.round (displacement / box_size)
        distance = np.linalg.norm(displacement)
        force_magnitude = -k * (distance - r0)
        force = force_magnitude * (displacement / distance)
        forces[i] -= force
        forces[i+1] += force
    return forces

def compute_lennard_jones_forces(positions, epsilon, sigma, box_size, interaction_type):
    """
    Computes a repulsive or attractive force between all particles if they are two
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

    forces = np.zeros_like(positions)
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            if interaction_type == 'repulsive' and np.abs(i - j) == 2:
                epsilon = epsilon
            elif interaction_type == 'attractive' and np.abs(i - j) > 2:
                epsilon = epsilon
            else:
                continue
            displacement = positions[j] - positions[i]
            displacement -= box_size * (displacement / box_size)
            distance = np.linalg.norm(displacement)
            if distance < cutoff:
                force_magnitude = 24 * epsilon * [ (sigma / distance)^{12} - 0.5 * (sigma / distance)^6 ] / distance
                force = force_magnitude * (displacement / distance)
                forces[i] -= force
                forces[j] += force
    return forces

def velocity_verlet(positions, velocities, forces, dt, mass):
    """
    Computes the force felt between neighboring particles in the polymer

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
    velocities += 0.5 * forces / mass * dt
    positions += velocities * dt
    positions = apply_pbc(positions, box_size)
    forces_new = compute_lennard_jones_forces(positions)
    velocities += 0.5 * forces_new / mass * dt
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
    kinetic_energy = 0.5 * mass * sum(np.norm(velocities, axis=1)^2)
    current_temperature = (2/3) * kinetic_energy / (n_particles * k_B)
    scaling_factor = np.sqrt(target_temperature / current_temperature)
    velocities *= scaling_factor
    return velocities