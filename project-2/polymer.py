#MD simulation of a polymer chain at constant T and V
import numpy as np
np.random.seed(42)

def intialize_chain(n_particles, box_size, r0):
    """
    
    """

    positions = np.zeros(n_particles, 3)
    current_position = [box_size/2, box_size/2, box_size/2]
    positions[0] = current_position
    for i in range(n_particles):
        direction = random unit vector
        next_position = current_position + r0 * direction
        positions[1] = apply_pbc(next_position, box_size)
        current_position = positions[i]
    return positions

def initialize_velocities(n_particles, target_temp, mass):
    """
    
    """

    velocities = random velocities
    velocities -= np.mean(velocities)
    return velocities

def apply_pbc(position, box_size):
    """
    
    """
    return position % box_size

def compute_harmonic_force(positions, k, r0, box_size):
    """
    
    """

    forces = zeros_like(positions)
    for i in range(n_particles - 1):
        displacement = positions[i+1] - positions[i]
        displacement = minimum_image(displacement, box_size)
        distance = norm(displacement)
        force_magnitude = -k * (distance - r0)
        force = force_magnitude * (displacement / distance)
        forces[i] -= force
        forces[i+1] += force
    return forces

def compute_lennard_jones_forces(positions, epsilon, sigma, box_size, interaction_type):
    """
    
    """

    forces = zeros_like(positions)
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            if interaction_type == 'repulsive' and |i - j| == 2:
                USE epsilon_repulsive
            elif interaction_type == 'attractive' and |i - j| > 2:
                USE epsilon_attractive
            else:
                continue
            displacement = positions[j] - positions[i]
            displacement = minimum_image(displacement, box_size)
            distance = norm(displacement)
            if distance < cutoff:
                force_magnitude = 24 * epsilon * [ (sigma / distance)^{12} - 0.5 * (sigma / distance)^6 ] / distance
                force = force_magnitude * (displacement / distance)
                forces[i] -= force
                forces[j] += force
    return forces

def velocity_verlet(positions, velocities, forces, dt, mass):
    velocities += 0.5 * forces / mass * dt
    positions += velocities * dt
    positions = apply_pbc(positions, box_size)
    forces_new = compute_forces(positions)
    velocities += 0.5 * forces_new / mass * dt
    return positions, velocities, forces_new

def rescale_velocities(velocities, target_temperature, mass):
    kinetic_energy = 0.5 * mass * sum(np.norm(velocities, axis=1)^2)
    current_temperature = (2/3) * kinetic_energy / (n_particles * k_B)
    scaling_factor = np.sqrt(target_temperature / current_temperature)
    velocities *= scaling_factor
    return velocities