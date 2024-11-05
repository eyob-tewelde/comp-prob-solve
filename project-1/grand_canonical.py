#GCMC simulation


import numpy as np

#Define a function that creates an array of N x N filled with 0.
def initialize_lattice(size):
    lattice = np.zeros([size, size])
    return lattice

def compute_neighbor_indices(size):
    neighbor_indices = {}
    for x in range(0, size):
        for y in range(0, size):
            neighbors = [
                ((x - 1) % size, y),
                ((x + 1) % size, y),
                (x, (y - 1) % size),
                (x, (y + 1) % size)
            ]
            neighbor_indices[(x, y)] = neighbors
    return neighbor_indices

