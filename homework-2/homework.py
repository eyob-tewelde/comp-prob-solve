#!/usr/bin/env python
# coding: utf-8

# Hw 1-2<br>
# Part 1: Importing and Exporting Data

# Dictonaries containing the cartesian coordinates of water, diatomic hydrogen, and benzene

# In[16]:


water = {
    "O1": [0, 0, 0.1173],
    "H2": [0, 0.7572, -0.4692],
    "H3": [0, -0.7572, -0.4692]
}


# In[17]:


hydrogen = {
    "H1": [0, 0, 0],
    "H2": [0, 0, 0.7414]
}


# In[18]:


benzene = {
    "C1": [0, 1.397, 0],
    "C2": [1.2098, 0.6985, 0],
    "C3": [1.2098, -0.6985, 0],
    "C4": [0, -1.397, 0],
    "C5": [-1.2098, -0.6985 ,0],
    "C6": [-1.2098, 0.6985, 0],
    "H7": [0, 2.481, 0],
    "H8": [2.1486, 1.2405, 0],
    "H9": [2.1486, -1.2405, 0],
    "H10": [0, -2.481, 0],
    "H11": [-2.1486, -1.2405, 0],
    "H12": [-2.1486, 1.2405, 0],
}


# In[19]:


#print(hydrogen)
#print(benzene)
#print(water)


# Part 2: Bond Length Calculation

# In[20]:


import numpy as np


# In[21]:


def calc_bond_length(atom1, atom2):
    """
    Calculates the distance between two atoms. Provides a warning if the
    bond length is greater than 2 Angstroms
    Parameters:
        atom1: list of float or int
            a list of Cartesian Coordinates for a given atom
        atom2: list of float or int
            a list of Cartesian Coordinates for a given atom
    Returns:
        int: the distance between the two atoms in Angstroms
        str: a warning message if the calculated distance is greater than 2 Angstroms
    """
    atom1_array = np.array(atom1)
    atom2_array = np.array(atom2)
    diff = atom1_array - atom2_array
    diff_squared = diff ** 2
    sum = diff_squared.sum()
    d = np.sqrt(sum)
    #if d >= 2:
        #print("Uh oh! These atoms might not be in a covalent bond.")
    #else:
        #print('the bond length is: ' + str(float(d)))
    
    return float(d)


# Part 3: Bond Angle Calculation

# In[22]:


def calc_bond_angle(atom1, atom2, atom3):
    """
    Calculates the bond angle between three atoms.
    Parameters:
        atom1: list of float or int
            a list of Cartesian Coordinates for a given atom
        atom2: list of float or int
            a list of Cartesian Coordinates for a given atom
        atom3: list of float or int
            a list of Cartesian Coordinates for a given atom
    Returns:
        int: the angle between the atoms
        str: acute, obtuse, or right depending on the bond angle
    """
    atm1_array = np.array(atom1)
    atm2_array = np.array(atom2)
    atm3_array = np.array(atom3)
    atm1_atm2 = atm1_array - atm2_array
    atm2_atm3 = atm2_array - atm3_array
    mag12 = np.linalg.norm(atm1_atm2)
    mag23 = np.linalg.norm(atm2_atm3)
    angle_rad = np.arccos(np.dot(atm1_atm2, atm2_atm3) / (mag12 * mag23))
    angle = np.degrees(angle_rad)
    
    # if angle > 90:
    #     print(str(angle) + ": obtuse")
    # elif angle == 90:
    #     print(str(angle) + ": right")
    # else:
    #     print(str(angle) + ": acute")
    return float(angle)


# Part 4: Automating the Calculation of Unique Bond Lengths and Angles

# In[23]:


def calc_all_bond_lengths(molecule):
    """
    Calculates the bond length of every unique pair of atoms in a molecule
    Parameters:
        molecule: dict of {str: list of float}
            A dictionary that contains the atom names and coordinates for every atom
            in a molecule
    Returns
        bond_lengths: list of tuple(str, float)
            A list that contains every unique pair of atoms and their bond lengths
    
    """
    dup = []
    bond_lengths = []
    for atom1, coord1 in molecule.items():
        for atom2, coord2 in molecule.items():
            if atom1 != atom2:
                pair = sorted((atom1, atom2))
                if pair not in dup:
                    dup.append(pair)
                    bond_lengths.append((atom1 + " + " + atom2, calc_bond_length(coord1, coord2)))
    
    return bond_lengths


# In[24]:


def calc_all_bond_angles(molecule):
    """
    
    """
    dup = []
    bond_angles = []
    for atom1, coord1 in molecule.items():
        for atom2, coord2 in molecule.items():
            for atom3, coord3 in molecule.items():
                if atom1 != atom2 and atom1 != atom3 and atom2 != atom3:
                    group = sorted((atom1, atom2, atom3))
                    if group not in dup:
                        dup.append(group)
                        bond_angles.append((atom1 + " + " + atom2 + " + " + atom3, calc_bond_angle(coord1, coord2, coord3)))
        
    return bond_angles


# In[25]:


#water_bond_angles, water_bond_lengths = calc_all_bond_angles(water), calc_all_bond_lengths(water)
#hydrogen_bond_angles, hydrogen_bond_lengths = calc_all_bond_angles(hydrogen), calc_all_bond_lengths(hydrogen)
#benzene_bond_angles, benzene_bond_lengths = calc_all_bond_angles(benzene), calc_all_bond_lengths(benzene)


# In[26]:


#print(water_bond_angles, water_bond_lengths)
#print(hydrogen_bond_angles, hydrogen_bond_lengths)
#print(benzene_bond_angles, benzene_bond_lengths)

