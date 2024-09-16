#hw 1-2
#Part 1: Importing and Exporting Data

#dictonaries containing the cartesian coordinates of water, diatomic hydrogen, and benzene
water = {
    "O1": [0, 0, 0.1173],
    "H2": [0, 0.7572, -0.4692],
    "H3": [0, -0.7572, -0.4692]
}

hydrogen = {
    "H1": [0, 0, 0],
    "H2": [0, 0, 0.7414]
}

benzene = {
    "C1": [0, 1.397, 0] ,
    "C2": [1.20098, 0.6985, 0],
    "C3": [1.20098, -0.6985, 0],
    "C4": [0, -1.397, 0],
    "C5": [-1.20098, -0.6985 ,0],
    "C6": [-1.20098, 0.6985, 0],
    "H7": [0, 2.481, 0],
    "H8": [2.1486, 1.2405, 0],
    "H9": [2.1486, -1.2405, 0],
    "H10": [0, -2.481, 0],
    "H11": [-2.1486, -1.2405, 0],
    "H12": [-2.1486, 1.2405, 0],
}

#print(hydrogen)
#print(benzene)
#print(water)

#Part 2: Bond Length Calculation

import numpy as np

def calc_bond_length(atom1, atom2):
    """
    Calculates the distance between two atoms. Provides a warning if the
    bond length is greater than 2 Angstroms

    Parameters:
        atom1: [float/int, float/int, float/int]
            a list of Cartesian Coordinates for a given atom

        atom2: [float/int, float/int, float/int]
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

    if d >= 2:
        print("Uh oh! These atoms are not in a covalent bond.")
    
    return print('the bond length is: ' + str(float(d)))


#Part 3: Bond Angle Calculation

def calc_bond_angle(atom1, atom2, atom3):
    """
    Calculates the bond angle between three atoms.

    Parameters:
        atom1: [float, float, float]
            a list of Cartesian Coordinates for a given atom

        atom2: [float, float, float]
            a list of Cartesian Coordinates for a given atom

        atom2: [float, float, float]
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
    
    if angle > 90:
        print(str(angle) + ": obtuse")
    elif angle == 90:
        print(str(angle) + ": right")
    else:
        print(str(angle) + ": acute")

    return None

