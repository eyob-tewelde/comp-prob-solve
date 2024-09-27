#homework-2-1

#Part 2: Argon Trimer

from homework import calc_bond_length, calc_bond_angle
from optimize_argon_dimer import lennard_jones
from scipy.optimize import minimize

def v_total(argon_coord):
    """
    Calculates the total lennard-jones potential for the an argon trimer.

    Parameters:
        argon_coord: list[r12, x3, y3]
            A list that contains:
                r12: the distance between atom1 and atom 2
                x3: the x coordinate of atom3
                y3: the y coordinate of atom3
    Returns:
        the potential of an argon trimer.
    """
    atom1 = [0,0]
    atom2 = [argon_coord[0], 0]
    atom3 = [argon_coord[1], argon_coord[2]]

    r13, r23 = calc_bond_length(atom1, atom3), calc_bond_length(atom2, atom3)

    return lennard_jones(argon_coord[0]) + lennard_jones(r13) + lennard_jones(r23)

def minimize_v_total(guess):
    """
    Finds the r12, x3, and y3 that minimizes the total lennard-jones potential
    for an argon trimer.

    Parameters:
        argon_coord: list[r12, x3, y3]
            A list that contains:
                r12: the distance between atom1 and atom 2
                x3: the x coordinate of atom3
                y3: the y coordinate of atom3
    Returns:
        the potential of the three atom molecule
    """

    result = minimize(
        v_total,
        x0=guess,
        method="Nelder-Mead",
        tol=1e-6
    )

    return result


initial_guess = [2,2,2]
print(minimize_v_total(initial_guess)) #r12 = 3.816, x3 = 1.908, y3 = 3.305

ar_1 = [0,0]
ar_2 = [3.816, 0]
ar_3 = [1.908, 3.305]

print(f"""The optimal bond lengths of Ar\u2083 are:
r12: 3.816 Angstroms
r13: {calc_bond_length(ar_1, ar_3):.3f} Angstroms
r23: {calc_bond_length(ar_2, ar_3):.3f} Angstroms

""")

print(f"""The optimal bond angles of Ar\u2083 is:
{calc_bond_angle(ar_1, ar_2, ar_3):.3f} degrees

Given the optimal bond lengths and angle, Ar\u2083 has a bent structure.
""")


#Create XYZ file
data = [
    ("Ar", 0.000, 0.000),
    ("Ar", 3.816, 0.000),
    ("Ar", 1.908, 3.304)
]

xyz_file_path = "argon_trimer.xyz"

with open(xyz_file_path, "w") as file:
    file.write(str(3))
    file.write("\nArgon trimer\n")
    
    # Write the atomic data
    for atom in data:
        symbol, x, y = atom
        file.write(f"{symbol}   {x:.6f}   {y:.6f}\n")

