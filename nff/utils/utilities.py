import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice
import pickle

def reorganize(cation_len, anion_len, mol_N, filename, file_type):
    """
    Take an xyz file, save a pickle file with sorted IL molecules
    Problem: Lammps xyz output file, has all the cations in a row
        and then all the anions. For CG mapping they need to follow one another
        (could also split the xyz file in half for cation and anion, then
        map to the CG coordinates, and then conc back together)
    args:
        cation_len: number of atoms in cation
        anion_len: number of atoms in anion
        mol_N: amount of molecules in the simulation
    return:
        sorted xyz coordinates
    """
    amount_of_timesteps = 0
    xyz_numbers = []
    mol_len = cation_len + anion_len
    atom_amount = mol_len * mol_N
    atom_amount_str = str(atom_amount)
    read_file = open(filename, 'r')
    if file_type == 'xyz':
        read_lines = read_file.readlines()
        for lines in read_lines:
            if (lines.startswith(atom_amount_str) or lines.startswith('Atoms.')):
                amount_of_timesteps += 1
                continue
            else:
                xyz_numbers.append(lines)
        amount_of_timesteps = int(amount_of_timesteps/2)
    elif file_type == 'force':
        read_lines = iter(read_file.readlines())
        for lines in read_lines:
            if (lines.startswith('ITEM: TIMESTEP')):
                amount_of_timesteps += 1
                lines = next(islice(read_lines, 7, 8), '')
                continue
            else:
                xyz_numbers.append(lines)
    else:
        print("NameError file_type must be either 'xyz' or 'force'")
        raise

    df_xyz = pd.DataFrame([sub.split(" ") for sub in xyz_numbers])
    if file_type == 'xyz':
        df_xyz[3].replace(regex=True,inplace=True,to_replace=r'\n',value=r'')
        df_xyz.drop(0, axis=1, inplace=True)
    else:
        df_xyz.drop([0,4], axis=1, inplace=True)

    xyz_np = df_xyz.to_numpy()
    xyz_np = xyz_np.astype(np.float)
    xyz_np = xyz_np.reshape((amount_of_timesteps,atom_amount,3))
    sorted_xyz = np.empty((amount_of_timesteps, atom_amount,3))

    for timestep in range(amount_of_timesteps):
        for molecule in range(mol_N):
            sorted_xyz[timestep, molecule * mol_len:molecule *
                       mol_len + cation_len] = xyz_np[timestep, molecule *
                                                      cation_len:molecule *
                                                      cation_len + cation_len]
            sorted_xyz[timestep, molecule * mol_len + cation_len:molecule *
                    mol_len + mol_len] = xyz_np[timestep, mol_N * cation_len + molecule *
                                                   anion_len:mol_N * cation_len +
                                                   molecule * anion_len + anion_len]

    sorted_xyz = sorted_xyz.reshape(amount_of_timesteps, mol_N, mol_len,3)
    print('Done sorting')
    return sorted_xyz
