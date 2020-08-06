"""Tools to fix pbc on a one/two molecule system"""
import numpy as np
import torch
import mdtraj as md
import pandas as pd

def get_box_dimensions_pbc(filename, timesteps):
    """
    Calculates the box dimensions of a force.dat file
    averaging over the whole simulation length
    Best if the box dimensions do not vary a lot
    
    args:
        filename - name of force.dat file
        timesteps - amount of timesteps for simulation
        
    return - cell dimensions in a python list
    """
    box_dim = []
    read_file = open(filename, 'r')

    read_lines = iter(read_file.readlines())
    for lines in read_lines:
        if (lines.startswith('ITEM: BOX BOUNDS')):
            lines = next(read_lines)
            box_dim.append(lines)
            lines = next(read_lines)
            box_dim.append(lines)
            lines = next(read_lines)
            box_dim.append(lines)
            
    df_xyz = pd.DataFrame([sub.split(" ") for sub in box_dim])
    df_xyz[1].replace(regex=True,inplace=True,to_replace=r'\n',value=r'')
    xyz_np = df_xyz.to_numpy()
    xyz_np = xyz_np.astype(np.float)
    xyz_np = xyz_np.reshape(timesteps,3,2)
    meanmean = xyz_np.mean(axis=(0))
    cell = [meanmean[0,1]-meanmean[0,0], meanmean[0,1]-meanmean[0,0], meanmean[0,1]-meanmean[0,0]]
    return cell

def split_mol(n_mol, mol_1_len, mol_2_len, path):
    """
    Splits a two molecule system if the atoms are arranged
    such that all the atoms of mol_1 are before mol_2
    
    Args:
        n_mol: amount of molecules in system
        mol_1_len: amount of atoms in mol_1
        mol_2_len: amount of atoms in mol_2
        path: path to file
    """
    traj = md.formats.XYZTrajectoryFile(path, mode='r').read()
    traj_mol_1 = traj[:, 0:n_mol*mol_1_len, :]
    traj_mol_2 = traj[:, n_mol*mol_1_len:n_mol*(mol_1_len+mol_2_len), :]
    both_molecules = [traj_mol_1, traj_mol_2]
    timesteps = traj_mol_1.shape[0]
    return both_molecules, timesteps

def trajconv(n_mol, mol, box_len, mol_matrix=None, path=None):
    '''
    currently only works a only one molecule type
    
    Using split_mol can be used on two molecules seperately
    Args: 
    
    path:    path to xyz file 
    mol_matrix: case where the numpy XYZ array has been generated before
    mol:     atomic compositions of each molecules (list)
    box:     box vector 
    n_mol:   number of molecules 
    
    example: 
        path = "../../sim/topotools_ethane/ethane-nvt.xyz"
        mol = [1, 1, 2, 2, 2, 2, 2, 2]
        n_mol = 384
        box_len = [15.759675 + 16.201924, 16.908199 + 16.866200, 14.949001 + 14.949001]
        trajconv(n_mol, n_atom, box_len, path)
    example for two mol system:
        path = 'data/pairs_100_xyz.xyz'
        ils_mol = [[7, 6, 7, 6, 6, 6, 1, 6, 1, 1, 1, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 1, 1, 1],
                    [5, 9, 9, 9, 9]]
        n_mol = 100
        cation_len = 25
        anion_len = 5
        box_len = [31.74956656457811, 31.74956656457811, 31.74956656457811]
        ils_matrix, timestep_amount = split_mol(n_mol, cation_len, anion_len, path)
        both_conversion = [[],[]]
        for i, ils in enumerate(ils_matrix):
            empty[i] = trajconv(number_of_molecules, ils_mol[i], cell, mol_matrix=ils)
    '''
    n_atom = len(mol)
    box_len = torch.Tensor(box_len)

    # read data 
    if path == None:
        traj = mol_matrix
    else:
        traj = md.formats.XYZTrajectoryFile(path, mode='r').read()
    traj = torch.Tensor(traj)
    n_frame = traj.shape[0]
    
    # build atom id 
    top=[] 
    for i in range(0, n_mol):
        atom = []
        for j in range(0, n_atom ):       
            atom.append(j + i * n_atom)
        top.append(atom)
    top = torch.LongTensor(top)
    top_1st = top[:, 0].reshape(-1, 1).repeat(1, n_atom)

    # compute intra moelcular distance 
    dis = traj[:, top, :] - traj[:, top_1st, :]

    sub = (dis > 0.5 * box_len).type("torch.FloatTensor") * box_len
    add = (dis <= -0.5 * box_len).type("torch.FloatTensor") * box_len
#     import pdb; pdb.set_trace()
#     print(top)
#     print(top_1st)
    traj_unwrap = traj[:, top, :] + add - sub

    atom_type = torch.Tensor( np.array(mol * n_mol) )
    traj_unwrap = traj_unwrap.reshape(-1, n_mol * n_atom, 3)
    
    traj2write = torch.cat((atom_type.reshape(1, -1).repeat(n_frame, 1).reshape(-1, n_mol*n_atom, 1),traj_unwrap), 2).detach().numpy()
    
    return traj2write

def combine_mol(timestep_amount, converted_mol, num_mol, mol_len, mol_1_len, mol_2_len, write=True):
    """
    Takes two lists and interleaves them
    Args:
        timestep_amount: amount of timesteps for simulation
        converted_mol: list of size 2, that has both the molecule data
        num_mol: amount of molecules in system
        mol_len: length of a single molecule
        
    """
    completed_traj = np.zeros((timestep_amount, mol_len*num_mol, 4))
    for timestep in range(timestep_amount):
        for molecule in range(num_mol):
            completed_traj[timestep, molecule * mol_len:molecule * 
                       mol_len + mol_1_len] = converted_mol[0][timestep, molecule * 
                                                      mol_1_len:molecule * 
                                                      mol_1_len + mol_1_len]
            completed_traj[timestep, molecule * mol_len + mol_1_len:molecule * 
                        mol_len + mol_len] = converted_mol[1][timestep, molecule * 
                                                      mol_2_len: molecule * 
                                                      mol_2_len + mol_2_len]
    if write == True:
        return completed_traj
    elif write == False:
        completed_traj = np.delete(completed_traj, 0,2)
        completed_traj = completed_traj.reshape(-1, num_mol, mol_len, 3)
        return completed_traj

def write_traj(filename, frames):
    '''
        Write trajectory dataframes into .xyz format for VMD visualization
        to do: include multiple atom types 
        
        example:
            path = "../../sim/topotools_ethane/ethane-nvt_unwrap.xyz"
            traj2write = trajconv(n_mol, n_atom, box_len, path)
            write_traj(path, traj2write)
    '''    
    file = open(filename,'w')
    atom_no = frames.shape[1]
    for i, frame in enumerate(frames): 
        file.write( str(atom_no) + '\n')
        file.write('Atoms. Timestep: '+ str(i)+'\n')
        for atom in frame:
            if atom.shape[0] == 4:
                file.write(str(atom[0]) + " " + str(atom[1]) + " " + str(atom[2]) + " " + str(atom[3]) + "\n")
            elif atom.shape[0] == 3:
                file.write("1" + " " + str(atom[0]) + " " + str(atom[1]) + " " + str(atom[2]) + "\n")
            else:
                raise ValueError("wrong format")
    file.close()

def save_traj(traj, Z, name):
    """Summary
    
    Args:
        traj (np.array): traj
        Z (atomic number): Description
        name (filename): Description
    """
    traj = np.array(traj)
    Z = np.array(Z * traj.shape[0]).reshape(traj.shape[0], len(Z), 1)
    traj_write = np.dstack(( Z, traj))
    write_traj(filename=name, frames=traj_write)


