import os 
import numpy as np
import torch
from torch.autograd import Variable

from ase import units
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.io import Trajectory

import nff.utils.constants as const
from nff.md.utils import NeuralMDLogger, write_traj
from nff.io.ase import NeuralFF

DEFAULTNVEPARAMS = {
    'T_init': 120.0, 
#     'thermostat': NoseHoover,   # or Langevin or NPT or NVT or Thermodynamic Integration
#     'thermostat_params': {'timestep': 0.5 * units.fs, "temperature": 120.0 * units.kB,  "ttime": 20.0}
    'thermostat': VelocityVerlet,  
    'thermostat_params': {'timestep': 0.5 * units.fs},
    'nbr_list_update_freq': 20,
    'steps': 3000,
    'save_frequency': 10,
    'thermo_filename': './thermo.log', 
    'traj_filename': './atom.traj',
    'skip': 0
}


class Dynamics:
    
    def __init__(self, 
                atomsbatch,
                mdparam=DEFAULTNVEPARAMS,
                ):
    
        # initialize the atoms batch system 
        self.atomsbatch = atomsbatch
        self.mdparam = mdparam
   
        # todo: structure optimization before starting
        
        # intialize system momentum 
        MaxwellBoltzmannDistribution(self.atomsbatch, self.mdparam['T_init'] * units.kB)
        Stationary(self.atomsbatch)  # zero linear momentum
        ZeroRotation(self.atomsbatch)
        
        # set thermostats 
        integrator = self.mdparam['thermostat']
        
        self.integrator = integrator(self.atomsbatch, **self.mdparam['thermostat_params'])
        
        # attach trajectory dump 
        self.traj = Trajectory(self.mdparam['traj_filename'], 'w', self.atomsbatch)
        self.integrator.attach(self.traj.write, interval=self.mdparam['save_frequency'])
        
        # attach log file
        self.integrator.attach(NeuralMDLogger(self.integrator, 
                                        self.atomsbatch, 
                                        self.mdparam['thermo_filename'], 
                                        mode='a'), interval=self.mdparam['save_frequency'])
        
    def setup_restart(self, restart_param):
        """If you want to restart a simulations with predfined mdparams but longer
         youneed to prodive a dcionary like the following:

         note that the thermo_filename and traj_name should be different 

         restart_param = {'atoms_path': md_log_dir + '/atom.traj', 
                          'thermo_filename':  md_log_dir + '/thermo_restart.log',
                          'traj_filename': md_log_dir + '/atom_restart.traj',
                          'steps': 100
                          }
        
        Args:
            restart_param (dict): dictionary to contains restart paramsters and file paths
        """

        if restart_param['thermo_filename'] == self.mdparam['thermo_filename']:
            raise ValueError("{} is also used, \
                please change a differnt thermo file name".format(restart_param['thermo_filename']))

        if restart_param['traj_filename'] == self.mdparam['traj_filename']:
            raise ValueError("{} is also used, \
                please change a differnt traj file name".format(restart_param['traj_filename']))

        self.restart_param = restart_param
        new_atoms = Trajectory(restart_param['atoms_path'])[-1]
        
        self.atomsbatch.set_positions(new_atoms.get_positions())
        self.atomsbatch.set_velocities(new_atoms.get_velocities())

        # set thermostats 
        integrator = self.mdparam['thermostat']
        self.integrator = integrator(self.atomsbatch, **self.mdparam['thermostat_params'])

        # attach trajectory dump 
        self.traj = Trajectory(self.restart_param['traj_filename'], 'w', self.atomsbatch)
        self.integrator.attach(self.traj.write, interval=self.mdparam['save_frequency'])

        # attach log file
        self.integrator.attach(NeuralMDLogger(self.integrator, 
                                        self.atomsbatch, 
                                        self.restart_param['thermo_filename'], 
                                        mode='a'), interval=self.mdparam['save_frequency'])
        
        self.mdparam['steps'] = restart_param['steps']
        
    def run(self):
         
        epochs = int(self.mdparam['steps'] // self.mdparam['nbr_list_update_freq'])
        
        for step in range(epochs):
            self.integrator.run(self.mdparam['nbr_list_update_freq'])
            self.atomsbatch.update_nbr_list()
        self.traj.close()
        
    
    def save_as_xyz(self, filename='./traj.xyz'):
        
        traj = Trajectory(self.mdparam['traj_filename'], mode='r')
        
        xyz = []
        
        skip = self.mdparam['skip']
        traj = list(traj)[skip:] if len(traj) > skip else traj

        for snapshot in traj:
            frames = np.concatenate([
                snapshot.get_atomic_numbers().reshape(-1, 1),
                snapshot.get_positions().reshape(-1, 3)
            ], axis=1)
            
            xyz.append(frames)
            
        write_traj(filename, np.array(xyz))