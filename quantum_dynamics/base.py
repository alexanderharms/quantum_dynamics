import numpy as np
from sympy.functions.special.delta_functions import Heaviside

from .solvers import CrankNicholson1D

class BaseEnvironment():
    def __init__(self, num_dims, size, num_nodes):
        self.num_dims = num_dims
        self.size = size 
        #assert self.num_dims == len(self.size), \
        #        "Defined environment size does not match dimensionality"

        self.num_nodes = num_nodes
        self.node_spacing = (size[1] - size[0]) / (num_nodes-1)  
        self.space_vec = np.linspace(size[0], size[1], num_nodes)

class SquareWell1D(BaseEnvironment):
    def __init__(self, size, num_nodes):
        num_dims = 1
        super().__init__(num_dims, size, num_nodes)
        self.potential = np.zeros(self.num_nodes)

    def set_potential(self, size_pot):
        pot_val = 1e6
        pot = self.potential
        for node in range(self.num_nodes):
            pot[node] =  -pot_val \
                    * (Heaviside(size_pot[0]/self.node_spacing - node)
                    - Heaviside(size_pot[1]/self.node_spacing - node))
        self.potential = pot

class BaseWave1D():
    def __init__(self, pos_init, mom_init, environment):
        self.psi = np.zeros(environment.num_nodes, dtype=np.cfloat)
        self.pos_init = pos_init
        self.mom_init = mom_init
        self.environment = environment

class PulseWave1D(CrankNicholson1D, BaseWave1D):
    def __init__(self, pos_init, mom_init, environment):
        super().__init__(pos_init, mom_init, environment)
        
    def generate_pulse(self, pulse_width):
        pos_init = self.pos_init
        mom_init = self.mom_init
        space_vec = self.environment.space_vec
        energy_init = mom_init**2 / 2

        psi = np.exp(-((space_vec - pos_init)**2 / (2*pulse_width**2))) \
                * np.exp(1j*space_vec*mom_init)
        psi = psi / np.linalg.norm(psi)
        self.psi += psi

