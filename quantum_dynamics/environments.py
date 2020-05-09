import numpy as np

from .base import BaseEnvironment1D, BaseEnvironment2D

class SquareWell1D(BaseEnvironment1D):
    def __init__(self, size, num_nodes):
        super().__init__(size, num_nodes)
        self.potential = np.zeros(self.num_nodes)

    def set_potential(self, size_pot, pot_val=1e6):
       self.potential = pot_val \
               * (np.heaviside(size_pot[0] - self.space_vec, 1)
               + np.heaviside(self.space_vec - size_pot[1], 1))

class SquareWell2D(BaseEnvironment2D):
    def __init__(self, size, num_nodes):
        super().__init__(size, num_nodes)
        self.potential = np.zeros((self.num_nodes[0], self.num_nodes[1]))

    def set_potential(self, size_pot, pot_val=1e6):
        size_pot = np.array(size_pot)
        self.potential = pot_val \
                * np.logical_or(
                        np.heaviside(size_pot[0, 0] - self.space_vec[0], 1) 
                        + np.heaviside(self.space_vec[0] - size_pot[0, 1], 1),
                        np.heaviside(size_pot[1, 0] - self.space_vec[1] , 1) 
                        + np.heaviside(self.space_vec[1] - size_pot[1, 1], 1))

class Barrier1D(BaseEnvironment1D):
    def __init__(self, size, num_nodes):
        super().__init__(size, num_nodes)
        self.potential = np.zeros(self.num_nodes)

    def set_potential(self, pot_val, barrier_loc=None,
            barrier_size=None):
        if barrier_loc is None:
            barrier_loc = 0.5 * (self.size[1] - self.size[0])
        if barrier_size is None:
            barrier_size = 7.0/np.sqrt(2*pot_val)
        self.potential = pot_val \
                * (np.heaviside(barrier_loc - self.space_vec, 1)
                   + np.heaviside(barrier_loc + barrier_size 
                                  - self.space_vec, 1))

class Barrier2D(BaseEnvironment2D):
    def __init__(self, size, num_nodes):
        super().__init__(size, num_nodes)
        self.potential = np.zeros((self.num_nodes[0], self.num_nodes[1]))

    def set_potential(self, pot_val, barrier_loc=None,
            barrier_size=None):
        if barrier_loc is None:
            barrier_loc = [0.5 * (self.size[0, 0] - self.size[0, 1]), 0]
        if barrier_size is None:
            barrier_size = [7.0/np.sqrt(2*pot_val), 
                            self.size[1, 1] - self.size[1, 0]]
        self.potential = pot_val \
                * np.logical_and(
                        np.heaviside(barrier_loc[0] - self.space_vec[0], 1)
                        + np.heaviside(barrier_loc[0] + barrier_size[0]
                            - self.space_vec[0], 1),
                        np.heaviside(barrier_loc[1] - self.space_vec[1], 1) 
                        + np.heaviside(barrier_loc[1] + barrier_size[1] 
                            - self.space_vec[1], 1))
