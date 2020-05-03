import numpy as np

from .base import BaseEnvironment1D, BaseEnvironment2D

class SquareWell1D(BaseEnvironment1D):
    def __init__(self, size, num_nodes):
        super().__init__(size, num_nodes)
        self.potential = np.zeros(self.num_nodes)

    def set_potential(self, size_pot, pot_val=1e6):
        self.potential = -pot_val \
                * (np.heaviside(size_pot[0] - self.space_vec, 1)
                - np.heaviside(size_pot[1] - self.space_vec, 1))

class SquareWell2D(BaseEnvironment2D):
    def __init__(self, size, num_nodes):
        super().__init__(size, num_nodes)
        self.potential = np.zeros((self.num_nodes, self.num_nodes))

    def set_potential(self, size_pot, pot_val=1e6):
        self.potential[:, 0] = -pot_val \
                * (np.heaviside(size_pot[0, 0] - self.space_vec[:, 0], 1)
                - np.heaviside(size_pot[0, 1] - self.space_vec[:, 0], 1))
        self.potential[:, 1] = -pot_val \
                * (np.heaviside(size_pot[1, 0] - self.space_vec[:, 1], 1)
                - np.heaviside(size_pot[1, 1] - self.space_vec[:, 1], 1))

class Barrier1D(BaseEnvironment1D):
    def __init__(self, size, num_nodes):
        super().__init__(size, num_nodes)
        self.potential = np.zeros(self.num_nodes)

    def set_potential(self, size_pot, pot_val, barrier_width=None):
        if barrier_width is None:
            self.barrier_width = 7.0 / np.sqrt(2 * pot_val)
        else: 
            self.barrier_width = barrier_width
        print(self.barrier_width)
        self.potential = -pot_val \
                * (np.heaviside(size_pot[0] - self.space_vec, 1)
                - np.heaviside(size_pot[1] - self.space_vec 
                    + self.barrier_width, 1))
