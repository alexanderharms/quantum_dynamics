import numpy as np

from .base import BaseWave1D, BaseWave2D
from .solvers import CrankNicholson1D, CrankNicholson2D

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

class PulseWave2D(CrankNicholson2D, BaseWave2D):
    def __init__(self, pos_init, mom_init, environment):
        super().__init__(pos_init, mom_init, environment)
        
    def generate_pulse(self, pulse_width):
        pos_init = self.pos_init
        mom_init = self.mom_init
        space_vec = self.environment.space_vec
        energy_init = mom_init**2 / 2

        psi = np.exp(-((space_vec[0] - pos_init[0])**2 /
                        (2*pulse_width[0]**2) +
                       (space_vec[1] - pos_init[1])**2 /
                        (2*pulse_width[1]**2))) \
               * np.exp(1j*space_vec[0]*mom_init[0] 
                        + 1j*space_vec[1]*mom_init[1])

        psi = psi / np.linalg.norm(psi)
        self.psi += psi
