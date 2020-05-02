import numpy as np

from .base import BaseWave1D
from .solvers import CrankNicholson1D

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
