import numpy as np

from quantum_dynamics.base import Barrier1D, PulseWave1D
from quantum_dynamics.vis import animate_1d

# Set parameters
dt = 1e-5  # timestep size
env_bnd = [0, 1]
pot_bnd = [0.30, 0.55]

num_nodes = 2000
timesteps = 450

# Initial wave pulse
pos_init = 0.3
mom_init = 100 # Initial momentum
pulse_width = 0.05
energy_init = mom_init**2/2

# Only record every 10th frame for the animation
anim_constant = 2 
animation_frames = int((timesteps+1)/anim_constant)
psi_animate = np.zeros(shape=(num_nodes, animation_frames), 
                       dtype=np.cfloat)

barrier = Barrier1D(env_bnd, num_nodes)
barrier.set_potential(pot_bnd, pot_val=energy_init/0.6)

pulse = PulseWave1D(pos_init, mom_init, barrier)
pulse.generate_pulse(pulse_width)
pulse.prep_solver(dt)

anim_count = 0
for t in range(timesteps):
    pulse.solve()

    if (t+1) % anim_constant == 0:
        psi_animate[:, anim_count] = pulse.psi
        anim_count += 1


animate_1d(psi_animate, barrier, 
           animname='./output/1d_tunnel.mp4')
