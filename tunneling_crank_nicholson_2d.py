import numpy as np

from quantum_dynamics.environments import Barrier2D
from quantum_dynamics.waves import PulseWave2D
from quantum_dynamics.vis import animate_2d

# Set parameters
dt = 1e-5  # timestep size
env_bnd = [[0, 1], [0, 1]]
barrier_loc = [0.5, 0]

num_nodes = [2**7, 2**7]
timesteps = 1000

# Initial wave pulse
pos_init = [0.3, 0.3]
mom_init = [200, 0] # Initial momentum
pulse_width = [0.05, 0.05]
energy_init = np.array(mom_init)**2/2

# Only record every 10th frame for the animation
anim_constant = 2 
animation_frames = int((timesteps+1)/anim_constant)
psi_animate = np.zeros(shape=(num_nodes[0], num_nodes[1], animation_frames), 
                       dtype=np.cfloat)

barrier = Barrier2D(env_bnd, num_nodes)
barrier.set_potential(pot_val=energy_init[0]/0.6, barrier_loc=barrier_loc)

pulse = PulseWave2D(pos_init, mom_init, barrier)
pulse.generate_wave(pulse_width)
pulse.prep_solver(dt)

anim_count = 0
for t in range(timesteps):
    pulse.solve()

    if (t+1) % anim_constant == 0:
        psi_animate[:, :, anim_count] = pulse.psi
        anim_count += 1


animate_2d(psi_animate, barrier, 
           animname='./output/2d_tunnel.mp4')
