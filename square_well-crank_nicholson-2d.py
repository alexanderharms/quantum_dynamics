import numpy as np

from quantum_dynamics.environments import SquareWell2D
from quantum_dynamics.waves import PulseWave2D
from quantum_dynamics.vis import animate_2d

import matplotlib.pyplot as plt
# Set parameters
dt = 1e-5  # timestep size
env_bnd = [[0, 1], [0, 1]]
pot_bnd = [[0.3, 0.7], [0.2, 0.8]]

num_nodes = [2**3, 2**3]
timesteps = 15000

# Initial wave pulse
pos_init = [0.5, 0.5] 
# Initial momentum
mom_init = [500, 500] 
pulse_width = [1/2**7, 1/2**4]

# Only record every 10th frame for the animation
anim_constant = 10
animation_frames = int((timesteps+1)/anim_constant)
psi_animate = np.zeros(shape=(num_nodes[0], num_nodes[1],
                              animation_frames), 
                       dtype=np.cfloat)

print("Generate square well potential...")
sq_well = SquareWell2D(env_bnd, num_nodes)
sq_well.set_potential(pot_bnd)
print(sq_well.potential)

print("Generate pulse and prep solver...")
pulse = PulseWave2D(pos_init, mom_init, sq_well)
pulse.generate_pulse(pulse_width)
pulse.prep_solver(dt)

print("Start solving...")
anim_count = 0
for t in range(timesteps):
    pulse.solve()

    if (t+1) % anim_constant == 0:
        psi_animate[:, :, anim_count] = pulse.psi
        anim_count += 1


print("Done solving, start animating...")
animate_2d(psi_animate, sq_well, animname="./output/animate_2d.mp4")

