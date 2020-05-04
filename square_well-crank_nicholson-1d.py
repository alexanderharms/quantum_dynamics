import numpy as np

from quantum_dynamics.environments import SquareWell1D
from quantum_dynamics.waves import PulseWave1D
from quantum_dynamics.vis import animate_1d

# Set parameters
dt = 1e-5  # timestep size
env_bnd = [0, 1]
pot_bnd = [0.30, 0.55]

num_nodes = 2**3
timesteps = 20000

# Initial wave pulse
pos_init = 0.45
mom_init = 0 # Initial momentum
pulse_width = 0.01

# Only record every 10th frame for the animation
anim_constant = 10
animation_frames = int((timesteps+1)/anim_constant)
psi_animate = np.zeros(shape=(num_nodes, animation_frames), 
                       dtype=np.cfloat)

sq_well = SquareWell1D(env_bnd, num_nodes)
sq_well.set_potential(pot_bnd)
print(sq_well.potential)
exit()

pulse = PulseWave1D(pos_init, mom_init, sq_well)
pulse.generate_pulse(pulse_width)
pulse.prep_solver(dt)

anim_count = 0
for t in range(timesteps):
    pulse.solve()

    if (t+1) % anim_constant == 0:
        psi_animate[:, anim_count] = pulse.psi
        anim_count += 1


animate_1d(psi_animate, sq_well)

