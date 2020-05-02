import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_1d(psi, environment):
    # Animate absolute value of the wave function
    psi_mag = np.absolute(psi)
    psi_mag_max = np.max(psi_mag)

    fig = plt.figure()
    ax = plt.axes(xlim=(environment.size[0], environment.size[1]), 
            ylim=(0, 1.1 * psi_mag_max))
    psi_line, = ax.plot([], [], lw=2)
    pot_line, = ax.plot([], [], lw=2)

    def init():
        psi_line.set_data([], [])
        pot_line.set_data([], [])
        return psi_line, pot_line

    def animate(i):
        psi_line.set_data(environment.space_vec, psi_mag[:, i])
        pot_line.set_data(environment.space_vec, environment.potential)
        return psi_line, pot_line

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=psi_mag.shape[1], interval=20, blit=True)
    anim.save('./output/animation_1d.mp4', writer='ffmpeg')
