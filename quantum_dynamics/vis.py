import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

def animate_1d(psi, environment, 
               animname='./output/animation_1d.mp4'):
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
    anim.save(animname, writer='ffmpeg')

def animate_2d(psi, environment, 
               animname='./output/animation_2d.mp4'):
    # Animate absolute value of the wave function
    psi_mag = np.absolute(psi)

    fig = plt.figure()
    ax = plt.axes(xlim=(environment.size[0, 0], environment.size[0, 1]), 
                  ylim=(environment.size[1, 0], environment.size[1, 1]))

    def init():
        psi_cont = ax.contourf(environment.space_vec[0], 
                               environment.space_vec[1], 
                               np.zeros((environment.num_nodes[0],
                                         environment.num_nodes[1])),
                               levels = 10,
                               cmap=plt.cm.bone)
        pot_cont = ax.contour(environment.space_vec[0], 
                              environment.space_vec[1],
                              environment.potential)
        return psi_cont, pot_cont

    def animate(i):
        ax.collections = []
        psi_cont = ax.contourf(environment.space_vec[0], 
                               environment.space_vec[1], 
                               psi_mag[:, :, i],
                               levels = 10,
                               cmap=plt.cm.bone)
        pot_cont = ax.contour(environment.space_vec[0], 
                              environment.space_vec[1], 
                              environment.potential)
        return psi_cont, pot_cont

    anim = FuncAnimation(fig, animate, init_func=init, 
                         frames=psi_mag.shape[2], interval=40)
    writer_obj = animation.writers['ffmpeg']
    writer_obj = writer_obj(fps=25, bitrate=1800)
    anim.save(animname, writer=writer_obj)
