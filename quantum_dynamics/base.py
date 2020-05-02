import numpy as np

class BaseEnvironment1D():
    def __init__(self, size, num_nodes):
        self.size = size 
        self.num_nodes = num_nodes
        self.node_spacing = (size[1] - size[0]) / (num_nodes-1)  
        self.space_vec = np.linspace(size[0], size[1], num_nodes)

class BaseEnvironment2D():
    def __init__(self, size, num_nodes):
        self.size = size 
        self.num_nodes = num_nodes
        self.node_spacing = (size[:, 1] - size[:, 0]) / (num_nodes - 1)
        lin_x = np.linspspace(size[0, 0], size[0, 1], num_nodes[0])
        lin_y = np.linspspace(size[1, 0], size[1, 1], num_nodes[1])
        space_x, space_y = np.meshgrid(lin_x, lin_y)
        self.space_vec = np.array([space_x.flatten(), space_y.flatten()]).T

class BaseWave1D():
    def __init__(self, pos_init, mom_init, environment):
        self.psi = np.zeros(environment.num_nodes, dtype=np.cfloat)
        self.pos_init = pos_init
        self.mom_init = mom_init
        self.environment = environment


