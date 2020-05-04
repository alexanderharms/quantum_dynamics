import numpy as np

class BaseEnvironment1D():
    def __init__(self, size, num_nodes):
        self.size = size 
        self.num_nodes = num_nodes
        self.node_spacing = (size[1] - size[0]) / (num_nodes-1)  
        self.space_vec = np.linspace(size[0], size[1], num_nodes)

class BaseEnvironment2D():
    def __init__(self, size, num_nodes):
        self.size = np.array(size)
        self.num_nodes = np.array(num_nodes)
        self.node_spacing = (self.size[:, 1] - self.size[:, 0]) \
                / (self.num_nodes - 1)
        lin_x = np.linspace(self.size[0, 0], self.size[0, 1], 
                            self.num_nodes[0])
        lin_y = np.linspace(self.size[1, 0], self.size[1, 1], 
                            self.num_nodes[1])
        space_x, space_y = np.meshgrid(lin_x, lin_y)
        self.space_vec = [space_x, space_y]

class BaseWave1D():
    def __init__(self, pos_init, mom_init, environment):
        self.psi = np.zeros(environment.num_nodes, dtype=np.cfloat)
        self.pos_init = pos_init
        self.mom_init = mom_init
        self.environment = environment

class BaseWave2D():
    def __init__(self, pos_init, mom_init, environment):
        self.psi = np.zeros((environment.num_nodes[0], 
                             environment.num_nodes[1]), 
                             dtype=np.cfloat)
        self.pos_init = np.array(pos_init)
        self.mom_init = np.array(mom_init)
        self.environment = environment

