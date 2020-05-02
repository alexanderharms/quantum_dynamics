import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

class CrankNicholson1D():
    """ Mixin class for Crank Nicholson solver in 1D. """
    def prep_solver(self, dt):
        node_spacing = self.environment.node_spacing
        num_nodes = self.environment.num_nodes
        pot = self.environment.potential

        # Make left hand side matrix
        A = sp.diags([1/(4*node_spacing**2), 
                      1j/dt - 1/(2*node_spacing**2), 
                      1/(4*node_spacing**2)], 
                      [-1, 0 ,1], 
                      shape=(num_nodes, num_nodes)).tolil()
        A -= 0.5 * sp.diags(pot, 0)
        A = A.tocsc()

        # Make right hand side vector
        B = sp.diags([-1 / (4*node_spacing**2), 
                      1j/dt + 1/(2*node_spacing**2), 
                      -1 / (4*node_spacing**2)], 
                      [-1, 0, 1], 
                      shape=(num_nodes, num_nodes)).tolil()
        B += 0.5 * sp.diags(pot, 0)
        B = B.tocsc()

        self.solver_preparations = [A, B]

    def solve(self):
        A = self.solver_preparations[0]
        B = self.solver_preparations[1]

        b = B.dot(self.psi)
        self.psi = splinalg.bicgstab(A, b)[0]
        self.psi = self.psi / np.linalg.norm(self.psi)

