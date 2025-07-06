import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

#Class of the mesh element (triangle)
class elem(object):

    def __init__(self, vertices):
        self.vertices = vertices
        self.D = self.compute_surf()
        self.center = self.compute_center()
        self.phi, self.grad_phi = self.compute_functions()
    
    #Compute the shape functions and their gradient
    def compute_functions(self):
        v = self.vertices
        D = self.D
        v0 = v[0,:]
        v1 = v[1,:]
        v2 = v[2,:]
        x0,y0 = v0[0],v0[1]
        x1,y1 = v1[0],v1[1]
        x2,y2 = v2[0],v2[1]
        phi_0 = lambda x,y: 1 / D * ( (x1 * y2 - x2 * y1) + (y1 - y2)*x + (x2 - x1)*y)
        phi_1 = lambda x,y: 1 / D * ( (x2 * y0 - x0 * y2) + (y2 - y0)*x + (x0 - x2)*y)
        phi_2 = lambda x,y: 1 / D * ( (x0 * y1 - x1 * y0) + (y0 - y1)*x + (x1 - x0)*y)
        phi = [phi_0, phi_1, phi_2]
        phi_0x = 1 / D * (y1 - y2)
        phi_0y = 1 / D * (x2 - x1)
        phi_1x = 1 / D * (y2 - y0)
        phi_1y = 1 / D * (x0 - x2)
        phi_2x = 1 / D * (y0 - y1)
        phi_2y = 1 / D * (x1 - x0)
        grad_phi = [[ phi_0x, phi_0y], [phi_1x, phi_1y], [phi_2x, phi_2y]]
        return phi, grad_phi

    
    #Compute the determinant of thecoordinate matrix D = 2 * Area
    def compute_surf(self):
        v = self.vertices
        v0 = v[0,:]
        v1 = v[1,:]
        v2 = v[2,:]
        x0,y0 = v0[0],v0[1]
        x1,y1 = v1[0],v1[1]
        x2,y2 = v2[0],v2[1]
        return (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0)
    
    #Compute the coordinates of the center of the element
    def compute_center(self):
        v = self.vertices
        v0 = v[0,:]
        v1 = v[1,:]
        v2 = v[2,:]
        x0,y0 = v0[0],v0[1]
        x1,y1 = v1[0],v1[1]
        x2,y2 = v2[0],v2[1]

        return [(x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3]


    #Compute the linear stiffness matrix of the element
    #Gauss quadrature at the centroid
    def stiff_elem(self,a,b, T):
        grad_phi_A = self.grad_phi[a]
        grad_phi_B = self.grad_phi[b] 
        D = self.D
        return D / 2 * ( grad_phi_A[0]*grad_phi_B[0] + grad_phi_A[1]*grad_phi_B[1]) * T
    
    #Compute the non linear functional vector G_NL of the element
    #Gauss quadrature at the centroid
    def non_lin_functional(self, a,d_vect, alpha, beta, S, L):
        xc = self.center[0]
        yc = self.center[1]
        D = self.D
        phi_A = self.phi[a]
        uh = self.phi[0](xc,yc)*d_vect[0] + self.phi[1](xc,yc)*d_vect[1] + self.phi[2](xc,yc)*d_vect[2]
        return  uh * S * D / 2 * phi_A(xc, yc) * (1 - beta * uh / L + alpha * (uh / L)**2)
    
    #Compute non linear mass matrix of the element (first part -> grade 1)
    #Gauss quadrature at the centroid
    def mass_non_lin_elem_1(self,a,b,d_vect, alpha, beta, S, L):
        xc = self.center[0]
        yc = self.center[1]
        D = self.D
        phi_A = self.phi[a]
        phi_B = self.phi[b]
        uh = self.phi[0](xc,yc)*d_vect[0] + self.phi[1](xc,yc)*d_vect[1] + self.phi[2](xc,yc)*d_vect[2]
        return  S * D / 2  * (phi_A(xc,yc) * (1 - beta * uh * 1 / L +  alpha * (uh**2) * (1 / L ** 2))) * phi_B(xc,yc)
    
    #Compute non linear mass matrix of the element (second part -> grade 2)
    #Gauss quadrature at the centroid
    def mass_non_lin_elem_2(self,a,b,d_vect, alpha, beta, S, L):
        xc = self.center[0]
        yc = self.center[1]
        D = self.D
        phi_A = self.phi[a]
        phi_B = self.phi[b]
        uh = self.phi[0](xc,yc)*d_vect[0] + self.phi[1](xc,yc)*d_vect[1] + self.phi[2](xc,yc)*d_vect[2]
        I1 = (-beta * 1 / L + alpha * 2 * (uh) * (1 / L**2))* phi_B(xc,yc)
        return  D / 2 * S * phi_A(xc,yc) * I1 * uh
    
    #Compute the Right Hand Side vector of the element
    #Gauss quadrature at the centroid
    def compute_rhs(self,a,f):
        xc = self.center[0]
        yc = self.center[1]
        D = self.D
        phi_A = self.phi[a]
        return D / 2 * phi_A(xc,yc) * f([xc, yc])

#Baseline class for global matrices and vectors of the problem.
#It contains the shared methods and variables.

class matrix_base:
    def __init__(self, mesh):
        self.nodes = mesh.nodes
        self.conn_matrix = mesh.conn_matrix
        self.tag_list = mesh.tag_list
        self.vect = None

    def getShape(self):
        return self.vect.shape
    
    def apply_DC(self, boundaries):
        tag_list = self.tag_list
        tag_list_tot = []
        for ii in boundaries:
            tag_list_tot += tag_list[ii]

        # Gestisce sia vettori che matrici
        if self.vect.ndim == 2:
            A_new = np.delete(self.vect, tag_list_tot, axis=0)  # rimuove righe
            self.vect = np.delete(A_new, tag_list_tot, axis=1)  # rimuove colonne
        elif self.vect.ndim == 1:
            self.vect = np.delete(self.vect, tag_list_tot, axis=0)  # rimuove elementi

    def print(self):
        np.set_printoptions(threshold=np.inf)
        print(f"{self.__class__.__name__.upper()} = ", self.vect)

#Class of the Global Non linear functional G_NL based on the class matrix_base
class non_linear_functional(matrix_base):
    def __init__(self, mesh, alpha, beta, S, L):
        self.nodes = mesh.nodes
        self.conn_matrix = mesh.conn_matrix
        self.tag_list = mesh.tag_list
        self.alpha = alpha
        self.beta = beta
        self.S = S
        self.L = L
        self.vect = np.array([])

    #Assemble the global vector 
    def assemble(self, d):
        print("assembling non linear functional ... ")
        n = np.size(self.conn_matrix,0)
        n_nodes = np.size(self.nodes, 0)
        matrix = np.zeros((n_nodes))
        alpha = self.alpha
        beta = self.beta
        S = self.S
        L = self.L
        for ii in range(n):
            nodes_idx = self.conn_matrix[ii]
            vertices = self.nodes[nodes_idx,:]
            d_vect = d[nodes_idx]
            el = elem(vertices)
            for idx_1 in range(3):
                matrix[nodes_idx[idx_1]] += el.non_lin_functional(idx_1, d_vect, alpha, beta, S, L)
                if np.isnan(el.non_lin_functional(idx_1, d_vect, alpha, beta, S, L)):
                    print("D = ", el.D)
                    print(ii, "is NaN")
                    exit()
        self.vect = matrix
        return matrix

#Class of the Global linear Stiffness matrix, based on the class matrix_base
class stiff_matrix(matrix_base):
    def __init__(self, mesh, T):
        self.nodes = mesh.nodes
        self.conn_matrix = mesh.conn_matrix
        self.tag_list = mesh.tag_list
        self.T = T

    def assemble(self):
        print("assembling linear stiffness matrix ... ")
        n = np.size(self.conn_matrix,0)
        n_nodes = np.size(self.nodes, 0)
        matrix = np.zeros((n_nodes,n_nodes))
        for ii in range(n):
            nodes_idx = self.conn_matrix[ii]
            vertices = self.nodes[nodes_idx,:]
            el = elem(vertices)
            
            for idx_1 in range(3):
                for idx_2 in range(3):
                    matrix[nodes_idx[idx_1], nodes_idx[idx_2]] += el.stiff_elem(idx_1, idx_2, self.T)
                if np.isnan(el.stiff_elem(idx_1, idx_2, self.T)):
                    print(ii, " is NaN")
                    exit()
        self.vect = matrix
        return matrix
    

#Class of the Global non linear mass matrix (first part -> grade 1), based on the class matrix_base
class mass_non_lin_matrix_1(matrix_base):
    def __init__(self,mesh, alpha, beta, S, L):
        self.nodes = mesh.nodes
        self.conn_matrix = mesh.conn_matrix
        self.tag_list = mesh.tag_list
        self.alpha = alpha
        self.beta = beta
        self.S = S
        self.L = L
        self.vect = np.array([])

    def assemble(self, d):
        print("assembling mass matrix 1 ... ")
        n = np.size(self.conn_matrix,0)
        n_nodes = np.size(self.nodes, 0)
        matrix = np.zeros((n_nodes,n_nodes))
        alpha = self.alpha
        beta = self.beta
        S = self.S
        L = self.L
        for ii in range(n):
            nodes_idx = self.conn_matrix[ii]
            vertices = self.nodes[nodes_idx,:]
            d_vect = d[nodes_idx]
            el = elem(vertices)
            
            for idx_1 in range(3):
                for idx_2 in range(3):
                    matrix[nodes_idx[idx_1], nodes_idx[idx_2]] += el.mass_non_lin_elem_1(idx_1, idx_2, d_vect, alpha, beta, S, L)
                if np.isnan(el.mass_non_lin_elem_1(idx_1, idx_2, d_vect, alpha, beta, S, L)):
                    print(ii, " is NaN")
                    exit()
        self.vect = matrix
        return matrix

#Class of the Global non linear mass matrix (Second part -> grade 2), based on the class matrix_base
class mass_non_lin_matrix_2(matrix_base):
    def __init__(self, mesh, alpha, beta, S, L):
        self.nodes = mesh.nodes
        self.conn_matrix = mesh.conn_matrix
        self.tag_list = mesh.tag_list
        self.alpha = alpha
        self.beta = beta
        self.S = S
        self.L = L
        self.vect = np.array([])

    def assemble(self, d):
        print("assembling mass matrix 2 ... ")
        n = np.size(self.conn_matrix,0)
        n_nodes = np.size(self.nodes, 0)
        matrix = np.zeros((n_nodes,n_nodes))
        alpha = self.alpha
        beta = self.beta
        S = self.S
        L = self.L
        for ii in range(n):
            nodes_idx = self.conn_matrix[ii]
            vertices = self.nodes[nodes_idx,:]
            d_vect = d[nodes_idx]
            el = elem(vertices)
            
            for idx_1 in range(3):
                for idx_2 in range(3):
                    matrix[nodes_idx[idx_1], nodes_idx[idx_2]] += el.mass_non_lin_elem_2(idx_1, idx_2, d_vect, alpha, beta, S, L)
                if np.isnan(el.mass_non_lin_elem_2(idx_1, idx_2, d_vect, alpha, beta, S, L)):
                    print(ii, " is NaN")
                    exit()
        self.vect = matrix
        return matrix
    
#Class of the global Right Hand Side vector, based on the class matrix_base
class rhs(matrix_base):
    def __init__(self, mesh, f):
        self.f = f
        self.nodes = mesh.nodes
        self.conn_matrix = mesh.conn_matrix
        self.tag_list = mesh.tag_list
        self.vect = self.assemble()

    def assemble(self):
        print("assembling right hand side ... ")
        n = np.size(self.conn_matrix,0)
        n_nodes = np.size(self.nodes, 0)
        matrix = np.zeros((n_nodes))
        f = self.f
        for ii in range(n):
            nodes_idx = self.conn_matrix[ii]
            vertices = self.nodes[nodes_idx,:]
            el = elem(vertices)
            for idx_1 in range(3):
                matrix[nodes_idx[idx_1]] += el.compute_rhs(idx_1, f)
                if np.isnan(el.compute_rhs(idx_1, f)):
                    print("D = ", el.D)
                    print(ii, "is NaN")
                    exit()
        self.vect = matrix
        return matrix
    
    def plot(self):
        nodes = self.nodes
        elements = self.conn_matrix
        u = self.vect
        # Create a triangulation object
        triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

        # Plot the solution as a color map
        plt.figure(figsize=(8, 6))
        tpc = plt.tripcolor(triangulation, u, shading='flat', cmap='viridis')
        plt.colorbar(tpc, label='Solution value')
        plt.title('Solution over the Mesh')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.show()

#Class of the solution vector, based on the class matrix_base
class solution(matrix_base):
    def __init__(self, mesh):
        self.nodes = mesh.nodes
        self.conn_matrix = mesh.conn_matrix
        self.tag_list = mesh.tag_list
        self.vect = np.zeros(np.size(self.nodes, 0))

    def restore_DC_dofs(self, boundaries):
        tag_list = self.tag_list
        tag_list_tot = []
        for ii in boundaries:
            tag_list_tot += tag_list[ii]
        sol = np.zeros(np.size(self.nodes, 0))
        list_global = range(len(sol))
        list_full = [item for item in list_global if item not in tag_list_tot]
        sol[list_full] = self.vect
        self.vect = sol

    def assemble(self, f):
        F = np.zeros(np.size(self.nodes, 0))
        for idx,node in enumerate(self.nodes):
            F[idx] = f(node)
            self.vect = F

    def print(self):
        np.set_printoptions(threshold=np.inf)
        print("Solution = ", self.vect)

    

    def plot(self):
        nodes = self.nodes
        elements = self.conn_matrix
        u = self.vect
        # Create a triangulation object
        triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

        # Plot the solution as a color map
        plt.figure(figsize=(8, 6))
        tpc = plt.tripcolor(triangulation, u, shading='flat', cmap='viridis')
        plt.colorbar(tpc, label='Solution value')
        plt.title('Solution over the Mesh')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.show()
    
    def plot_overline(self, side, axis):

        if side == 0:
            side_name = "Upper bound"
        elif side == 1:
            side_name = "Right bound"
        elif side == 2:
            side_name = "Bottom bound"
        else:
            side_name = "Left bound"

        if axis == 0:
            axis_name = "x"
        elif axis == 1:
            axis_name = "y"
        else:
            axis_name = "s"
        plot_nodes = self.nodes[self.tag_list[side]]
        if axis < 2:
            plot_coord = plot_nodes[:,axis]
        else:
            diff = np.array(plot_nodes[-1]) - np.array(plot_nodes[0])
            diff = np.linalg.norm(diff)
            plot_coord = np.linspace(0, diff, len(plot_nodes))

        
        plot_vect = self.vect[self.tag_list[side]]
        plot = plt.plot(plot_coord, plot_vect)
        plt.title(f'Solution over {side_name}')
        plt.xlabel(axis_name)
        plt.ylabel('solution')
        plt.gca().set_aspect('auto')
        plt.grid(True)
        return plot
        

        

        
                
                


    