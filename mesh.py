import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

# Class mesh is an object that contains the node vector and the connection matrix
# The initialization function of the mesh class requires as inputs:
# 1. the horizontal mesh size
# 2. the vertical mesh size
# 3. the vertices of th domain geometry in clock-wise order

#                /|   
#               / |   
#              /  | h_vert  
#             /   |
#            /____|
#             h_or


class mesh(object):

    #Initialization of the mesh class
    def __init__(self, h_or, h_vert, vertices):
        self.h_or = h_or
        self.h_vert = h_vert
        self.v0 = vertices[0]
        self.v1 = vertices[1]
        self.v2 = vertices[2]
        self.v3 = vertices[3]
        self.compute_mesh() #Compute the mesh
        self.tag_list = [self.tag_list_1, self.tag_list_2, self.tag_list_3, self.tag_list_4]


    #Computes the structured mesh
    def compute_mesh(self):
        print("Computing mesh ...")
        n = abs(self.v0[1] - self.v3[1]) / self.h_vert
        conn_matrix = [] #Initializing the connection matrix 
        self.tag_list_2 = [] #List of nodes at the right boundary
        self.tag_list_4 = [] #List of nodes at the left boundary
        print(n)
        if n - int(n) == 0:
            n = int(n) 
        else:
            raise ValueError("The number of nodes of the column is not an integer")
        for ii in range(n+1): #For each row ... do
            left_v = (self.v3 - self.v0) / n * (ii) + self.v0  #top left node
            right_v = (self.v2 - self.v1) / n * (ii) + self.v1 #top right node
            if ii == 0: #If this is the top row
                nodes = self.compute_row(left_v, right_v) #Compute nodes coordinates in the present row
                n_old = np.size(nodes, 0) 
                self.tag_list_1 = range(n_old) #Add all the nodes of the row to the tag list of the top boundary
                self.tag_list_2.append(n_old - 1) #Add the right node of the row to the tag list of the right boundary
                self.tag_list_4.append(0) #Add the left node of the row to the tag list of the left boundary
            else:
                row_new = self.compute_row(left_v, right_v) #Compute nodes coordinates in the present row
                n_new = np.size(row_new, 0)
                n_apt = np.size(nodes, 0)
                nodes = np.concatenate((nodes, row_new), axis=0)
                self.tag_list_2.append(n_apt + n_new - 1) #Add the right node of the row to the tag list of the right boundary
                self.tag_list_4.append(n_apt) #Add the left node of the row to the tag list of the left boundary
                

                #Compute the connection matrix 
                for k in range(n_new):
                    el_1 = [int(n_apt + k - n_old), int(n_apt + k - n_old + 1), int(n_apt + k)] #First triangle of the square
                    el_2 = [int(n_apt + k - n_old + 1), int(n_apt + k + 1), int(n_apt + k),] #Second triabgle of the square
                    conn_matrix.append(el_1)
                    if k < n_new-1: conn_matrix.append(el_2) 
                n_old = n_new
            if ii == n:
                self.tag_list_3 = range(n_apt, n_apt + n_new)
        print("Number of nodes: ", np.size(nodes, 0))
        print("Number of elements: ", np.size(conn_matrix, 0))

        self.nodes = nodes
        self.conn_matrix = conn_matrix

    #Computes the x coordinate of each node of the row
    def compute_row(self, left_v, right_v):
        n = abs(right_v[0] - left_v[0]) / self.h_or
        if n - int(n) == 0:
            n = int(n)
        else:
            raise ValueError("The number of nodes of the row is not an integer")
        
        v_row = np.linspace(left_v, right_v, n+1) 
        return v_row

    def plot_mesh(self):
        nodes = self.nodes
        elements = self.conn_matrix
        fig, ax = plt.subplots(figsize=(20,20))
        patches_list = []

        for element in elements:
            polygon = patches.Polygon(nodes[element], closed=True, edgecolor='k')
            patches_list.append(polygon)

        patch_collection = PatchCollection(patches_list, facecolor='lightblue', edgecolor='k', linewidth=1)
        ax.add_collection(patch_collection)

        tagged_coords = nodes[self.tag_list_1]
        ax.scatter(tagged_coords[:, 0], tagged_coords[:, 1], color='red', s=50, label='Upper bound tag', zorder=3)

        tagged_coords = nodes[self.tag_list_2]
        ax.scatter(tagged_coords[:, 0], tagged_coords[:, 1], color='yellow', s=50, label='Right bound tag', zorder=3)

        tagged_coords = nodes[self.tag_list_4]
        ax.scatter(tagged_coords[:, 0], tagged_coords[:, 1], color='green', s=50, label='Left bound tag', zorder=3)

        tagged_coords = nodes[self.tag_list_3]
        ax.scatter(tagged_coords[:, 0], tagged_coords[:, 1], color='blue', s=50, label='Bottom bound tag', zorder=3)

    



        ax.autoscale()
        ax.set_aspect('equal')
        plt.title('Mesh Plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.show()
    