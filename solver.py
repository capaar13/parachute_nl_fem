import numpy as np
import scipy.linalg as sp
from assembling import stiff_matrix, rhs

#Definition of solver class
class solver(object):
    #Initialization requires:
    #The initial Guess
    #The resdiual tolerance
    #The increment tolerance
    #These variables remain constant across iterations

    #The initial sitffness matrix
    #The initial RHS vector
    #The initial Residual
    def __init__(self, guess, epsilon_R, epsilon_d, K, F, R):
        self.epsilon_R = epsilon_R
        self.epsilon_d = epsilon_d
        self.Res = R.vect
        self.delta = self.linear_solver(K.vect, self.Res)
        self.d = guess
        self.cond_1 = np.linalg.norm(self.Res) / np.linalg.norm(F.vect) < self.epsilon_R
        self.cond_2 = np.linalg.norm(self.delta) / np.linalg.norm(self.d.vect) < self.epsilon_d
        self.Res_history = []
        self.delta_history = []

    #Define the linear solver used to solve the system A * Delta_d = R
    def linear_solver(self, A, R):
        u = sp.solve(A, R)
        return u
    
    #Method used to upgrade the infos of the newton iteration and 
    # extract the incrment by solving the linear system
    def newton_iter(self, K, F, R):
        self.Res = R.vect
        print("RESIDUAL = ", np.linalg.norm(self.Res) / np.linalg.norm(F.vect))
        self.cond_1 = np.linalg.norm(self.Res) / np.linalg.norm(F.vect) < self.epsilon_R
        self.cond_2 = np.linalg.norm(self.delta) / np.linalg.norm(self.d.vect) < self.epsilon_d
        self.delta = self.linear_solver(K.vect, self.Res)
        self.d.vect = self.d.vect + self.delta
        self.Res_history.append(np.linalg.norm(self.Res))
        self.delta_history.append(np.linalg.norm(self.delta))

    #Method used to verify the convergence conditions
    def verification(self):
        if self.cond_1 and self.cond_2:
            print("Conditions are respected")
            ver = True
        elif self.cond_1 and not(self.cond_2):
            print("Only residual criterium is respected")
            ver = False
        elif self.cond_2 and not(self.cond_1):
            ver = False
            print("Only increment criterium is respected")
        else:
            ver = False
            print("No criterium is respected")
        return ver
    
    #Getter of the solution at the present iteration
    def get_solution(self):
        #print(self.d.vect)
        return self.d

        

        

