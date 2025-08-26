# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:25:50 2025

@author: lucaz

    Use scipy to replace superlu
    
"""

import numpy as np


# prepare directories of diagnostic

#initialise 

# while loop


class RT_1d_simulations:
    def __init__(self):
        self.steps_p_unit = 10
        self.nb_units = 2
        self.frequence = 3
    
        self.nx = 10
        self.xmin = -2
        self.xmax = 2
    
        self.Np = 3
        self.nv = 12
        self.vmin = -3
        self.vmax = 3
    
        self.tn = 0
        self.num = 0
        self.tdiag = 0
    
        self.chi = 0.8
        
        self.eps = 1.
    
        self.D0 = np.zeros((self.Np+1, self.nx+2 )) # consider p=-1 and ghost points in x
        self.C0 = np.zeros((self.Np+1, self.nx+2 )) # consider p=-1 and ghost points in x
        
        # Tools for the linear system
        self.nx+=1  # number of points
        # size of the matrix
        self.n = 2 * self.Np * self.nx
        # number of non-zero values
        self.nhz = 2 * ( (self.Np + 1) * self.nx + 2 * (self.nx - 1 ) * ( 3* self.Np - 2 ) );
        # to construct A with superlu
        self.a = np.zeros(self.nhz) #  dvector(0, nhz-1);  nhz values
        self.asub = np.zeros(self.nhz) # iTvector(0, nhz-1);  nhz values
        self.xa = np.zeros(self.n+1) #iTvector(0, n );  n+1 values
        # to construct rhs with superlu
        self.rhs = np.zeros(self.n) # dvector(1, n); # n values
        self.nx-=1  # back to number of intervals
    
        # zeross for diagnostics
        #-------------------------
        #--> project the PDF on a phase space mesh
        self._tmp_diag = np.zeros( (self.nx+1 , 2*self.nv + 1) ) # dmatrix(0, _nx, -_nv, _nv);
        self._tmp_diag1 = np.zeros(self.nx+1) # dvector(0, _nx);
        self._tmp_diag2 = np.zeros(self.nx+1) # dvector(0, _nx);
        self._tmp_diag3 = np.zeros(self.nx+1) # dvector(0, _nx);
    
        # --> Laguerre polynomials
        self._hp = np.zeros(self.nv+1) # dvector(0, _nv);
        self._hp_1 = np.zeros(self.nv+1) # dvector(0, _nv);
        self._hp_2 = np.zeros(self.nv+1) #dvector(0, _nv);

    def initialise():
        return 0
    
    def build_Am():
        return 0
    
    def bc_transport():
        return 0
    
    def kinetic_iteration():
        return 0
    
    def diagnostic():
        return 0


MySimulation = RT_1d_simulations()








