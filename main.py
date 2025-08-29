# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:25:50 2025

@author: lucaz

    Use scipy to replace superlu
    
"""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

# prepare directories of diagnostic

#initialise 

# while loop


class RT_1d_simulations:
    def __init__(self):
        self.steps_p_unit = 10
        self.nb_units = 2
        self.frequence = 2
    
        self.nx = 10
        self.xmin = -4
        self.xmax = 4
    
        self.Np = 3
        self.nv = 12
        self.vmin = -3
        self.vmax = 3
    
        self.tn = 0
        self.num = 0
        self.tdiag = 0
    
        self.chi = 0.8
        self.sigma = np.zeros(self.nx+1) # sigma = dvector(0, _nx )
        
        self.eps = 1.
    
        self.D0 = np.zeros((self.Np+2, self.nx+3 )) # consider p=-1 and ghost points in x
        self.C0 = np.zeros((self.Np+2, self.nx+3 )) # consider p=-1 and ghost points in x
        # _D0 = dmatrix(-1, _Np, -1, _nx + 1 );
        # _C0 = dmatrix(-1, _Np, -1, _nx + 1 );
        
        
        # Tools for the linear system
        self.nx+=1  # number of points
        # size of the matrix
        self.n = 2 * self.Np * self.nx
        # number of non-zero values
        self.nhz = 2 * ( (self.Np + 1) * self.nx + 2 * (self.nx - 1 ) * ( 3* self.Np - 2 ) );
        #------------------------------
        # to construct A with superlu
        #------------------------------        
        # non-zero values of the matrix,
        self.a = np.zeros(self.nhz) #  dvector(0, nhz-1);  nhz values
        # their row indeces
        self.asub = np.zeros(self.nhz) # iTvector(0, nhz-1);  nhz values
        #  the indices indicating the beginning of each column in the coefficient and row index array
        self.xa = np.zeros(self.n+1) #iTvector(0, n );  n+1 values
        self.lu = None
        
        # to construct rhs with superlu
        self.rhs = np.zeros(self.n) # dvector(1, n); # n values
        self.nx-=1  # back to number of intervals
    
        # zeross for diagnostics
        #-------------------------
        #--> project the PDF on a phase space mesh
        self.tmp_diag = np.zeros( (self.nx+1 , 2*self.nv + 1) ) # dmatrix(0, _nx, -_nv, _nv);
        self.tmp_diag1 = np.zeros(self.nx+1) # dvector(0, _nx);
        self.tmp_diag2 = np.zeros(self.nx+1) # dvector(0, _nx);
        self.tmp_diag3 = np.zeros(self.nx+1) # dvector(0, _nx);
    
        # --> Laguerre polynomials
        self.hp = np.zeros(self.nv+1) # dvector(0, _nv);
        self.hp_1 = np.zeros(self.nv+1) # dvector(0, _nv);
        self.hp_2 = np.zeros(self.nv+1) #dvector(0, _nv);
    
    def bc_transport(self, D):
        D[0] = 0
        D[-1] = 0
        return

    def initialise(self):
        #dx = (self._xmax - self._xmin) / self._nx;
        x= np.linspace(self.xmin, self.xmax, self.nx +1)
        
        self.D0[0, 1:-1] = np.exp(-0.5 * (x-1)**2) 
        self.C0[0, 1:-1] = np.exp(-0.5 * (x-1)**2)
        self.sigma = np.sign(x)
        
        self.bc_transport(self.D0);
        self.bc_transport(self.C0);

        return
    
    def build_Am(self):
        dt = 1/self.steps_p_unit
        dx = (self.xmax - self.xmin) / self.nx;
        tauTrans = dt / (self.eps * dx);
        
        self.nx+=1; # number of points in x
        
        i = 0;    # indicizza a e asub
        i0 = 0;   # Scorre il numero delle colonne e indicizza xa
        self.xa[i0] = 0;
        
        #---------------|
        #  FIRST HALH   |
        #---------------|
        #===================================================================================================
        # K = 0
        #===================================================================================================
        
        # PRIMA COLONNA 
        # j = 0
        self.a[i] = 1 + dt / (2 * self.eps * self.eps) + self.chi * dt * self.sigma[0] / (2 * self.eps) + tauTrans;        # 1  SIGMA
        self.asub[i] = i0;
        i +=1;
        self.a[i] = -tauTrans;                                        # 2  TRANSPORT
        self.asub[i] = i0 + 1;
        i +=1;
        self.a[i] = -tauTrans;                                        # 3  TRANSPORT
        self.asub[i] = i0 + self.nx;
        i +=1;
        self.a[i] = +tauTrans;                                        # 4  TRANSPORT
        self.asub[i] = i0 + self.nx + 1;
        i +=1;
        self.a[i] = -dt / (2 * self.eps * self.eps) - self.chi * dt * self.sigma[0] / (2 * self.eps);   # 5  SIGMA
        self.asub[i] = i0 + self.nx * self.Np;
        i +=1;
        
        self.xa[i0 + 1] = self.xa[i0] + 5;
        
        
        # COLONNE INTERMEDIE
        # j = 1, ..., nx - 2
        for j in range(1, self.nx - 1): # (int j = 1; j <= self.nx - 2; j++) 
        
            i0 = j;
        
            self.a[i] = 1 + dt / (2 * self.eps * self.eps) + self.chi * dt * self.sigma[j] / (2 * self.eps)+ tauTrans;         # 1  SIGMA  
            self.asub[i] = i0;
            i +=1;
            self.a[i] = -tauTrans;     # 2  TRANSPORT
            self.asub[i] = i0 + 1;
            i +=1;
            self.a[i] = -tauTrans;     # 3  TRANSPORT
            self.asub[i] = i0 + self.nx;
            i +=1;
            self.a[i] = +tauTrans;     # 4  TRANSPORT
            self.asub[i] = i0 + 1 + self.nx;
            i +=1;
            self.a[i] = -dt / (2 * self.eps * self.eps) - self.chi * dt * self.sigma[j] / (2 * self.eps);         # 5  SIGMA
            self.asub[i] = i0 + self.Np * self.nx;
            i +=1;
        
            self.xa[i0 + 1] = self.xa[i0] + 5;
        
        
        # ULTIMA COLONNA 
        # j = nx-1
        i0 = self.nx - 1;
        
        self.a[i] = 1 + dt / (2 * self.eps * self.eps) + self.chi * dt * self.sigma[self.nx - 1] / (2 * self.eps) + tauTrans;       # 1 SIGMA
        self.asub[i] = i0;
        i +=1;
        self.a[i] = -tauTrans;                                              # 2  TRANSPORT
        self.asub[i] = i0 + self.nx;
        i +=1;
        self.a[i] = -dt / (2 * self.eps * self.eps) - self.chi * dt * self.sigma[self.nx - 1] / (2 * self.eps);            # 3  SIGMA
        self.asub[i] = i0 + self.Np * self.nx;
        i +=1;
        
        self.xa[i0 + 1] = self.xa[i0] + 3;
        
        #===================================================================================================
        # K = 1, ..., Np - 2
        #===================================================================================================
        
        for p in range(1, self.Np - 1): # (int p = 1; p <= self.Np - 2; p++) 
        
            # PRIMA COLONNA
            # j = 0
            i0 = self.nx * p;
        
            self.a[i] = -p * tauTrans;    # 1  TRANSPORT
            self.asub[i] = i0 - self.nx;
            i +=1;
            self.a[i] = p * tauTrans;    # 2  TRANSPORT
            self.asub[i] = i0 - self.nx + 1;
            i +=1;
            self.a[i] = 1 + dt / (self.eps * self.eps) + self.chi * dt * self.sigma[0] / self.eps + (2 * p + 1) * tauTrans;         # 3  SIGMA
            self.asub[i] = i0;
            i +=1;
            self.a[i] = -(2 * p + 1) * tauTrans;         # 4  TRANSPORT
            self.asub[i] = i0 + 1;
            i +=1;
            self.a[i] = -(p + 1) * tauTrans;               # 5  TRANSPORT
            self.asub[i] = i0 + self.nx;
            i +=1;
            self.a[i] = (p + 1) * tauTrans;               # 6  TRANSPORT
            self.asub[i] = i0 + self.nx + 1;
            i +=1;
        
            self.xa[i0 + 1] = self.xa[i0] + 6;
        
        
            # COLONNE INTERMEDIE
            # j = 1,... nx-2
            for j in range(1, self.nx -1): # (int j = 1; j <= self.nx - 2; j++) 
                i0 = j + self.nx * p;
        
                self.a[i] = -p * tauTrans;         # 1  TRANSPORT
                self.asub[i] = i0 - self.nx;
                i +=1;
                self.a[i] = +p * tauTrans;    # 2  TRANSPORT
                self.asub[i] = i0 - self.nx + 1;
                i +=1;
                self.a[i] = 1 + dt / (self.eps * self.eps) + self.chi * dt * self.sigma[j] / self.eps + (2 * p + 1) * tauTrans;    # 3  SIGMA
                self.asub[i] = i0;
                i +=1;
                self.a[i] = -(2 * p + 1) * tauTrans;       # 4  TRANSPORT
                self.asub[i] = i0 + 1;
                i +=1;
                self.a[i] = -(p + 1) * tauTrans;  # 5  TRANSPORT
                self.asub[i] = i0 + self.nx;
                i +=1;
                self.a[i] = +(p + 1) * tauTrans;       # 6  TRANSPORT
                self.asub[i] = i0 + self.nx + 1;
                i +=1;
        
                self.xa[i0 + 1] = self.xa[i0] + 6;
            
        
            # ULTIMA COLONNA
            # j = nx - 1
            i0 = self.nx - 1 + self.nx * p;
        
            self.a[i] = -p * tauTrans;            # 1  TRANSPORT
            self.asub[i] = i0 - self.nx;
            i +=1;
            self.a[i] = 1 + dt / (self.eps * self.eps) + self.chi * dt * self.sigma[self.nx - 1] / self.eps + (2 * p + 1) * tauTrans;   # 2  SIGMA
            self.asub[i] = i0;
            i +=1;
            self.a[i] = -(p + 1) * tauTrans;                  # 3  TRANSPORT
            self.asub[i] = i0 + self.nx;
            i +=1;
        
            self.xa[i0 + 1] = self.xa[i0] + 3;
        
        
        #===================================================================================================
        # K = Np - 1
        #===================================================================================================
        
        # PRIMA COLONNA
        # j = 0
        i0 = self.nx * (self.Np - 1);
        
        self.a[i] = -(self.Np - 1) * tauTrans;    # 1  TRANSPORT
        self.asub[i] = i0 - self.nx;
        i +=1;
        self.a[i] = (self.Np - 1) * tauTrans;    # 2  TRANSPORT
        self.asub[i] = i0 - self.nx + 1;
        i +=1;
        self.a[i] = 1 + dt / (self.eps * self.eps) + self.chi * dt * self.sigma[0] / self.eps + (2 * self.Np - 1) * tauTrans;         # 3  SIGMA
        self.asub[i] = i0;
        i +=1;
        self.a[i] = -(2 * self.Np - 1) * tauTrans;         # 4  TRANSPORT
        self.asub[i] = i0 + 1;
        i +=1;
        
        self.xa[i0 + 1] = self.xa[i0] + 4;
        
        # COLONNE INTERMEDIE
        # j = 1,... nx - 2
        for j in range(1, self.nx - 1): # (int j = 1; j <= self.nx - 2; j++) {
            i0 = j + self.nx * (self.Np - 1);
        
            self.a[i] = -(self.Np - 1) * tauTrans;         # 1  TRANSPORT
            self.asub[i] = i0 - self.nx;
            i +=1;
            self.a[i] = (self.Np - 1) * tauTrans;    # 2  TRANSPORT
            self.asub[i] = i0 - self.nx + 1;
            i +=1;
            self.a[i] = 1 + dt / (self.eps * self.eps) + self.chi * dt * self.sigma[j] / self.eps + (2 * self.Np - 1) * tauTrans;     # 3  SIGMA
            self.asub[i] = i0;
            i +=1;
            self.a[i] = -(2 * self.Np - 1) * tauTrans;         # 4  TRANSPORT
            self.asub[i] = i0 + 1;
            i +=1;
        
            self.xa[i0 + 1] = self.xa[i0] + 4;
        
        
        # ULTIMA COLONNA
        # j = nx - 1
        i0 = self.nx * self.Np - 1;
        
        self.a[i] = -(self.Np - 1) * tauTrans;           # 1  TRANSPORT
        self.asub[i] = i0 - self.nx;
        i +=1;
        self.a[i] = 1 + dt / (self.eps * self.eps) + self.chi * dt * self.sigma[self.nx - 1] / self.eps + (2 * self.Np - 1) * tauTrans;  # 2  SIGMA
        self.asub[i] = i0;
        i +=1;
        
        self.xa[i0 + 1] = self.xa[i0] + 2;
        
        #----------------|
        #  SECOND HALH   |
        #----------------|
        #===================================================================================================
        # K = 0
        #===================================================================================================
        # PRIMA COLONNA 
        # j = 0
        
        i0 = self.nx * self.Np;
        
        self.a[i] = -dt / (2 * self.eps * self.eps) + self.chi * dt * self.sigma[0] / (2 * self.eps);   # 1  SIGMA
        self.asub[i] = i0 - self.nx * self.Np;
        i +=1;
        self.a[i] = 1 + dt / (2 * self.eps * self.eps) - self.chi * dt * self.sigma[0] / (2 * self.eps) + tauTrans;        # 2  SIGMA
        self.asub[i] = i0;
        i +=1;
        self.a[i] = -tauTrans;                                        # 3  TRANSPORT
        self.asub[i] = i0 + self.nx;
        i +=1;
        
        self.xa[i0 + 1] = self.xa[i0] + 3;
        
        
        # COLONNE INTERMEDIE
        # j = 1, ..., nx - 2
        for j in range(1, self.nx -1): # (int j = 1; j <= self.nx - 2; j++) {
        
            i0 = j + self.nx * self.Np;
        
            self.a[i] = -dt / (2 * self.eps * self.eps) + self.chi * dt * self.sigma[j] / (2 * self.eps);         # 1  SIGMA
            self.asub[i] = i0 - self.Np * self.nx;
            i +=1;
            self.a[i] = -tauTrans;                       # 2   TRANSPORT
            self.asub[i] = i0 - 1;
            i +=1;
            self.a[i] = 1 + dt / (2 * self.eps * self.eps) - self.chi * dt * self.sigma[j] / (2 * self.eps) + tauTrans;         # 3  SIGMA  
            self.asub[i] = i0;
            i +=1;
            self.a[i] = +tauTrans;     # 4  TRANSPORT
            self.asub[i] = i0 - 1 + self.nx;
            i +=1;
            self.a[i] = -tauTrans;     # 5  TRANSPORT
            self.asub[i] = i0 + self.nx;
            i +=1;
        
        
            self.xa[i0 + 1] = self.xa[i0] + 5;
        
        
        # ULTIMA COLONNA 
        # j = nx - 1
        i0 = self.nx - 1 + self.nx * self.Np;
        
        self.a[i] = -dt / (2 * self.eps * self.eps) + self.chi * dt * self.sigma[self.nx - 1] / (2 * self.eps);            # 1  SIGMA
        self.asub[i] = i0 - self.Np * self.nx;
        i +=1;
        self.a[i] = -tauTrans;                                                # 2  TRANSPORT
        self.asub[i] = i0 - 1;
        i +=1;
        self.a[i] = 1 + dt / (2 * self.eps * self.eps) - self.chi * dt * self.sigma[self.nx - 1] / (2 * self.eps) + tauTrans;       # 3 SIGMA
        self.asub[i] = i0;
        i +=1;
        self.a[i] = +tauTrans;                                              # 4  TRANSPORT
        self.asub[i] = i0 + self.nx - 1;
        i +=1;
        self.a[i] = -tauTrans;                                              # 5  TRANSPORT
        self.asub[i] = i0 + self.nx;
        i +=1;
        
        self.xa[i0 + 1] = self.xa[i0] + 5;
        
        #===================================================================================================
        # K = 1, ..., Np - 2
        #===================================================================================================
        
        for p in range(1, self.Np - 1): # (int p = 1; p <= self.Np - 2; p++) {
        
            # PRIMA COLONNA
            # j = 0
            i0 = self.nx * p + self.nx * self.Np;
        
            self.a[i] = -p * tauTrans;    # 1  TRANSPORT
            self.asub[i] = i0 - self.nx;
            i +=1;
            self.a[i] = 1 + dt / (self.eps * self.eps) - self.chi * dt * self.sigma[0] / self.eps + (2 * p + 1) * tauTrans;         # 2  SIGMA
            self.asub[i] = i0;
            i +=1;
            self.a[i] = -(p + 1) * tauTrans;               # 3  TRANSPORT
            self.asub[i] = i0 + self.nx;
            i +=1;
        
            self.xa[i0 + 1] = self.xa[i0] + 3;
        
        
            # COLONNE INTERMEDIE
            # j = 1,... nx-2
            for j in range(1, self.nx - 1): #(int j = 1; j <= self.nx - 2; j++) {
                i0 = j + self.nx * p + self.nx * self.Np;
        
                self.a[i] = p * tauTrans;         # 1  TRANSPORT
                self.asub[i] = i0 - self.nx - 1;
                i +=1;
                self.a[i] = -p * tauTrans;    # 2  TRANSPORT
                self.asub[i] = i0 - self.nx;
                i +=1;
                self.a[i] = -(2 * p + 1) * tauTrans;         # 3  TRANSPORT
                self.asub[i] = i0 - 1;
                i +=1;
                self.a[i] = 1 + dt / (self.eps * self.eps) - self.chi * dt * self.sigma[j] / self.eps + (2 * p + 1) * tauTrans;    # 4  SIGMA
                self.asub[i] = i0;
                i +=1;
                self.a[i] = (p + 1) * tauTrans;  # 5  TRANSPORT
                self.asub[i] = i0 + self.nx - 1;
                i +=1;
                self.a[i] = -(p + 1) * tauTrans;       # 6  TRANSPORT
                self.asub[i] = i0 + self.nx ;
                i +=1;
        
                self.xa[i0 + 1] = self.xa[i0] + 6;
            
        
            # ULTIMA COLONNA
            # j = nx - 1
            i0 = self.nx - 1 + self.nx * p + self.nx * self.Np;
        
            self.a[i] = p * tauTrans;            # 1  TRANSPORT
            self.asub[i] = i0 - self.nx - 1;
            i +=1;
            self.a[i] = - p * tauTrans;            # 2  TRANSPORT
            self.asub[i] = i0 - self.nx;
            i +=1;
            self.a[i] = -(2 * p + 1) * tauTrans;            # 3  TRANSPORT
            self.asub[i] = i0 - 1;
            i +=1;
            self.a[i] = 1 + dt / (self.eps * self.eps) - self.chi * dt * self.sigma[self.nx - 1] / self.eps + (2 * p + 1) * tauTrans;   # 4  SIGMA
            self.asub[i] = i0;
            i +=1;
            self.a[i] = (p + 1) * tauTrans;                  # 5  TRANSPORT
            self.asub[i] = i0 + self.nx - 1;
            i +=1;
            self.a[i] = -(p + 1) * tauTrans;                  # 6  TRANSPORT
            self.asub[i] = i0 + self.nx;
            i +=1;
        
            self.xa[i0 + 1] = self.xa[i0] + 6;
        
        
        #===================================================================================================
        # K = Np - 1
        #===================================================================================================
        
        # PRIMA COLONNA
        # j = 0
        i0 = self.nx * (self.Np - 1) + self.nx * self.Np;
        
        self.a[i] = -(self.Np - 1) * tauTrans;    # 1  TRANSPORT
        self.asub[i] = i0 - self.nx;
        i +=1;
        self.a[i] = 1 + dt / (self.eps * self.eps) - self.chi * dt * self.sigma[0] / self.eps + (2 * self.Np - 1) * tauTrans;         # 2  SIGMA
        self.asub[i] = i0;
        i +=1;
        
        self.xa[i0 + 1] = self.xa[i0] + 2;
        
        # COLONNE INTERMEDIE
        # j = 1,... nx - 2
        for j in range(1, self.nx -1): # (int j = 1; j <= self.nx - 2; j++) {
            i0 = j + self.nx * (self.Np - 1) + self.nx * self.Np;
        
            self.a[i] = (self.Np - 1) * tauTrans;         # 1  TRANSPORT
            self.asub[i] = i0 - self.nx - 1;
            i +=1;
            self.a[i] = -(self.Np - 1) * tauTrans;    # 2  TRANSPORT
            self.asub[i] = i0 - self.nx;
            i +=1;
            self.a[i] = -(2 * self.Np - 1) * tauTrans;         # 3  TRANSPORT
            self.asub[i] = i0 - 1;
            i +=1;
            self.a[i] = 1 + dt / (self.eps * self.eps) - self.chi * dt * self.sigma[j] / self.eps + (2 * self.Np - 1) * tauTrans;     # 4  SIGMA
            self.asub[i] = i0;
            i +=1;
        
            self.xa[i0 + 1] = self.xa[i0] + 4;
        
        
        # ULTIMA COLONNA
        # j = nx - 1
        i0 = 2 * self.nx * self.Np - 1;
        
        self.a[i] = (self.Np - 1) * tauTrans;           # 1  TRANSPORT
        self.asub[i] = i0 - self.nx - 1;
        i +=1;
        self.a[i] = -(self.Np - 1) * tauTrans;           # 2  TRANSPORT
        self.asub[i] = i0 - self.nx;
        i +=1;
        self.a[i] = -(2 * self.Np - 1) * tauTrans;           # 3  TRANSPORT
        self.asub[i] = i0 - 1;
        i +=1;
        self.a[i] = 1 + dt / (self.eps * self.eps) - self.chi * dt * self.sigma[self.nx - 1] / self.eps + (2 * self.Np - 1) * tauTrans;  # 4  SIGMA
        self.asub[i] = i0;
        i +=1;
        
        self.xa[i0 + 1] = self.xa[i0] + 4;

        #A = csc_matrix((nzval, rowind, colptr), shape=(self.n, self.n))
        A = csc_matrix((self.a, self.asub, self.xa), shape=(self.n, self.n))
        
        print("Fine build di A")
        
        self.lu = splu(A)
        print("LU decomposta.\n")
        
        return 
    

    
    def kinetic_iteration(self, tn, dt):
        if tn == 0:
            self.build_Am()
            print("LU costruita con successo in kinetic step!\n")
        
        #---------------------------------------------
        # create the rhs from the laguerre's coeff
        #---------------------------------------------
        i = 0;
        for p in range(1, self.Np): #(int p = 0; p <= _Np-1; p++) {
            for k in range(1, self.nx + 2): #for (int k = 0; k <= _nx; k++) {
                self.rhs[i-1] = self.D0[p][k];
                i+=1;

        # for C-, i is already good, so fill the last half of rhs
        for p in range(1, self.Np): # (int p = 0; p <= _Np-1; p++) {
            for k in range(1, self.nx + 2): # (int k = 0; k <= _nx; k++) {
                self.rhs[i-1] = self.C0[p][k];
                i+=1;
        #---------------------------------------------
        # Solve the system IN PLACE, IF POSSIBLE
        #---------------------------------------------
        
        # splu needs CSC format for sparse matrices
        
        self.rhs = (self.lu).solve(self.rhs)
        print("Sistema risolto\n")
        #---------------------------------------------
        # recreate the D0 and C0 from rhs 
        #---------------------------------------------
        
        i = 0;
        for p in range(1, self.Np): #(int p = 0; p <= _Np-1; p++) {
            for k in range(1, self.nx + 2): #for (int k = 0; k <= _nx; k++) {
                self.D0[p][k] = self.rhs[i-1];
                i+=1;

        # for C-, i is already good, so fill the last half of rhs
        for p in range(1, self.Np): # (int p = 0; p <= _Np-1; p++) {
            for k in range(1, self.nx + 2): # (int k = 0; k <= _nx; k++) {
                self.C0[p][k] = self.rhs[i-1];
                i+=1;
        
        
        self.bc_transport(self.D0);
        self.bc_transport(self.C0);
        
        
        return
    
    def diagnostic():
        return


MySimulation = RT_1d_simulations()

print("1) Intialise the distribution function :")
MySimulation.initialise()
print("Done.\n")

print("2) Save the initial data :")
# MySimulation.diagnostic(tn, num)
print("Done.\n")


dt = 1/MySimulation.steps_p_unit
Time = dt*MySimulation.nb_units * MySimulation.steps_p_unit


while MySimulation.tn < Time:
    
    MySimulation.kinetic_iteration(MySimulation.tn, dt);
    
    MySimulation.tn += dt
    MySimulation.tdiag += dt
    
    print(MySimulation.tn)
    # Diagnostic to do !








