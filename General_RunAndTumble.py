# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:43:15 2023

Author: ZIVIANI Luca
Institution: Universite Paris Dauphine PSL

    * Numerical simulation of the Run and Tumble equation with
      general tumbling kernel

    * Choose the following parameters:
        - gamma     |   Tails of the local equilibrium, gamma > 0
        - psi       |   Smooth version of sign
        - Chi       |   Sensitivity, 0 < Chi < 1
        - Tmax      |   Final time
        - Lmax      |   Size of the space interval [-Lmax, Lmax]
        - Vmax      |   Size of the velocity interval [-Vmax, Vmax]
        - dx        |   Grid size in x
        - dv        |   Grid size in v

    * Return:
        Solution of the run and tumble equation at time Tmax

Reference paper: https://arxiv.org/pdf/2505.08061

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

def LocalEquilibrium(V, dv, gamma):
    """
    Build the renormalized local equilibrium M of parameter gamma on the interval V
    
    M = exp ( - |v|^gamma / gamma )

    """    
    Y = np.exp(-(np.abs(V)**gamma) / gamma)
    Z = np.sum(Y * dv)
    Y = np.exp(-(np.abs(V)**gamma) / gamma) / Z
    return Y

def psi(m):
    """
    Function psi : (smooth) version of sign.
    """
    #return (2 * np.arctan(m) / np.pi)**3
    return np.sign(m)

def lmbda(gradM, V, Nx, Nv, Chi):
    """
    Function Lambda = 1 + chi psi( v.grad(M))
    """
    XV = gradM * V[:, np.newaxis]
    l = np.ones((Nv, Nx)) - Chi * psi(XV)
    return l


def normalized(F, dx, dv):
    """
    Normalization of F
    """
    Z = np.sum(F) * dx * dv
    f = F / Z
    return f


def RunAndTumble(Chi, gamma, Tmax, Lmax, Vmax, dx, dv, SteadyState):
    """
    Upwind scheme for the Run and Tumble model
    """
    # GRID AND INITIALIZATION

    dt = dx / Vmax                          # t-step calculated from the CFL
    nt = round(Tmax / dt)                   # number of iterations in time
    Nx = round(2 * Lmax / dx + 1)           # number of x-points
    Nv = round(2 * Vmax / dv + 1)           # number of v-points
    F = np.zeros((Nv, Nx))                  # initialization on [-Lmax,Lmax]x[-Vmax,Vmax]
    FF = np.zeros((Nv, Nx))                 # initialization on [-Lmax,Lmax]x[-Vmax,Vmax]
    RhoG = np.zeros(Nx)                     # initialization of the density of F on [-Lmax,Lmax]

    t = np.arange(dt, Tmax + dt, dt)        # time points
    X = np.arange(-Lmax, Lmax + dx, dx)     # coordinates of x-points
    V = np.arange(-Vmax, Vmax + dv, dv)     # coordinates of v-points

    decay = np.zeros(nt)

    # DEFINITION OF THE MAIN FUNCTIONS

    gradM = -X / ((1 + X**2)**0.5)

    l=lmbda(gradM,V,Nx,Nv,Chi)

    G= LocalEquilibrium(V,dv,gamma);

    # INITIALISATION

    F = np.maximum(np.zeros((1, Nx)), 4 - X**2) * np.maximum(np.zeros(( Nv,1)), 4 - V[:, np.newaxis]**2)

    F=normalized(F,dx,dv)

    # MAIN ALGORITHM

    # Tools for Upwind
    A = np.where(V > 0)[0]
    C = np.where(V <= 0)[0]

    for j in range(nt):
        FF = np.copy(F)

        # TRANSPORT Step: F->FF
        FF[A, 1:] -= dt / dx * (FF[A, 1:] - FF[A, :-1])* V[A, np.newaxis]
        FF[C, :-1] += dt / dx * (FF[C, :-1] - FF[C, 1:]) * V[C, np.newaxis]

        # COLLISION Step: FF->F
        mass = dv * np.sum(l * FF, axis=0)  # Integral of lambda*f with respect to v
        RHS = mass * (G[:, np.newaxis]) - l * FF

        I = np.arange(1, Nx)
        F[:, I] = FF[:, I] + dt * RHS[:, I]

        I = np.arange(Nx - 1)
        F[:, I] = FF[:, I] + dt * RHS[:, I]

        decay[j] = np.sum(np.abs(F - SteadyState) ** 2 / SteadyState) * dx * dv


    RhoG = dv * np.sum(F, axis=0)  # Integral of f with respect to v, i.e. rho
    mult = np.ones(Nx) * (V[:, np.newaxis] ** 2) # v^2 - to compute second moment of F
    FP = F * mult
    PG = dv * np.sum(FP, axis=0) # Second moment of F

    return F, RhoG,PG,decay,t,X,V


Chi=0.8
gamma=1
Tmax=20
Lmax=60
Vmax=30
dx=0.25
dv=0.25
Nx = round(2*Lmax/dx+1)
Nv = round(2*Vmax/dv+1)

# Initialise a Steady state (not used the first iterations)
SteadyState = np.ones((Nv, Nx))

X = np.arange(-Lmax, Lmax + dx, dx)     # coordinates of x-points
V = np.arange(-Vmax, Vmax + dv, dv)     # coordinates of v-points

# Computation of the Steady state
(F, RhoG,PG,decay,t,X,V)=RunAndTumble(Chi, gamma, Tmax, Lmax, Vmax, dx, dv, SteadyState)

# Recompute the solution but with the distance from the Steady state.
(F, RhoG,PG,decay,t,X,V)=RunAndTumble(Chi, gamma, Tmax, Lmax, Vmax, dx, dv, F)

# Save of the data
with open('RT_Tmax'+str(Tmax)+'_Chi'+str(Chi)+'_gamma'+str(gamma)+'_Lmax'+str(Lmax)+'_Vmax'+str(Vmax)+'.pkl', 'wb') as file:
    pickle.dump((F,RhoG,PG,decay,t,X,V),file)


