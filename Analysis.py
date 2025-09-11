import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # for colormaps
import matplotlib.colors as colors
import pickle
import os

script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

num = 10

def mass(x,v,f):
    dx = x[1]-x[0]
    dv = v[1]-v[0]
    
    return sum(sum(f))*dx*dv

def massRho(x,rho):
    dx = x[1]-x[0]
    
    return sum(rho)*dx

def plot_f(x, v, f):
    """

    Parameters
    ----------
    x : Array of float64 of lenght Nx
        Example:    x = np.linspace(-3, 3, Nx)
    
    v : Array of float64 of lenght Nv
        Example:    v = np.linspace(-3, 3, Nv)

    f : Array of float64 of size [Nv,Mx]
        Example:    [xx,vv] = np.meshgrid(x,v)
                    f= np.exp(-xx**2 - vv**2).

    Returns
    -------
    Make a plot of f

    """
    [xx,vv] = np.meshgrid(x,v)
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, vv, np.transpose( f), cmap=cm.jet)
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title("Plot of f at i = "+str(num))
    ax.set_zlim(0, np.max(f)) 
    plt.show()
    return ax

with open('f_'+str(num)+'.pkl', 'rb') as filef:
    (x, v, f)= pickle.load(filef)
    
with open('rho_'+str(num)+'.pkl', 'rb') as rfile:
    (x, rho) = pickle.load(rfile)

#plt.semilogy(x,rho)

plot_f(x, v, f)

print("mass rho = "+str(massRho(x,rho)))
print("mass f = "+str(mass(x,v,f)))





