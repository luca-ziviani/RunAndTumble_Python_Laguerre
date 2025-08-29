import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # for colormaps
import matplotlib.colors as colors
import pickle
import os

script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

num = 1

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
    plt.title("Plot of f")
    ax.set_zlim(0, np.max(f)) 
    plt.show()
    return ax

with open('f_'+str(num)+'.pkl', 'rb') as filef:
    (x, v, f)= pickle.load(filef)
    
with open('rho_'+str(num)+'.pkl', 'rb') as rfile:
    (x, rho) = pickle.load(rfile)

#plt.plot(x,rho)

plot_f(x, v, f)






