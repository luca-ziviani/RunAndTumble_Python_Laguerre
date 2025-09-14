import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # for colormaps
import matplotlib.colors as colors
import pickle
import os
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)



def mass(x,v,f):
    dx = x[1]-x[0]
    dv = v[1]-v[0]
    
    return sum(sum(f))*dx*dv

def massRho(x,rho):
    dx = x[1]-x[0]
    
    return sum(rho)*dx

def color_f(x, v, f):
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
    Make a density plot of f

    """
    [xx,vv] = np.meshgrid(x,v)  
    plt.figure()
    pcm = plt.pcolor(xx, vv, np.transpose(f),  norm=LogNorm(), cmap=cm.jet, shading='auto')
    plt.colorbar(pcm, label="f(x,v)")  # barra dei colori
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title("Plot of f at t = " + str(num))
    plt.show()
    
    return
    

def plot_f(x, v, f,log=False):
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
    if log:
        ax.plot_wireframe(xx[1:-1,1:-1], vv[1:-1,1:-1], np.transpose(np.log10(f[1:-1,1:-1])),norm = LogNorm(), rstride=10, cstride=10, color = 'black')
        plt.title("LogPlot of f at i = "+str(num*dt))
    else:
        ax.plot_wireframe(xx[1:-1,1:-1], vv[1:-1,1:-1], np.transpose(f[1:-1,1:-1]), rstride=10, cstride=10, color = 'black')
        plt.title("Plot of f at i = "+str(num*dt))
    plt.xlabel('x')
    plt.ylabel('v')
    
    #ax.set_zlim(0, np.max(f)) 
    plt.show()
    return ax



def animate_f(num):
    """

    Parameters
    ----------
    num : int
        number of frames (files .pkl to open).

    Returns
    -------
    animation
    
    """
    f_values = []
    
    for i in range(num):
        with open(folder+'f_'+str(i)+'.pkl', 'rb') as filef:
            (x, v, f)= pickle.load(filef)
        f_values.append(f)
    
    xx, vv = np.meshgrid(x, v)
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(xx, vv, np.transpose(f_values[0]),  cmap=cm.jet, shading="auto")
    fig.colorbar(pcm, ax=ax, label="f(x,v)")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    title = ax.set_title("Plot of f at t = 0")
    
    # funzione di aggiornamento dell’animazione
    def update(frame):
        pcm.set_array(np.transpose(f_values[frame]).ravel())
        title.set_text("Plot of f at t = " + str( round(frame*0.2, 1) ))
        return pcm, title
    
    ani = FuncAnimation(fig, update, frames=len(f_values), interval=200, blit=False)
    
    plt.show()

    return ani

period = 20 # find it in Data.txt
num = 20
folder = "ToSteadyState_eps05/"

with open(folder+'f_'+str(num)+'.pkl', 'rb') as filef:
    (x, v, f)= pickle.load(filef)
    
with open(folder+'rho_'+str(num)+'.pkl', 'rb') as rfile:
    (x, rho) = pickle.load(rfile)



#ani = animate_f(251)
# to SAVE
#ani.save("animazione.gif", writer="pillow", fps=1)
# for mp4, but install ffmpeg
# ani.save("animazione.mp4", writer="ffmpeg", fps=5)

#plt.semilogy(x,rho, label = "rho")

#dx = x[1]-x[0]
#y= np.exp(-0.8/4*np.abs(x)) /( sum(np.exp(-0.8/4*np.abs(x))) *dx)  # perché /4 e non /2 !!!!
#y = np.exp(-2*np.sqrt(1.8) *np.abs(x)**0.5) /( sum(np.exp(-2*np.sqrt(1.8) *np.abs(x)**0.5) )*dx)
#plt.semilogy(x, y, label = "steady state")
#plt.legend()


plot_f(x, v, f,True)

#color_f(x, v, f)

#print("mass rho = "+str(massRho(x,rho)))
#print("mass f = "+str(mass(x,v,f)))





