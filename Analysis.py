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

def color_f(x, v, f, log = False):
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
    if log:
        pcm = plt.pcolor(xx, vv, np.transpose(f),  norm=LogNorm(), cmap=cm.jet, shading='auto')
    else:
        pcm = plt.pcolor(xx, vv, np.transpose(f), cmap=cm.jet, shading='auto')
    plt.colorbar(pcm, label="f(t,x,v)")  # barra dei colori
    plt.xlabel('x')
    plt.ylabel('v')
    #plt.title("Plot of f at t = " + str(num*20) + r", $\epsilon = 0.1$")
    plt.title("t = " + str(num*period))
    plt.clim(0,0.15)
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
    plt.figure()
    ax = plt.axes(projection='3d')
    if log:
        ax.plot_wireframe(xx[1:-1,1:-1], vv[1:-1,1:-1], np.transpose(np.log10(f[1:-1,1:-1])),norm = LogNorm(), rstride=10, cstride=10, color = 'black')
        #plt.title("LogPlot of f at t = "+str(num*period) + r", $\epsilon = 0.1$")
        plt.title("t = "+str(num*period),fontsize = 15)
    else:
        ax.plot_wireframe(xx[1:-1,1:-1], vv[1:-1,1:-1], np.transpose(f[1:-1,1:-1]), rstride=10, cstride=10, color = 'black')
        plt.title("Plot of f at t = "+str(num*period) + r", $\epsilon = 0.1$")
    z_ticks = [-5, -10, -15, -20]
    z_labels = [r"$10^{-5}$", r"$10^{-10}$", r"$10^{-15}$", r"$10^{-20}$"]
    ax.set_zticks(z_ticks)  
    ax.set_zticklabels(z_labels)
    ax.tick_params(labelsize = 10)
    plt.xlabel('x',fontsize = 10)
    plt.ylabel('v',fontsize = 10)
    ax.view_init(elev=36, azim=30)
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
    title = ax.set_title(r"Plot of f at t = 0, $\epsilon = 0.1$")
    
    # funzione di aggiornamento dell’animazione
    def update(frame):
        pcm.set_array(np.transpose(f_values[frame]).ravel())
        title.set_text("Plot of f at t = " + str( round(frame*0.2, 1) ) + r", $\epsilon = 0.1$")
        return pcm, title
    
    ani = FuncAnimation(fig, update, frames=len(f_values), interval=200, blit=False)
    
    plt.show()

    return ani

def animate_rho(num):
    """

    Parameters
    ----------
    num : int
        number of frames (files .pkl to open).

    Returns
    -------
    animation
    
    """
    rho_values = []
    
    for i in range(num):
        with open(folder+'rho_'+str(i)+'.pkl', 'rb') as filef:
            (x, rho)= pickle.load(filef)
        rho_values.append(rho)
    
    fig, ax = plt.subplots()
    y = rho_values[10]
    line, = ax.semilogy(x,y)
    #line, = ax.plot(x,y)
    ax.set_xlabel("x")

    title = ax.set_title(r"Plot of $\rho_f$ at t = 0$"+ r", $\epsilon = 0.1$")
    
    # funzione di aggiornamento dell’animazione
    def update(frame):
        y = rho_values[frame]
        line.set_data(x,y)
        title.set_text(r"Plot of $\rho_f$ at t = " + str( round(frame*0.2, 1) ) + r", $\epsilon = 0.1$")
        return line, title
    
    ani = FuncAnimation(fig, update, frames=len(rho_values), interval=200, blit=False)
    
    plt.show()

    return ani



period = 0.2  # find it in Data.txt
num = 250
folder = "AnimationData_eps01/"



with open(folder+'f_'+str(num)+'.pkl', 'rb') as filef:
    (x, v, f)= pickle.load(filef)
    
with open(folder+'rho_'+str(num)+'.pkl', 'rb') as rfile:
    (x, rho) = pickle.load(rfile)

#plt.plot(x,np.sign(rho))

ani=animate_rho(251)

#ani = animate_f(251)
# to SAVE
#ani.save("animazione.gif", writer="pillow", fps=1)
# for mp4, but install ffmpeg
# ani.save("animazione.mp4", writer="ffmpeg", fps=5)

#plt.semilogy(x,rho, label = r"$\rho(x)$")

#dx = x[1]-x[0]
#y=  np.exp(-0.8/2*np.abs(x)) /( sum(np.exp(-0.8/2*np.abs(x))) *dx)  # perché /4 e non /2 !!!!
#y = np.abs(x)**0.25*np.exp(-2*np.sqrt(1.8) *np.abs(x)**0.5) /( sum(np.abs(x)**0.25*np.exp(-2*np.sqrt(1.8) *np.abs(x)**0.5) )*dx)
#plt.semilogy(x, y, label = r"$|x|^{1/4}\exp(-2\sqrt{1+\chi}|x|^{1/2})$", color = "black")
#plt.xlabel("x", fontsize=15)
#plt.title("t = "+str(num*period))
#plt.legend()


#plot_f(x, v, f,True)

#color_f(x, v, f)

#print("mass rho = "+str(massRho(x,rho)))
#print("mass f = "+str(mass(x,v,f)))





