import matplotlib.pyplot as plt
from pylab import cm
import autograd.numpy as np
from autograd import grad, elementwise_grad as egrad

def f(x):
    return 3*x**2 + 6*x + 2

def g(x,y):
    return 3*x**2 + 6*x*y + 2*y


def plot2d(f, t0=-10, t1=10, h=0.02, style='k'):
    t = np.arange(t0,t1,h)

    fig = plt.figure()
    plt.plot(t, f(t), style)
    
    return fig
    
pl = plot2d(f)
    

def plot3d(f, x0=-10, x1=10, y0=-10, y1=10, h=0.02):
    x1, x2 = np.meshgrid(np.arange(x0,x1, h), 
                         np.arange(y0,y1, h))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.contour3D(x1,x2,f(x1,x2),100,cmap=cm.jet)
    ax.view_init(60,35)
    
    
    return fig

pl = plot3d(g)

