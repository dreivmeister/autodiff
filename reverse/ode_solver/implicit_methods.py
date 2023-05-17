import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt


def newton(f,fp,x0,tol,N):
    i = 0
    fc = abs(f(x0))
    while fc.any() > tol:
        xc = x0 - (f(x0)/fp(x0))
        fc = abs(f(xc))
        x0 = xc
        i += 1
        
        if i > N:
            break
        
    return x0

# specific
def implicit_euler_step(h, f, f_y, tk, yk):
    # z is values[i+1,:]
    g  = lambda z: z - yk - h*f(tk,z)
    gp = lambda z: 1 - h*f_y(tk,z) # f_y is df/dy
    return newton(g, gp, yk, 1e-5, 10)
    
def bdf_2_step(h, f, f_y, tk, yk, yk1):
    g = lambda z: -3*z + 4*yk -yk1 + 2*h*f(tk,z)
    gp = lambda z: -3 + 2*h*f_y(tk,z)
    return newton(g,gp,yk,1e-5,10)
    
def trap_step(h, f, f_y, tk, yk):
    g  = lambda z: z - yk - h/2*(f(tk+h,z) + f(tk,yk))
    gp = lambda z: 1 - h/2*(f_y(tk+h,z) + f_y(tk,yk)) # f_y is df/dy
    return newton(g,gp,yk,1e-5,10)

# linear multistep
def bdf_2(f, t_span, y0, h=0.01):
    f_y = egrad(f,argnum=1)
    
    t0, t1 = t_span
    num_steps = int((t1-t0)/h + 1)
    tk = t0 + 2*h
    
    y = np.zeros((num_steps,len(y0)))
    y[0, :] = y0
    y[1, :] = y0
    
    i = 1
    while tk < t1:
        y[i+1,:] = bdf_2_step(h, f, f_y, tk+h, y[i,:], y[i-1,:])
        tk += h
        i += 1
    
    return y



# generics
def implicit_method_fixed_step(f, t_span, y0, h=0.01, method='implicit_euler'):
    f_y = egrad(f,argnum=1)
    
    t0, t1 = t_span
    num_steps = int((t1-t0)/h + 1)
    tk = t0 + h
    
    y = np.zeros((num_steps,len(y0)))
    y[0, :] = y0
    
    i = 0
    while tk < t1:
        y[i+1, :] = globals()[method + '_step'](h, f, f_y, tk, y[i,:])
        tk += h
        i += 1
    
    return y

    

# examples
def lotkavolterra(t, N, eps1=1.5, gamma1=1, eps2=3, gamma2=1):
    return np.array([N[0] * (eps1 - gamma1*N[1]), -N[1] * (eps2 - gamma2*N[0])])

sol = implicit_method_fixed_step(lotkavolterra, np.array([0,15]), np.array([5,10]), method='trap')


plt.plot(sol[:,0])
plt.plot(sol[:,1])
plt.show()