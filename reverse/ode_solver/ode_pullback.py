import autograd.numpy as np
from autograd import elementwise_grad as egrad
from explicit_methods import explicit_method_fixed_step

# provides functionality to backpropagate cotangent information 
# from the output of an ODE integration to the input of that ODE also called Final Time Integration
# this is used for example in Neural ODEs to backpropagate gradient information dL/dtheta in order to do gradient based optimization
# https://ceyron.github.io/implicit-autodiff-table/

t0 = 0
t1 = 3

def oned(t,u):
    #Heterogeneous first-order linear constant coefficient 
    #ordinary differential equation
    return -2*u + np.exp(-2*(t-6)**2)

f_t = egrad(oned,argnum=0) # df/dtheta
f_u = egrad(oned,argnum=1) # df/du


def final_time_integration(f, t_span, u0, h=0.1):
    # integrate f = du/dt but only return the last value
    return explicit_method_fixed_step(f, t_span, u0, h)[-1]



ut = final_time_integration(oned, np.array([t0,t1]), np.array([3]))
print(ut)


def forward_euler_reverse_time(f, t_span, u0, h=0.1):
    # t_span = (t1, t0), t1 > t0
    t1, t0 = t_span
    num_steps = int((t1-t0)/h + 1)
    tk = t1 - h
    
    y = np.zeros((num_steps,len(u0)))
    y[0, :] = u0
    
    i = 0
    while tk > t0:
        # doesnt work for adams bashforth
        y[i+1] = y[i] + h * f(tk, y[i])
        tk -= h
        i += 1
        
        h = min(h, tk-t0)
    print(tk)
    # y contains the values from y[0] = yT to y[-1] = y0
    return y

def backward_f(t,u):
    return -(f_u(t,u))*u


# u cotangent as initial cond
lam = forward_euler_reverse_time(backward_f, np.array([t1,t0]), np.array([1.]))



t = np.linspace(t1,t0,int((t1-t0)/0.1)+1)

theta_dot = np.dot(lam,f_u(t,lam).T)
print(theta_dot.shape)
