import numpy as np
import matplotlib.pyplot as plt


# specifics
def euler_step(h, f, tk, yk):
    return yk + h * f(tk, yk)

def heun_step(h,f,tk,yk):
    k0 = f(tk, yk)
    k1 = f(tk + h, yk + h*k0)
    return yk + h/2*(k0+k1)

def rk4_step(h, f, tk, yk):
    k1 = f(tk, yk)
    k2 = f(tk + h/2, yk + h/2 * k1)
    k3 = f(tk + h/2, yk + h/2 * k2)
    k4 = f(tk + h, yk + h * k3)
    return yk + h * 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

def adams_bashforth_2_step(h, f, tk, yk, tk_1, yk_1):
    return yk + 3/2*h*f(tk,yk) - 1/2*h*f(tk_1,yk_1)

def trapezoidal_step(h, f, tk, yk):
    return yk + h/2*(f(tk, yk) + f(tk+h,yk + h*f(tk, yk)))

# linear multistep
def adams_bashforth_2(f, t_span, y0, h=0.01):
    t0, t1 = t_span
    num_steps = int((t1-t0)/h + 1)
    tk = t0 + h
    
    y = np.zeros((num_steps,len(y0)))
    y[0, :] = y0
    y[1, :] = y0
    
    i = 1
    while tk < t1:
        y[i+1] = adams_bashforth_2_step(h, f, tk, y[i], tk-h, y[i-1])
        tk += h
        i += 1
    return y

def heun(f, t_span, y0, h=0.01):
    # simple predictor corrector
    t0, t1 = t_span
    y = [y0]
    tk = t0 + h
    
    while tk < t1:
        # euler approx predictor
        y1e = y[-1] + h*f(tk, y[-1])
        # trapezoidal corrector
        y1 = y[-1] + h/2*(f(tk, y[-1]) + f(tk+h,y1e))
        
        y.append(y1)
        tk += h
    return np.asarray(y)




# generics
def explicit_method_fixed_step(f, t_span, y0, h=0.01, method='euler'):
    
    # edge case
    if method == 'adams_bashforth_2':
        return adams_bashforth_2(f, t_span, y0, h)
    
    t0, t1 = t_span
    num_steps = int((t1-t0)/h + 1)
    tk = t0 + h
    
    y = np.zeros((num_steps,len(y0)))
    y[0, :] = y0
    
    i = 0
    while tk < t1:
        # doesnt work for adams bashforth
        y[i+1] = globals()[method + '_step'](h, f, tk, y[i])
        tk += h
        i += 1
        
        h = min(h, t1-tk)
    return y

def explicit_method_adaptive_step(f, t_span, y0, h=0.02, hmin=0.01, hmax=2, eps=1e-3, method1='euler', method2='heun'):
    # doesnt work for adams bashforth
    t0, t1 = t_span
    y = [y0]
    tk = t0 + h
    order = 2
    
    while tk < t1:
        # compute preds        
        yt = globals()[method1 + '_step'](h, f, tk, y[-1])
        zt = globals()[method2 + '_step'](h, f, tk, y[-1])
        
        
        # comp. error ratio
        err = eps/(np.linalg.norm(zt-yt)+1e-13)
        
        
        
        # update step size
        hh = h * min(hmax,max(hmin,0.9 * err ** (1/order)))
        
        if err < 1:
            h = hh
            continue
        else:
            h = hh
            tk = tk + h
            y.append(yt)
            
        # last step
        h = min(h, t1-tk)
    return np.asarray(y)
            



# examples
# def oned(t,u,c=2.3):
#     #Heterogeneous first-order linear constant coefficient 
#     #ordinary differential equation
#     return -2*u + np.exp(-2*(t-6)**2)


# # example
# def lotkavolterra(t, N, eps1=1.5, gamma1=1, eps2=3, gamma2=1):
#     return np.array([N[0] * (eps1 - gamma1*N[1]), -N[1] * (eps2 - gamma2*N[0])])

# sol = explicit_method_adaptive_step(lotkavolterra, np.array([0,15]), np.array([5,10]),method1='euler',method2='rk4')
# print(sol.shape)

# plt.plot(sol[:,0])
# plt.plot(sol[:,1])
# plt.show()



#TODO: make three test problems one eq, two eq (lotkavolterra), three eq (lorenz)
# maybe also a stiff problem
#multistep, explicit, implicit, predictor corrector, bdf, leapfrog, shooting





"""
#https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/14IVPs/rkf45/complete.html
def rk_fehlberg(f, t_span, y0, h=0.1, eps_step=0.001):
    t0, tend = t_span
    hmin = 0.025
    hmax = 1.6
    t = [t0]
    y = [y0]
    
    k = 0
    while t[-1] < tend:
        K1 = f(t[-1],y[-1])
        K2 = f(t[-1] + (1/4)*h, y[-1] + (1/4)*h*K1)
        K3 = f(t[-1] + (3/8)*h, y[-1] + (3/8)*h*(1/4*K1 + 3/4*K2))
        K4 = f(t[-1] + (12/13)*h, y[-1] + (12/13)*h*(161/169*K1 - 600/169*K2 + 608/169*K3))
        K5 = f(t[-1] + h, y[-1] + h*(8341/4104*K1 - 32832/4104*K2 + 29440/4104*K3 - 845/4104*K4))
        K6 = f(t[-1] + (1/2)*h, y[-1] + (1/2)*h*(-6080/10260*K1 + 41040/10260*K2 - 28352/10260*K3 + 9295/10260*K4 - 5643/10260*K5))
        
        yt = y[-1] + h*(2375*K1 + 11264*K3 + 10985*K4 - 4104*K5)/20520
        zt = y[-1] + h*(33440*K1 + 146432*K3 + 142805*K4 - 50787*K5 + 10260*K6)/282150
        
        s = (eps_step*h)/(2*abs(yt - zt)) ** (1/4)
        print(s)
        if s < 0.75:
            # step size too big
            # repeat step
            h = max(h/2, hmin)
            continue
        
        t.append(t[-1]+h)
        y.append(yt)
        k += 1
        
        
        if s > 1.5:
            # step size too small
            h = min(2*h, hmax)
    
    return np.asarray(y)
"""