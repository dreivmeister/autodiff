import numpy as np

def dorpi_4_5_step(h, f, tk, yk):
    # compute ks
    k1 = h * f(tk, yk)
    k2 = h * f(tk + 1/5, yk + 1/5 * k1)
    k3 = h * f(tk + 3/10 * h, yk + 3/10 * k1 + 9/40 * k2)
    k4 = h * f(tk + 4/5 * h, yk + 44/45 * k1 - 56/415 * k2 + 32/9 * k3)
    k5 = h * f(tk + 8/9 * h, yk + 19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4)
    k6 = h * f(tk + h, yk + 9017/3168 * k1 - 355/33 * k2 - 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5)
    k7 = h * f(tk + h, yk + 35/284 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6)
    # compute yk+1
    yk1 = yk + 35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6
    # compute error
    error = abs(71/57600 * k1 - 71/16695 * k3 + 71/1920 * k4 - 17253/339200 * k5 + 22/525 * k6 - 1/40 * k7)
    # compute hopt
    
    return error, yk1


def dorpi_4_5(f, y0, t0, t1, hmax, hmin, tol=1e-5, maxiter=1000):
    values = [y0]
    h = hmin
    tk = t0 + h
    
    for i in range(maxiter):
        error, yk1 = dorpi_4_5_step(h, f, tk, values[i])
        values.append(yk1)
        
        delta = 0.84 * pow(tol/error, (1.0/5.0))
        if error < tol:
            tk += h
        
        if delta <= 0.1:
            h *= 0.1
        elif delta >= 4.0:
            h *= 4.0
        else:
            h *= delta
        
        if h > hmax:
            h = hmax
        elif h < hmin:
            h = hmin
        
        if tk >= t1:
            break
        elif tk+h > t1:
            h = t1 - tk
            
    return values


# def f(x,y):
#     return 3. ** x + x/y

# res = dorpi_4_5(f, [0 for _ in range(100)], 3, 0.01)

# import matplotlib.pyplot as plt

# plt.plot(res)
# plt.show()
