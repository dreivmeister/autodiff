import numpy as np
import matplotlib.pyplot as plt
def subval(x, i, v):
    x_ = list(x)
    x_[i] = v
    return tuple(x_)

def make_vjp(fun, x):
    end_valnode = fun(x)
    
    if end_valnode is None:
        def vjp(g):
            return np.zeros_like(x)
    else:
        def vjp(g):
            end_valnode.backward(g)
            return x.grad
    return vjp, end_valnode


def grad(fun, argnum=0):
    
    def gradfun(*args, **kwargs):
        unary_fun = lambda x: fun(*subval(args, argnum, x), **kwargs)
        
        vjp, ans = make_vjp(unary_fun, args[argnum])
        return vjp(np.ones_like(ans.data))
    
    return gradfun

def newtons_method(f, curr_val, max_iter=100, tol=1e-8):
    # x_k+1 = x_k - (f(x_k)/f'(x_k))    
    # for now single variable
    f_prime = grad(f)
    
    
    for _ in range(max_iter):
        
        new_val = curr_val - (f(curr_val)/(f_prime(curr_val) + 1e-10))
        if abs(new_val.data - curr_val.data) < tol:
            break
        curr_val = new_val
    return curr_val




# y = function(a, b)
# y.backward()


# print(y)
# print(a)
# print(b)
# a.grad = 0
# b.grad = 0


# df_dx = grad(function,argnum=0)
# df_dy = grad(function,argnum=1)
# print(df_dx(a,b))
# a.grad = 0
# b.grad = 0
# print(df_dy(a,b))


# def tanh(x, theta):
#     y = (-2.0 * x).exp() + theta
#     return (1.0 - y) / (1.0 + y*theta)

# x = Tensor(1.0)
# theta = Tensor(4.0)

# y = tanh(x, theta)

# deltanh_delx = grad(tanh,argnum=0)
# deltanh_deltheta = grad(tanh,argnum=1)

# x_cotangent = 1.0
# print(-x_cotangent*(deltanh_deltheta(x,theta)/deltanh_delx(x,theta)))

# y.backward()

# deltanh_delx = x.grad
# deltanh_deltheta = theta.grad

# x_cotangent = 1.0

# theta_cotangent = -x_cotangent*(deltanh_deltheta/deltanh_delx)
# print(theta_cotangent)





# only works for the jacobian, so first derivative
# plt.plot(
#     x.data, tanh(x).data,
#     x.data, grad(tanh)(x)
# )
# plt.show()