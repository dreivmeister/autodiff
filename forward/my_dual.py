import matplotlib.pyplot as plt
import numpy as np
from time import time

def arg_checker(other, name):
    if isinstance(other, Dual):
        return other
    elif isinstance(other, (int, float)):
        # constant
        return Dual(other, 0.)
    elif isinstance(other, (np.ndarray, list)):
        return Dual(other, np.zeros_like(other))
    else:
        raise AttributeError(f'wrong args {name}')



class Dual:
    def __init__(self, real, dual=None) -> None:
        self.real = real
        self.dual = dual
        
        # real can be int or float
        if isinstance(real, (int,float,np.int32,np.float64)):
            self.real = real
            if isinstance(dual, (int,float,np.int32,np.float64)):
                self.dual = dual
            elif isinstance(dual, (np.ndarray, list)):
                self.dual = np.asarray(dual)
            elif dual is None:
                self.dual = 1.0
        # real can be nparray (n-dim) or list
        elif isinstance(real, (np.ndarray, list)):
            self.real = np.asarray(real)
            if isinstance(dual, (np.ndarray, list)):
                self.dual = np.asarray(dual)
            elif dual is None:
                self.dual = np.ones_like(self.real)
        else:
            raise AttributeError('wrong init')
    
    def __repr__(self):
        if isinstance(self.real, np.ndarray) and self.real.ndim > 1:
            return f"{self.__class__.__name__}(\n{self.real},\n{self.dual})"
        return f"{self.__class__.__name__}({self.real},{self.dual})"
    
    @staticmethod
    def from_array(X):
        X = np.array(X)
        # if np.ndim(X) != 1:
        #     raise AttributeError('wrong dim from_array')
        if len(X) == 1:
            return Dual(X[0])
        
        I = np.identity(len(X))
        return iter(Dual(x, I[i]) for i,x in enumerate(X))
    
    @staticmethod
    def is_constant(x):
        if isinstance(x.dual, np.ndarray):
            return True if np.allclose(x.dual,0) else False
        elif isinstance(x.dual, (int,float,np.int32,np.float32)):
            return True if np.isclose(x.dual,0) else False
        else:
            raise AttributeError(f'wrong type is_constant {type(x.dual)}')
            
            
    
    # elementwise operations
    def __add__(self, other):
        other = arg_checker(other, name='add')
        
        if isinstance(other, Dual):
            return Dual(self.real+other.real,
                        self.dual+other.dual)
        else:
            raise AttributeError('wrong type add')
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = arg_checker(other, name='mul')
        
        if isinstance(other, Dual):
            # product rule
            return Dual(self.real*other.real,
                        self.real*other.dual+self.dual*other.real)
        else:
            raise AttributeError('wrong type mul')
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        other = arg_checker(other, name='sub')
        
        if isinstance(other, Dual):
            # product rule
            return Dual(self.real-other.real,self.dual-other.dual)
        else:
            raise AttributeError('wrong type sub')
    def __rsub__(self, other):
        return self - other
    
    def __truediv__(self, other):
        other = arg_checker(other, name='truediv')
        
        if isinstance(other, Dual):
            # quotient rule
            return Dual(self.real/other.real,
                        (other.real * self.dual - self.real * other.dual) / (other.real**2))
        else:
            raise AttributeError('wrong type truediv')
    def __rtruediv__(self, other):
        return self / other
    
    def __pow__(self, other):
        other = arg_checker(other, name='pow')
        
        if isinstance(other, Dual):
            return Dual(self.real ** other.real,
                        other.real * self.real ** (other.real-1) * self.dual)
        else:
            raise AttributeError('wrong type pow')

    def __neg__(self):
        return Dual(-self.real, -self.dual)


def sin(x):
    x = arg_checker(x, name='sin')
    
    if isinstance(x, Dual):
        return Dual(np.sin(x.real),
                    np.cos(x.real)*x.dual)
    else:
        raise AttributeError('wrong type sin')
    
def cos(x):
    x = arg_checker(x, name='sin')
    
    if isinstance(x, Dual):
        return Dual(np.cos(x.real),
                    -np.sin(x.real)*x.dual)
    else:
        raise AttributeError('wrong type cos')
    
def exp(x):
    x = arg_checker(x, name='exp')
    
    
    if isinstance(x, Dual):
        return Dual(np.exp(x.real),
                    np.exp(x.real)*x.dual)
    else:
        raise AttributeError('wrong type exp')
    
# maybe improve later
def log(x, base=np.e):
    if isinstance(x, Dual):
        pass
    elif isinstance(x, (int, float)):
        if x <= 0:
            raise Exception('cant do log on 0 or neg')
        # constant
        x = Dual(x, 0.)
    elif isinstance(x, (np.ndarray, list)):
        x = np.array(x)
        if x.any() <= 0:
            raise Exception('cant do log on 0 or neg')
        x = Dual(x, np.zeros_like(x))
    else:
        raise AttributeError('wrong args log')
    
    if isinstance(x, Dual):
        return Dual(np.log(x.real) / np.log(base),
                    x.dual / (x.real * np.log(base)))
    else:
        raise AttributeError('wrong type log')

def sqrt(x):
    if isinstance(x, Dual):
        pass
    elif isinstance(x, (int, float)):
        if x < 0:
            raise Exception('cant do sqrt on neg')
        # constant
        x = Dual(x, 0.)
    elif isinstance(x, (np.ndarray, list)):
        x = np.array(x)
        if x.any() < 0:
            raise Exception('cant do sqrt on neg')
        x = Dual(x, np.zeros_like(x))
    else:
        raise AttributeError('wrong args sqrt')
    
    if isinstance(x, Dual):
        v = np.sqrt(x.real)
        return Dual(v, (0.5 / v) * x.dual)
    else:
        raise AttributeError('wrong type sqrt')
    
def transpose(A):
    # matrix is 2D
    A = arg_checker(A, name='transpose')
    
    if A.real.ndim != 2:
        raise AttributeError('wrong ndims in transpose')
    
    if isinstance(A, Dual):
        return Dual(
            A.real.T,
            A.dual.T
        )
    else:
        raise AttributeError('wrong type transpose')
    
def inner_product(a, b):
    # dot product of two 1D arrays
    # inner product
    a = arg_checker(a, name='inner_prod')
    b = arg_checker(b, name='inner_prod')
    
    if a.real.ndim != 1 or b.real.ndim != 1:
        raise AttributeError('wrong ndims in inner_prod')
    if len(a.real) != len(b.real):
        raise AttributeError('wrong nelems in inner_prod')
    
    if isinstance(a, Dual) and isinstance(b, Dual):
        return Dual(
            np.dot(a.real,b.real),
            np.dot(a.dual,b.real)+np.dot(a.real,b.dual))
        #a.dual*np.sum(b.real)+b.dual*np.sum(a.real)
        # dont know about the tangent
    else:
        raise AttributeError('wrong type inner_prod')
    
def outer_product(a, b):
    # dot product of two 1D arrays
    # inner product
    a = arg_checker(a, name='outer_prod')
    b = arg_checker(b, name='outer_prod')
    
    if a.real.ndim != 1 or b.real.ndim != 1:
        raise AttributeError('wrong ndims in outer_prod')
    if len(a.real) != len(b.real):
        raise AttributeError('wrong nelems in outer_prod')
    
    if isinstance(a, Dual) and isinstance(b, Dual):
        return Dual(
            np.outer(a.real,b.real),
            np.outer(a.dual,b.real)+np.outer(a.real,b.dual))
        #a.dual*np.sum(b.real)+b.dual*np.sum(a.real)
        # dont know about the tangent
    else:
        raise AttributeError('wrong type outer_prod')    

    
def matrix_vector_product(A, x):
    # A*x as matrix vector product A (2D), x (1D)
    A = arg_checker(A, name='mvp')
    x = arg_checker(x, name='mvp')
    
    if A.real.ndim != 2 or x.real.ndim != 1:
        raise AttributeError('wrong ndims in mvp')
    if A.real.shape[1] != x.real.shape[0]:
        raise AttributeError('wrong nelems in mvp')
    
    if isinstance(A, Dual) and isinstance(x, Dual):
        return Dual(
            np.dot(A.real,x.real),
            np.dot(A.dual,x.real)+np.dot(A.real,x.dual))
    else:
        raise AttributeError('wrong type mvp')

def matrix_matrix_product(A, B):
    # A*B as matrix matrix product A (2D), B (2D)
    A = arg_checker(A, name='mmp')
    B = arg_checker(B, name='mmp')
    
    if A.real.ndim != 2 or B.real.ndim != 2:
        raise AttributeError('wrong ndims in mmp')
    if A.real.shape[1] != B.real.shape[0]:
        raise AttributeError('wrong nelems in mmp')
    
    if isinstance(A, Dual) and isinstance(B, Dual):
        return Dual(
            np.matmul(A.real,B.real),
            np.matmul(A.dual,B.real)+np.matmul(A.real,B.dual))
    else:
        raise AttributeError('wrong type mmp')
    
def scalar_root_finding(g,theta):
    # scalar_root_finding is f parameterized by g and theta
    # g is the function which should be solved
    # g(x, theta) = 0
    # x is the solution
    first_guess = 1
    x, _, dg_dx = newtons_method(lambda u: g(u,theta),first_guess)
    _,dg_dtheta = val_and_grad(lambda u: g(x,u),theta)
    
    
    print(x)
    print(dg_dx)
    print(dg_dtheta)
    
    return Dual(
        x,
        -(1./dg_dx)*dg_dtheta.real
    )
    
def linear_system(A, b):
    A_inv = np.linalg.inv(A.real)
    x = np.dot(A_inv,b.real)
    return Dual(
        x,
        np.dot(A_inv,b.dual - np.dot(A.dual,x))
    )
    

def jacobian(results):
    jac_matrix = []
    for r in results:
        if isinstance(r, Dual):
            jac_matrix.append(r.dual)
    return np.vstack(jac_matrix)

def val_and_grad(function, eval_point):
    # eval_point - list of scalar(s) or scalar
    # like: [2,4,6] or 2 or [2.3,234.2]
    
    if isinstance(eval_point, (int,float)):
        v = Dual.from_array([eval_point])
    elif isinstance(eval_point, (list,np.ndarray)):
        v = Dual.from_array(eval_point)
    
    # one input
    if isinstance(v, Dual):
        function_out = function(v)
    # mult input
    else:
        function_out = function(*v)
        
    # one output
    if isinstance(function_out, Dual):
        return function_out.real, function_out.dual
    
    # mult output
    # extract val
    out_vec = []
    for out in function_out:
        out_vec.append(out.real)
            
    return np.array(out_vec), jacobian(function_out)

def gradient_checker(f, x, eps=1e-8):
    if isinstance(x, (int, float)):
        pass
    elif isinstance(x, (np.ndarray, list)):
        x = np.array(x)
    else:
        raise AttributeError('wrong type gradcheck')
    return (f(x+eps)-f(x))/eps

def evalf(x):
    return x.real if Dual.is_constant(x) else x

    
def f(x,y):
    return (
        (
            4*(x ** 2) + 3 - y - 12) / (2*x),
            x * y,
            14 + x ** 2 * 4
        )

def g(x, theta):
    return x**2 - sin(theta)

# Vector function mapping 3 inputs to 3 outputs.
def h(x, y): 
    return (
        x**2 + y**2 - 20,
        x - y + 2
            )

def newtons_method(f, curr_val, max_iter=10000, tol=1e-8):
    # x_k+1 = x_k - (f(x_k)/f'(x_k))    
    # for now single variable
    
    for _ in range(max_iter):
        func_val, der_val = val_and_grad(f, curr_val)
        
        new_val = curr_val - (func_val/(der_val + 1e-10))
        if abs(new_val - curr_val).all() < tol:
            break
        curr_val = new_val
    return curr_val, *val_and_grad(f, curr_val)


def newtons_method_kk(f, curr_val, max_iter=10000, tol=1e-8):
    
    for _ in range(max_iter):
        func_val, jacobian = val_and_grad(f, curr_val)
        xk1_xk = np.linalg.solve(jacobian,-func_val)
        next_val = xk1_xk + curr_val
        
        if abs(next_val - curr_val).all() < tol:
            break        
        curr_val = next_val
    
    return curr_val, *val_and_grad(f, curr_val)
        
        
# if user wants to track gradients, needs pass a Dual object
# else it will be handled like a constant with zero gradient
if __name__=="__main__":
    # #h : R^n -> R^m
    
    # A = Dual([[1,2],[3,4]])
    # b = Dual([3,4])
    # x = linear_system(A,b)
    # print(x)
    
    
    theta = 1
    x = scalar_root_finding(g,theta)
    print(x)
    
    
    
    # final_val, final_out, final_der = newtons_method_kk(h,[-1,-2])
    # print(f"sol: {final_val}\nval: {final_out}\nder: {final_der}\n")
    
    
    
    
    
    # out_vector, jac_matrix = val_and_grad(g, [4])
    # print(out_vector) # value vector
    # print(jac_matrix) # jacobian
    
    # final_val, final_out, final_der = newtons_method(g,1)
    # print(f"sol: {final_val}\nval: {final_out}\nder: {final_der}\n")
    
    
    # x = Dual([1,2])
    # print(x)
    # y = Dual([3,4])
    # print(y)
    # z = dot_product(x,y)
    # print(z)
    
    # A = Dual([[1,2],
    #           [3,4],
    #           [5,6]])
    # print(A)
    # x = Dual([1,2])
    # print(x)
    # z = matrix_vector_product(A,x)
    # print(z)
    
    
    # A = Dual([[1,2],
    #           [3,4],
    #           [5,6]])
    # print(A)
    # B = Dual([[1,2,5],
    #           [6,2,3]])
    # print(B)
    # C = matrix_matrix_product(A,B)
    # print(C)
    
    