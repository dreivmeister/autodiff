import numpy as np

D_NUMBER = 'NUMBER'
D_VECTOR = 'VECTOR'


class Dual:
    def __init__(self, real, dual=None) -> None:
        """
        Dual(1.0)
        Dual([1.,2.])
        Dual([1.,2.],[2.,3.])
        """
        self.type = None
        self.real = None
        self.dual = None
        
        # real can be int or float
        if isinstance(real, (int,float)):
            self.type = D_NUMBER
            self.real = real
            if isinstance(dual, (int, float)):
                self.dual = dual
            elif dual is None:
                self.dual = 1.0
        # real can be nparray (n-dim) or list
        elif isinstance(real, (np.ndarray, list)):
            self.type = D_VECTOR
            self.real = np.array(real)
            if isinstance(dual, (np.ndarray, list)):
                self.dual = np.array(dual)
            elif dual is None:
                self.dual = np.ones_like(self.real)
        else:
            raise AttributeError('wrong init')
    # elementwise operations
    def __add__(self, other):
        if isinstance(other, Dual):
            pass
        elif isinstance(other, (int, float)):
            # constant
            other = Dual(other, 0.)
        elif isinstance(other, (np.ndarray, list)):
            other = Dual(other, np.zeros_like(other))
        else:
            raise AttributeError('wrong args add')
        
        if isinstance(other, Dual):
            return Dual(self.real+other.real,self.dual+other.dual)
        else:
            raise AttributeError('wrong type add')
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if isinstance(other, Dual):
            pass
        elif isinstance(other, (int, float)):
            # constant
            other = Dual(other, 0.)
        elif isinstance(other, (np.ndarray, list)):
            other = Dual(other, np.zeros_like(other))
        else:
            raise AttributeError('wrong args mul')
        
        if isinstance(other, Dual):
            # product rule
            return Dual(self.real*other.real,self.real*other.dual+self.dual*other.real)
        else:
            raise AttributeError('wrong type mul')
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        if isinstance(other, Dual):
            pass
        elif isinstance(other, (int, float)):
            # constant
            other = Dual(other, 0.)
        elif isinstance(other, (np.ndarray, list)):
            other = Dual(other, np.zeros_like(other))
        else:
            raise AttributeError('wrong args sub')
        
        if isinstance(other, Dual):
            # product rule
            return Dual(self.real-other.real,self.dual-other.dual)
        else:
            raise AttributeError('wrong type sub')
    def __rsub__(self, other):
        return self - other
    
    def __truediv__(self, other):
        if isinstance(other, Dual):
            pass
        elif isinstance(other, (int, float)):
            # constant
            other = Dual(other, 0.)
        elif isinstance(other, (np.ndarray, list)):
            other = Dual(other, np.zeros_like(other))
        else:
            raise AttributeError('wrong args truediv')
        
        if isinstance(other, Dual):
            # quotient rule
            return Dual(self.real/other.real,
                        (other.real * self.dual - self.real * other.dual) / (other.real**2))
        else:
            raise AttributeError('wrong type truediv')
    def __rtruediv__(self, other):
        return self / other
        
def f(x):
    return (4*(x * x) + 3 - x - 12) / (2*x)

def gradient_checker(f, x, eps=1e-8):
    if isinstance(x, (int, float)):
        pass
    elif isinstance(x, (np.ndarray, list)):
        x = np.array(x)
    else:
        raise AttributeError('wrong type gradcheck')
    return (f(x+eps)-f(x))/eps

def pushforward(f, primal, tangent):
    inp = Dual(primal, tangent)
    output = f(inp)
    return output.real, output.dual

def val_and_grad(f, x):
    if isinstance(x, (int, float)):
        v = 1.0
    elif isinstance(x, (np.ndarray, list)):
        v = np.ones_like(x)
    f_x, df_dx = pushforward(f, x, v)
    return f_x, df_dx
    
        
if __name__=="__main__":
    #d = Dual(1.0)
    
    print(val_and_grad(f,4.))
    print(gradient_checker(f,4.))
    print(val_and_grad(f,[1.,2.]))
    print(gradient_checker(f,[1.,2.]))
    