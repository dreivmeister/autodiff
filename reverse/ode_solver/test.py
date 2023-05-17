import autograd.numpy as np
from explicit_methods import explicit_method

#from implicit_methods import implicit_method
"""
to implement:
- two explicit methods (rk23, ...)
- three implicit methods (implicit euler, BDF, crank nich., implicit adams)
"""




class OdeSolve:
    def __init__(self, f, t_span, y0, args=None, method='euler') -> None:
        # f(t, y, *args)
        self.t_span = t_span # (t0,t1)
        if t_span[1] < t_span[0]:
            raise AttributeError('t1 must be greater than t0')
        self.y0 = y0 # (y0,)
        self.args = args
        if args is not None:
            self.f = lambda t, y: f(t, y, *args)
        else:
            self.f = f
        self.method = method
        
    
    def solve(self, ret_res=True):
        
        if self.method in ['euler','rk4']:
            self.sol = explicit_method(self.f, self.t_span, self.y0, self.method)
        elif self.method in ['imp_euler']:
            self.sol = implicit_method(self.f, self.t_span, self.y0, self.method)
        else:
            raise AttributeError(f'wrong method name: {self.method}')
        
        if ret_res:
            return self.sol
    
    def plot(self):
        import matplotlib.pyplot as plt
        
        # 2d data
        if self.sol.ndim == 1:
            plt.plot(self.sol)
            plt.show()
        else:
            num_funcs = self.sol.shape[1]
            for i in range(num_funcs):
                plt.plot(self.sol[:,i])
            plt.show()
            
        
        




# lotka = OdeSolve(lotkavolterra, [0,15], [5,10], (1.5, 1, 3, 1), 'euler')

# sol = lotka.solve()
# print(sol.shape)
# lotka.plot()








