class NeuralODE:
    """
    z0 - initial state
    f - Neural Network is a Module
    t0 - initial time step
    t1 - final time step
    h - step size
    """
    def __init__(self, f, t0, t1, h):
        self.f = f
        self.t0 = t0
        self.t1 = t1
        self.h = h
        self.theta = f.parameters()
    
    def forward(self, z0):
        num_steps = int((self.t1-self.t0)/self.h)
        tk = self.t0
        z1 = z0
        for _ in range(num_steps):
            tk += self.h
            z1 = euler_step(self.h, self.f, tk, z1)
        return z1
    
    def backward(self, z1):        
        num_steps = int((self.t1-self.t0)/self.h)
        tk = self.t1
        z0 = z1
        for _ in range(num_steps):
            tk -= self.h
            z0 = euler_step(self.h, self.f, tk, z0)
        return z0