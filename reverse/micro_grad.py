import numpy as np
from graphviz import Digraph

# Based on https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
# and https://github.com/ShivamShrirao/simple_autograd_numpy/blob/main/tensor.py

class Tensor:
    def __init__(self, data, prev=(), op=lambda x: None, name=None, *args, **kwargs):
        self.data = np.asarray(data)
        self.prev = prev
        self.grad = 0
        self.op = op
        self.grad_fn = lambda x: None
        self.broadcast_dim = None
        self.name = name
        
        if self.data.ndim == 1:
            self.data = np.expand_dims(self.data, axis=1)

    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data)
        self.grad = gradient

        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t.prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        for t in reversed(topo):
            t.grad_fn(t.grad)
            
    def checkbroadcast(self, other):
        for n,(i,j) in enumerate(zip(self.shape, other.shape)):
            if i==j:
                continue
            if i<j:
                self.broadcast_dim = n
                break
            else:
                other.broadcast_dim = n
                break

    def __repr__(self):
        r = repr(self.data)
        return r[:10].replace('array','tensor') + r[10:]

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.checkbroadcast(other)
        out = Tensor(self.data + other.data, (self, other), op=self.__add__)
        
        def grad_fn(gradient):
            self.grad += gradient if self.broadcast_dim is None else gradient.sum(axis=self.broadcast_dim, keepdims=True)
            other.grad += gradient if other.broadcast_dim is None else gradient.sum(axis=other.broadcast_dim, keepdims=True)
        out.grad_fn = grad_fn
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), op=self.__mul__)
        
        def grad_fn(gradient):
            self.grad += gradient * other.data
            other.grad += gradient * self.data
        out.grad_fn = grad_fn
        
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data ** other, (self,), op=self.__pow__)
        
        def grad_fn(gradient):
            self.grad += gradient * (other * (self.data ** (other-1)))
        out.grad_fn = grad_fn
        
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), op=self.__matmul__)
        
        def grad_fn(gradient):
            self.grad += gradient @ other.data.T
            other.grad += self.data.T @ gradient
        out.grad_fn = grad_fn
        
        return out
    
    def relu(self):
        out = Tensor(self.data*(self.data>0), (self,), op=self.relu)
        
        def grad_fn(gradient):
            self.grad += gradient * (self.data > 0)
        out.grad_fn = grad_fn
        
        return out

    def sigmoid(self):
        out = Tensor(1.0 / (1 + np.exp(-self.data)), (self,), op=self.sigmoid)
        
        def grad_fn(gradient):
            self.grad += gradient * (out.data * (1 - out.data))
        out.grad_fn = grad_fn
        
        return out

    def sin(self):
        out = Tensor(np.sin(self.data), (self,), op=self.sin)
        
        def grad_fn(gradient):
            self.grad += gradient * np.cos(self.data)
        out.grad_fn = grad_fn
        
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), op=self.exp)
        
        def grad_fn(gradient):
            self.grad += gradient * out.data
        out.grad_fn = grad_fn
        
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), op=self.log)
        
        def grad_fn(gradient):
            self.grad += gradient * (1. / self.data)
        out.grad_fn = grad_fn
        
        return out
    
    def l2(self):
        out = Tensor((1/2)*np.linalg.norm(self.data, ord=2)**2, (self,), op=self.l2)
        
        def grad_fn(gradient):
            self.grad += gradient * self.data
        out.grad_fn = grad_fn
        
        return out
    
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * self**-1

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    # plot utility
    def generate_graph(self):
        dot = Digraph(comment='DAG')
        visited = set()
        def build_graph(t):
            if t not in visited:
                visited.add(t)
                if t.name:
                    nm = t.name
                    shape = "box"
                    color = ""
                else:
                    nm = t.op.__name__
                    shape = ""
                    color = "lightblue2"
                dot.node(str(hash(t)), nm, shape=shape, color=color, style='filled')
                for p in t.prev:
                    dot.edge(str(hash(p)), str(hash(t)))
                    build_graph(p)
        build_graph(self)
        return dot
    
    
#NN
class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            # or: p.grad = np.zeros_like(p.data)
    
    def step(self, lr):
        # sgd update step
        for p in self.parameters():
            p.data -= lr * p.grad
    
    def parameters(self):
        return []

class Layer(Module):
    
    def __init__(self, nin, nout, nonlin=True) -> None:
        super().__init__()
        k = np.sqrt(1/nin)
        self.w = Tensor(np.random.uniform(-k, k, (nout, nin)))
        self.b = Tensor(np.random.uniform(-k, k, nout))
        self.nonlin = nonlin
    
    def __call__(self, x):
        act = self.w @ x + self.b
        return act.relu() if self.nonlin else act
    
    def parameters(self):
        return [self.w, self.b]
    
class MLP(Module):
    
    def __init__(self, nin, nouts) -> None:
        super().__init__()
        # nin is an integer
        # nouts is a list of integers
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def line_plot(data, ylabel='loss', xlabel='epochs'):
    import matplotlib.pyplot as plt
    
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    
if __name__=="__main__":
    # W = Tensor(np.array([[1,0],
    #                      [0,1]], dtype=np.float32), name='W')
    # # need column vectors with one dimension on axis=1 x.shape=(n,1)
    # # happens implicitly
    # b = Tensor(np.array([2,3], dtype=np.float32), name='b')
    x = Tensor(np.random.uniform(-10, 10, 3))
    #print(x.shape)
    
    # l = Layer(3,5)
    # y = l(x)
    # print(f"{l.w.shape} @ {x.shape} + {l.b.shape} = {y.shape}")
    # print(y)
    # y.backward()
    # #print(x.grad, l.w.grad, l.b.grad)
    # print(l.w)
    # l.step(3e-4)
    # print(l.w)
    
    #a = Tensor(np.array([1],dtype=np.float32))
    losses = []
    mlp = MLP(3, [16,1])
    for i in range(200):
        y = mlp(x)
        
        loss = (y-1).l2()
        losses.append(loss.data)
        mlp.zero_grad()
        loss.backward()
        mlp.step(3e-4)
    
    line_plot(losses)
    #print(mlp(Tensor(np.array([1,3,5]))))
    
    
    
    
    
    # y = (W @ x + b).relu()
    # print(y)
    # y.backward()
    # print(W.grad) 
    # print(x.grad)
    
    
    
    
    # a = Tensor(np.array([1], dtype=np.float32), name='a')
    # b = Tensor(np.array([2], dtype=np.float32), name='b')
    # c = Tensor(np.array([3], dtype=np.float32), name='c')

    # d = (a * b).sin()
    # e = (c - (a / b)).exp()
    # f = d + e
    # y = (f ** 4).log() * c
    # print(y)
    # y.backward()

    # print(a.grad, b.grad, c.grad, d.grad, e.grad, f.grad)

    # a = Tensor(np.array([2], dtype=np.float32), name='a')

    # def fn(a):
    #     b = a.sin()
    #     c = a.log()
    #     d = c/b*a
    #     return (c+d/a).exp()

    # e = fn(a)
    # print(e)

    # e.backward()
    
    

    #e.generate_graph()