import numpy as np
from scipy import signal
from graphviz import Digraph

# Based on https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
# and https://github.com/ShivamShrirao/simple_autograd_numpy/blob/main/tensor.py

class Tensor:
    def __init__(self, data, prev=(), op=lambda x: None, name=None, *args, **kwargs):
        self.data = np.asarray(data, dtype=np.float32)
        self.prev = prev
        self.grad = 0
        self.op = op
        self.grad_fn = lambda x: None
        self.broadcast_dim = None
        self.name = name
        
        
        # if one dimensional -> add a batch dimension of 1
        # should maybe remove this when not working with neural nets
        # if self.data.ndim == 1:
        #     self.data = np.expand_dims(self.data, axis=0)
        
    def __repr__(self):
        if self.data.ndim > 1:
            return f'Tensor(data=\n{self.data}, grad=\n{self.grad})'
        return f'Tensor(data={self.data}, grad={self.grad})'
    
    def __getitem__(self, key):
        sliced_data = self.data[key]
        try:
            sliced_gradient = self.grad[key]
        except TypeError:
            sliced_gradient = 0
        
        sliced_tensor = Tensor(sliced_data)
        sliced_tensor.grad = sliced_gradient
        
        # or also:
        sliced_tensor.prev = (self,)
        sliced_tensor.op = self.__getitem__
        
        
        def grad_fn(gradient):
            if isinstance(self.grad, int) and self.grad == 0:
                self.grad = np.zeros_like(self.data)
            self.grad[key] += gradient
        sliced_tensor.grad_fn = grad_fn
        
        
        sliced_tensor.broadcast_dim = self.broadcast_dim
        sliced_tensor.name = self.name
        
        
        return sliced_tensor
    
    def __setitem__(self, key, value):
        # value is a tensor
        self.data[key] = value.data
        try:
            self.grad[key] = value.grad
        except TypeError:
            self.grad = value.grad
        
        # or also:
        #value.prev.add(value)
        self.prev = (value,)
        self.op = self.__setitem__
        
        def grad_fn(gradient):
            if isinstance(value.grad, int) and value.grad == 0:
                value.grad = np.zeros_like(value.data)
            value.grad += gradient[key]
        self.grad_fn = grad_fn
        
        self.broadcast_dim = value.broadcast_dim
        self.name = value.name
        

    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data, dtype=np.float32)
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
            
    
    @staticmethod
    def concatenate(seq,axis=0):
        # seq is a tuple of Tensors qhich should be concated
        # along axis (axis=0 - along rows), (axis=1 - along cols)
        # assume seq only contains 2d arrays
        n = len(seq)
        out = Tensor(np.concatenate([a.data for a in seq],axis=axis), (*seq,), op=Tensor.concatenate)
        
        def grad_fn(gradient):
            # gradient is of shape out
            gradient_split = np.split(gradient,n,axis=axis)
            
            for i in range(n):
                #seq[i].grad = np.zeros_like(gradient_split[i])
                seq[i].grad += gradient_split[i]
        out.grad_fn = grad_fn
        
        return out
    

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
            if other.data.ndim == 2:
                self.grad += gradient @ other.data.T
            else:
                self.grad += gradient @ np.transpose(other.data, (0,2,1))
            other.grad += np.transpose(self.data, (0,2,1)) @ gradient
        out.grad_fn = grad_fn
        
        return out
    
    def reshape(self, new_shape):
        
        old_shape = self.shape
        out = Tensor(np.reshape(self.data, new_shape), (self,), op=self.reshape)
        
        def grad_fn(gradient):
            self.grad += np.reshape(gradient, old_shape)
        out.grad_fn = grad_fn
        
        return out
            
    
    def sum(self, axis=None, keepdims=False):
        # only 2D matrices
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), op=self.sum)
        
        def grad_fn(gradient):
            
            if keepdims == False:
                if axis == None or axis == 0:
                    self.grad += gradient * np.ones_like(self.data)
                if axis == 1:
                    self.grad += gradient.T * np.ones_like(self.data)
            else:
                self.grad += gradient * np.ones_like(self.data)
                    
            # or: self.grad += gradient * np.ones_like(self.data.T)
            # might not work when self.data is non square
        out.grad_fn = grad_fn
        
        return out
    
    def max(self, axis=None, keepdims=True):
        m = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(m, (self,), op=self.max)
        
        def grad_fn(gradient):
            self.grad += gradient * np.equal(self.data, m).astype(np.float32)
        out.grad_fn = grad_fn
        
        return out
    
    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), op=self.mean)
        if axis == None:
            n = self.data.size
        elif axis == 0:
            n = self.data.shape[0]
        else:
            n = self.data.shape[1]
        
        def grad_fn(gradient):
            if keepdims == False:
                if axis == None or axis == 0:
                    self.grad += gradient * np.ones_like(self.data)*(1./n)
                if axis == 1:
                    self.grad += gradient.T * np.ones_like(self.data)*(1./n)
            else:
                self.grad += gradient * np.ones_like(self.data)*(1./n)
                
        out.grad_fn = grad_fn
        
        return out
    
    def var(self, axis=None, keepdims=False):
        out = Tensor(np.var(self.data, axis=axis, keepdims=keepdims), (self,), op=self.var)
        m = np.mean(self.data, axis=axis, keepdims=keepdims)
        n = self.data.size
        if axis == None:
            n = self.data.size
        elif axis == 0:
            n = self.data.shape[0]
        else:
            n = self.data.shape[1]
        
        
        def grad_fn(gradient):
            if keepdims == False:
                if axis == None or axis == 0:
                    self.grad += gradient * ((2./n)*(self.data - m))
                if axis == 1:
                    self.grad += gradient.T * ((2./n)*(self.data - np.expand_dims(m,axis=1)))
            else:
                self.grad += gradient * ((2./n)*(self.data - m))
        out.grad_fn = grad_fn
        
        return out
    
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), op=self.tanh)
        
        def grad_fn(gradient):
            self.grad += gradient * (t**2-1)
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
    
    def cos(self):
        out = Tensor(np.cos(self.data), (self,), op=self.cos)
        
        def grad_fn(gradient):
            self.grad += gradient * -np.sin(self.data)
        out.grad_fn = grad_fn
        
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), op=self.exp)
        
        def grad_fn(gradient):
            self.grad += gradient * out.data
        out.grad_fn = grad_fn
        
        return out
    
    def sqrt(self):
        return self ** (1/2)

    def log(self):
        out = Tensor(np.log(self.data), (self,), op=self.log)
        
        def grad_fn(gradient):
            self.grad += gradient * (1. / self.data)
        out.grad_fn = grad_fn
        
        return out
    
    def l2(self):
        # computes l2 loss of a Tensor
        # can be used for regression
        out = Tensor((1/2)*np.linalg.norm(self.data, ord=2)**2, (self,), op=self.l2)
        
        def grad_fn(gradient):
            # could also do: self.grad += out.grad * self.data
            self.grad += gradient * self.data
        out.grad_fn = grad_fn
        
        return out
    
    def softmax(self, axis=1):
        # computes softmax of self tensor
        # subtract max
        self.data = self.data - np.max(self.data, axis=axis, keepdims=True)
        # Compute the standard softmax formula.
        ex = np.exp(self.data)
        sigma = ex / np.sum(ex, axis=axis, keepdims=True)
        
        out = Tensor(sigma, (self,), op=self.softmax)
        
        def grad_fn(gradient):
            sums = np.sum(gradient[:, np.newaxis, :] * sigma[:, :, np.newaxis], axis=2)
            self.grad += sigma * (gradient - sums)
        out.grad_fn = grad_fn
        
        return out
        
    def dropout(self, p_drop, training=True):
        if training:
            p_keep = 1 - p_drop
            binary_mask = np.random.rand(self.data.shape[1],self.data.shape[2]) < p_keep
            result = self.data * binary_mask
            result /= p_keep # inverted dropout
        
            out = Tensor(result, (self,), op=self.dropout)
        
            def grad_fn(gradient):
                self.grad += gradient * binary_mask
            out.grad_fn = grad_fn
            
            return out
        
        return out
        
    
    
    def transpose(self, new_axis=None):
        if self.data.ndim == 2:
            out = Tensor(self.data.T, (self,), op=self.transpose)
            
            def grad_fn(gradient):
                # gradient.shape == out.shape
                self.grad += gradient.T 
            out.grad_fn = grad_fn
        # nd case
        elif new_axis is not None:
            old_shape = self.data.shape
            out = Tensor(np.transpose(self.data, new_axis), (self,), op=self.transpose)
            
            def grad_fn(gradient):
                # gradient.shape == out.shape
                self.grad += np.transpose(gradient, new_axis) 
            out.grad_fn = grad_fn
                
        return out
    
    def conv2d(self, kernels, output_shape):
        # self is the input image
        # other is a kernel
        # forward
        
        # TODO: multiple kernels, multiple input planes
        batch_size, out_channels, out_height, out_width = output_shape
        out = np.random.randn(*output_shape)
        
        print(f"out {out.shape}")
        print(f"kernels {kernels.shape}")
        print(f"self {self.shape}")
        
        in_channels = self.shape[1]
        #out = np.zeros_like(*output_shape)
        for k in range(batch_size):
            for i in range(out_channels):
                for j in range(in_channels):
                    out[k,i] += signal.correlate2d(self.data[k,j],kernels.data[i,j], "valid")
        
        out = Tensor(out, (self, kernels), op=self.conv2d)
        def grad_fn(gradient):
            print(f"gradient {gradient.shape}")
            self.grad = np.zeros_like(self.data)
            kernels.grad = np.zeros_like(kernels.data)

            for k in range(batch_size):
                for i in range(out_channels):
                    for j in range(in_channels):
                        kernels.grad[i,j] += signal.correlate2d(self.data[k,j],gradient[k,i],"valid")
                        self.grad[k,j] += signal.convolve2d(gradient[k,i],kernels.data[i,j],"full")
            # or: other.grad = signal.correlate2d(self.data,gradient,mode="valid") 
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
                    if nm == "<lambda>":
                        nm = str(t.data)
                        
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
    

    
    
class Head(Module):
    def __init__(self, block_size, n_embd, head_size, dropout=0.2, mask=False):
        self.key = LinearLayer(n_embd, head_size, bias=False)
        self.query = LinearLayer(n_embd, head_size, bias=False)
        self.value = LinearLayer(n_embd, head_size, bias=False)
        self.do_mask = mask
        if mask:
            m = np.zeros((block_size,block_size))
            m[np.triu_indices(block_size,1)] = -np.inf
            self.mask = Tensor(m)
        
        self.dropout = Dropout(dropout)
        
    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x) # (batch_size,block_size,token_dim) @ (n_embd, head_size) 
        q = self.query(x) # (B,T,C)
        
        wei = q @ k.transpose((0,2,1)) # transpose last two dims
        if self.do_mask:
            wei += self.mask
        wei = wei.softmax(axis=2) # (B, T, T)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out
    
    def parameters(self):
        return [*self.key.parameters(),*self.query.parameters(),*self.value.parameters()]



class MHA(Module):
    
    def __init__(self, block_size, n_embd, num_heads, head_size, dropout=0.5, do_mask=False):
        self.heads = [Head(block_size=block_size,n_embd=n_embd,head_size=head_size,mask=do_mask) for _ in range(num_heads)]
        self.proj = LinearLayer(n_embd, n_embd)
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        out = Tensor.concatenate([h(x) for h in self.heads],axis=-1)
        out = self.dropout(self.proj(out))
        return out

    def parameters(self):
        return [self.proj.parameters()] + [p for head in self.heads for p in head.parameters()]


class FeedForward(Module):
    
    def __init__(self, n_embd):
        self.ll1 = LinearLayer(n_embd, 4*n_embd,nonlin='relu')
        self.ll2 = LinearLayer(4*n_embd, n_embd)
        self.drop = Dropout(0.5)
    
    def __call__(self, x):
        return self.drop(self.ll2(self.ll1(x)))
    
    def parameters(self):
        return [*self.ll1.parameters(), *self.ll2.parameters()]
        
        


class Block(Module):
    
    def __init__(self, block_size, n_embd, num_heads, dropout=0.5, do_mask=False):
        # block_size - context_length - length of sample
        # n_embd - embedding_dimension - d_model
        # num_heads - number of heads in MHA
        # head_size - embedding dimension in single head
        head_size = n_embd // num_heads
        self.sa = MHA(block_size,n_embd,num_heads,head_size,dropout,do_mask)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        
    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
    def parameters(self):
        return [*self.sa.parameters(),*self.ln1.parameters(),*self.ln2.parameters(),*self.ffwd.parameters()]
        
        
        









    
    

class LinearLayer(Module):
    
    def __init__(self, nin, nout, bias=True, nonlin=None) -> None:
        super().__init__()
        k = np.sqrt(1/nin)
        self.w = Tensor(np.random.uniform(-k, k, (nout, nin)))
        
        if bias:
            self.b = Tensor(np.random.uniform(-k, k, (1,nout)))
        self.bias = bias
        self.nonlin = nonlin
    
    def __call__(self, x):
        if self.bias:
            act = x @ self.w.transpose() + self.b
        else:
            act = x @ self.w.transpose()
        return getattr(act, self.nonlin)() if self.nonlin else act
        #return act.relu() if self.nonlin else act
    
    def parameters(self):
        if self.bias:
            return [self.w, self.b]
        else:
            return [self.w]

class LogisticLayer(Module):
    
    def __init__(self, nfeat, bias=False) -> None:
        super().__init__()
        self.l = LinearLayer(nfeat, nfeat, bias=bias, nonlin='sigmoid')
    
    def __call__(self, x):
        return self.l(x)
    
    def parameters(self):
        # or: return [*self.l.parameters()]
        return [p for p in self.l.parameters()]
        

class MLP(Module):
    
    def __init__(self, nin, nouts, nonlin='relu') -> None:
        super().__init__()
        # nin is an integer
        # nouts is a list of integers
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            if i != len(nouts)-1:
                self.layers.append(LinearLayer(sz[i],sz[i+1],nonlin=nonlin))
            else:
                self.layers.append(LinearLayer(sz[i],sz[i+1],nonlin=False))
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    
class Convolution(Module):
    # only valid only stride 1
    
    def __init__(self, input_shape, kernel_size, num_filters) -> None:
        super().__init__()
        # input_shape - shape of input image (batch_size, channel_dim, height, width)
        # kernel_size - square kernel size only, int
        # depth - num of kernels, num of channels in output image
        batch_size, in_channels, input_height, input_width = input_shape
        self.num_filters = num_filters # num of kernels
        self.input_shape = input_shape
        self.input_depth = in_channels
        self.output_shape = (batch_size, num_filters, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (num_filters, in_channels, kernel_size, kernel_size)
        
        self.kernels = Tensor(np.random.randn(*self.kernels_shape))
    def __call__(self, x):
        # x is a Tensor of shape (batch_size, channel_dim, height, width)
        #out = Tensor(np.copy(self.bias))
        return x.conv2d(self.kernels, self.output_shape)
    
    def parameters(self):
        return [self.kernels]
    
class MultiHeadAttention(Module):
    
    def __init__(self, h, d_model, T, mask=False):
        # maybe dont need T
        self.h = h # num heads
        self.d_model = d_model # embedding dim
        self.T = T # sequence length
        self.d_k = self.d_v = int(d_model/h) #dimension key,value
        self.mask = None
        if mask:
            # gen mask
            # shape might be wrong
            m = np.zeros((self.d_model,self.d_model))
            m[np.triu_indices(self.d_model,1)] = -np.inf
            self.mask = Tensor(m)
            
        
        # one list of three matrices for each head
        self.weights_heads = []
        for i in range(h):
            k = np.sqrt(1/self.d_k)
            # for each head
            #nin - number of columns
            #nout - number of rows
            
            weights_headi = [Tensor(np.random.uniform(-k, k, (self.d_model, self.d_k))),
                             Tensor(np.random.uniform(-k, k, (self.d_model, self.d_k))),
                             Tensor(np.random.uniform(-k, k, (self.d_model, self.d_v)))]
            # weights_headi = [Wi^Q,Wi^K,Wi^V]
            self.weights_heads.append(weights_headi)
        
        # output linear projection
        self.W_O = Tensor(np.random.uniform(-k, k, (int(self.h*self.d_v), self.d_model)))
    
    def __call__(self, Q, K, V):
        
        head_attentions = []
        for i in range(self.h):
            WQ = self.weights_heads[i][0]
            WK = self.weights_heads[i][1]
            WV = self.weights_heads[i][2]
            head_attentions.append(Attention(Q @ WQ, K @ WK, V @ WV, self.mask))
            
        concated_heads = Tensor.concatenate(head_attentions,axis=1)
        lin_proj_concated_heads = concated_heads @ self.W_O
        
        return lin_proj_concated_heads
    
    def parameters(self):
        return [self.W_O] + [mat for head in self.weights_heads for mat in head]
    
    
class Dropout(Module):
    def __init__(self, p_drop) -> None:
        self.p_drop = p_drop
    
    def __call__(self, x, training=True):
        return x.dropout(self.p_drop, training)
    
    def parameters(self):
        return []
    
class BatchNorm1D(Module):
    #https://github.com/renan-cunha/BatchNormalization/blob/master/src/feed_forward/layers.py
    def __init__(self, num_features, momentum=0.1) -> None:
        self.num_features = num_features
        self.gamma = Tensor(np.ones(self.num_features))
        self.beta = Tensor(np.zeros(self.num_features))
        self.eps = 1e-5
        self.momentum = momentum
        
        
        # not supported yet
        self.running_mean = Tensor(np.zeros(self.num_features))
        self.running_var = Tensor(np.ones(self.num_features))
        
    def __call__(self, x, training=True):
        # x is of shape (N, num_features)
        # or maybe not dont know
        # mean and var along axis=0
        if training:
            m = x.mean(axis=0, keepdims=True)
            v = x.var(axis=0, keepdims=True) + self.eps
            
            # running mean and var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * m
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * v
            
            
            return self.gamma*((x - m)/v.sqrt()) + self.beta
        # testing
        return self.gamma/self.running_var * x + (self.beta - (self.gamma*self.running_mean)/self.running_var)
        
    
    def parameters(self):
        return [self.gamma, self.beta]
        

class TemporalAffine(Module):
    """
    Inputs:
    batch_size - N
    series_length - T
    in_dim - D
    out_dim - M
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    """
    
    def __init__(self, batch_size, series_length, in_dim, out_dim):
        self.W = Tensor(np.random.rand(in_dim, out_dim))
        self.b = Tensor(np.random.rand(out_dim))
        
        self.N = batch_size
        self.T = series_length
        self.D = in_dim
        self.M = out_dim
        
    def __call__(self, x):
        # x of shape (batch_size, series_length, out_dim)
        return (x.reshape(self.N * self.T, self.D) @ self.W).reshape(self.N, self.T, self.M) + self.b
    
    def parameters(self):
        return [self.W, self.b]



class LayerNorm(Module):
    
    def __init__(self, normalized_shape):
        self.normalized_shape = normalized_shape
        # compute mean and variance over last D dims of input
        #self.D = len(normalized_shape) 
        self.gamma = Tensor(np.ones(normalized_shape))
        self.beta = Tensor(np.zeros(normalized_shape))
        self.eps = 1e-5
        
    def __call__(self, x):
        # x is of shape normalized_shape
        m = x.mean(axis=1, keepdims=True)
        v = x.var(axis=1, keepdims=True) + self.eps
        
        return ((x - m)/v.sqrt())*self.gamma + self.beta
        
        
    def parameters(self):
        return [self.gamma, self.beta]
    

class VanillaRNNBlock(Module):
    
    def __init__(self, N, T, D, H, h0):
        """
        - batch size - N
        - seq length - T
        - elem dim   - D
        - hidden dim - H
        """
        
        self.N = N
        self.T = T
        self.H = H
        self.prev_h = h0 # previous hidden state is h0
        self.Wx = Tensor(np.random.rand(D,H))
        self.Wh = Tensor(np.random.rand(H,H))
        self.b = Tensor(np.random.rand(H))
    
    def rnn_step(self, x):
        # x is of shape (N,D)
        return (x @ self.Wx + self.prev_h @ self.Wh + self.b).tanh()
    
    
    def __call__(self, x):
        # x is of shape (N,T,D)
        seq = []
        
        for i in range(self.T):
            step_out = vanilla_rnn_step(x[:,i,:])
            seq.append(step_out)
            self.prev_h = step_out
        
        return Tensor.concatenate(seq).reshape((self.N,self.T,self.H))
    
    def parameters(self):
        return [self.Wx, self.Wh, self.b]
            
        
        


def vanilla_rnn_step(x, prev_h, Wx, Wh, b):
    #https://github.com/jariasf/CS231n/blob/master/assignment3/cs231n/rnn_layers.py
    """
    all inputs are Tensors
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    """
    
    return (x @ Wx + prev_h @ Wh + b).tanh()
    
def vanilla_rnn(x, h0, Wx, Wh, b):
    """
    all inputs and outputs are tensors
    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    """
    
    # hacky solution incoming
    seq = []
    N, T, D = x.shape
    H = h0.shape[1]
    prev_h = h0
    
    for i in range(T):
        prev_h = vanilla_rnn_step(x[:,i,:], prev_h, Wx, Wh, b)
        seq.append(prev_h)
    
    return Tensor.concatenate(seq).reshape((N,T,H))
    



    
def softmax(logits):
    # logits is a 2D Tensor
    # each row is one sample, each column is one feature
    logits -= logits.max(axis=1)
    ex = logits.exp()
    ex_sum = ex.sum(axis=1, keepdims=True)
    sigma = ex / ex_sum
    return sigma

def logistic_prediction(inputs, targets, logistic_layer):
    preds = logistic_layer(inputs)
    return negative_log_likelihood(preds, targets)

def l2_loss(preds, targets):
    # preds is a prediction vector
    # target contains the target values
    return (preds - targets).l2()

def negative_log_likelihood(probs, targets):
    # binary classification
    # preds is a probability vector
    # targets is a vector of zeros and ones
    label_probs = probs * targets + (1 - probs) * (1 - targets)
    return -(label_probs.log().sum())

def cross_entropy(probs, targets):
    # preds is a probability vector (each column sums to one)
    # targets is a one hot vector
    log_probs = probs.log()
    return -(targets * log_probs).sum()
        
def hinge_loss(logits, targets):
    # logits is not a prob vector (columns dont sum to 1)
    num_samples = logits.data.shape[0]
    return 1./num_samples * (1. - targets * logits).relu().sum()


def Attention(Q,K,V,mask):
    # Q, K and V must be of the following shapes (Assumption)
    # Q, K, and V are Tensors
    # Q - (m, d_k)
    # K - (n, d_k), K.T - (B, d_k, n)
    # V - (n, d_v)
    # Scaled Dot-Product Attention
    # Attention(Q,K,V) = softmax((Q @ K.T) / sqrt(d_k)) @ V
    d_k = Q.shape[1]
    #d_v = V.shape[1]
    scaling_factor = np.sqrt(d_k)
    
    D = Q @ K.transpose() # (m, n)
    D = D / scaling_factor
    if mask:
        D = D + mask
    D = softmax(D)
    A = D @ V # (m, d_v)
    return A




def line_plot(tensor_data, ylabel='loss', xlabel='epochs'):
    import matplotlib.pyplot as plt
    
    plt.plot([p.data for p in tensor_data])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    
if __name__=="__main__":
    
    
    # m = 10
    # n = 6
    # d_k = 4
    # d_v = 8
    # Q = Tensor(np.random.randn(m,d_k))
    # K = Tensor(np.random.randn(n, d_k))
    # V = Tensor(np.random.randn(n, d_v))
    
    # scaled_dot_attention = Attention(Q,K,V)
    # scaled_dot_attention.backward()
    # print(Q)
    # print(K)
    # print(V)
    
    # dims
    # T = m = n = 20 # (m and n dont have to be the same)
    # d_model = 64
    # h = 4
    # d_k = d_v = d_model/h
    # Q = Tensor(np.random.randn(T,d_model))
    # K = Tensor(np.random.randn(T, d_model))
    # V = Tensor(np.random.randn(T, d_model))
    # # Q @ Wi^Q - (T, d_k)
    # # K @ Wi^K - (T, d_k)
    # # V @ Wi^V - (T, d_v)
    # # headi - (T, d_v)
    # # C = Concat(headi,...headn) - (T, d_v*h) = (T, d_model)
    # # C @ W^O - (T, d_model) (W^O is square)
    
    # MHA = MultiHeadAttention(h,d_model,T)
    # out = MHA(Q,K,V)
    # print(out.shape)
    # out.backward()
    
    
    x = Tensor(np.random.randn(10,5))
    LN = LayerNorm(5)
    y = LN(x)
    print(y.data.var(axis=1))
    print(y.shape)
    y.backward()
    #print(x)
    
    
    
    
    
    
    
    
    
    
    # x = Tensor(np.array([1,2,4]), name="x")
    # y = Tensor(np.array([1,3,9]), name="y")
    
    # z = 4 * (x + y)
    
    # z.backward()
    # z.generate_graph().render()
    
    
    
    # in_shape = (10,1,32,32)
    
    
    # x = Tensor(np.random.randn(*in_shape))
    # #print(x.shape)
    # #print(k.shape)
    
    # #out = x.correlate2d(k)
    # #print(out)
    
    # c = Convolution(in_shape,kernel_size=3,num_filters=32)
    
    # out = c(x)
    # out.backward()
    
    # print(f"x after back {x}")
    # print(f"k after back {c.kernels}")
    
    
    # loss = out.sum()
    # loss.backward()
    # #loss.generate_graph().render()
    
    # print(out)
    # print(x)
    # print(c.kernels[0])
    
    
    
    
    #print(x.shape)
    #print(targets)
    # y = ll(x)
    # print(y.shape)
    # print(y)
    
    # batch_size = 5
    # num_features = 3
    # inputs = Tensor(np.random.uniform(-10, 10, (batch_size, num_features)))
    # # # two classes
    # #targets = Tensor(np.random.randint(2, size=(batch_size, num_features)).astype(np.float32))
    # targets2 = Tensor(np.eye(num_features)[np.random.choice(num_features, batch_size)])
    #print(targets2)
    # #print(inputs)
    # outputs = softmax(inputs)
    # #print(outputs)
    
    # outputs.backward()
    
    # print(inputs)
    
    # out = inputs.max(axis=0)
    # print(out)
    
    # out.backward()
    
    # print(inputs)
    
    
    # out = inputs.tanh()
    # print(out)
    # out.backward()
    # print(inputs)
    
    # print(inputs.shape)
    # probs = softmax(inputs)
    # #print(probs)
    # ce_loss = cross_entropy(probs, targets2)
    # print(ce_loss)
    # ce_loss.backward()
    
    
    
    # ll = LogisticLayer(num_features)
    
    # losses = []
    # for i in range(1000):
    #     # includes the prediction step
    #     loss = logistic_prediction(inputs, targets, ll)
    #     losses.append(loss)
    #     # if i % 10 == 0:
    #     #     print(loss)
    #     ll.zero_grad()
    #     loss.backward()
    #     ll.step(0.01)
    # line_plot(losses)
    
    
    
    
    # bs = 10
    # x = Tensor(np.random.uniform(-10, 10, (10,3)))
    # print(x.shape)
    
    # l = Layer(3,5)
    # y = l(x)
    # print(y.shape)
    
    # W = Tensor(np.array([[1,0],
    #                      [0,1]], dtype=np.float32), name='W')
    # # need column vectors with one dimension on axis=1 x.shape=(n,1)
    # # happens implicitly
    # b = Tensor(np.array([2,3], dtype=np.float32), name='b')
    # x = Tensor(np.random.uniform(-10, 10, 3))
    # #print(x.shape)
    
    # # l = Layer(3,5)
    # # y = l(x)
    # # print(f"{l.w.shape} @ {x.shape} + {l.b.shape} = {y.shape}")
    # # print(y)
    # # y.backward()
    # # #print(x.grad, l.w.grad, l.b.grad)
    # # print(l.w)
    # # l.step(3e-4)
    # # print(l.w)
    
    # #a = Tensor(np.array([1],dtype=np.float32))
    # bs = 32
    # nin = 3
    # losses = []
    # mlp = MLP(nin, [16,16,1])
    # for i in range(100):
    #     x = Tensor(np.random.uniform(-10, 10, (bs,nin)))
    #     y = mlp(x)
    #     loss = (y-1).l2()
    #     losses.append(loss)
    #     mlp.zero_grad()
    #     loss.backward()
    #     mlp.step(3e-4)
    
    # print(losses[0])
    # print(losses[-1])
    # line_plot(losses)
    # pred = mlp(Tensor(np.array([1,3,5])))
    # print(pred)
    
    
    
    
    
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