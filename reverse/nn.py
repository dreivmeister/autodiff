from engine import *


class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            # or: p.grad = np.zeros_like(p.data)
    
    def step(self, lr):
        # sgd update step
        for p in self.parameters():
            p.data = p.data - lr * p.grad
    
    def parameters(self):
        return []
    
    
class LinearLayer(Module):
    
    def __init__(self, nin, nout, bias=True, nonlin=None) -> None:
        super().__init__()
        k = np.sqrt(1/nin)
        self.w = Tensor.uniform(-k,k,(nout,nin))
        #self.w = Tensor(np.random.uniform(-k, k, (nout, nin)))
        
        if bias:
            self.b = Tensor.uniform(-k,k,(1,nout))
            # or: self.b = Tensor.uniform(-k,k,(nout,))
            #self.b = Tensor(np.random.uniform(-k, k, (1,nout)))
        self.bias = bias
        self.nonlin = nonlin
    
    def __call__(self, x):
        act = x @ self.w.transpose((-1,-2))
        if self.bias:
            act = act + self.b
        return getattr(act, self.nonlin)() if self.nonlin else act
        #return act.relu() if self.nonlin else act
    
    def parameters(self):
        if self.bias:
            return [self.w, self.b]
        return [self.w]
    

class MLP(Module):
    
    def __init__(self, nin, nouts, nonlin='relu') -> None:
        super().__init__()
        # nin is an integer
        # nouts is a list of integers
        sizes = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            if i != len(nouts)-1:
                self.layers.append(LinearLayer(sizes[i],sizes[i+1],nonlin=nonlin))
            else:
                self.layers.append(LinearLayer(sizes[i],sizes[i+1],nonlin=False))
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    
class Dropout(Module):
    def __init__(self, p_drop) -> None:
        self.p_drop = p_drop
    
    def __call__(self, x, training=True):
        return x.dropout(self.p_drop, training)
    
    def parameters(self):
        return []
    
    
class BatchNorm1D(Module):
    # input of shape (N,D)
    #https://github.com/renan-cunha/BatchNormalization/blob/master/src/feed_forward/layers.py
    def __init__(self, num_features, momentum=0.1) -> None:
        self.num_features = num_features
        self.gamma = Tensor.ones(num_features)
        self.beta = Tensor.zeros(num_features)
        self.momentum = momentum
        
        self.running_mean = Tensor.zeros(num_features)
        self.running_var = Tensor.ones(num_features)
        
    def __call__(self, x, training=True):
        # x is of shape (N, num_features)
        # or maybe not dont know
        # mean and var along axis=0
        if training:
            m = Tensor.mean(x, axis=0, keepdims=True)
            v = Tensor.var(x, axis=0, keepdims=True) + 1e-5
            
            # running mean and var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * m
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * v
            
            return self.gamma*((x - m)/v.sqrt()) + self.beta
        # testing
        return self.gamma/self.running_var * x + (self.beta - (self.gamma*self.running_mean)/self.running_var)
        
    def parameters(self):
        return [self.gamma, self.beta]
    
    
class LayerNorm(Module):
    # input of shape (N,D)
    # or other i think
    def __init__(self, normalized_shape):
        # normalized_shape is equivalent to num_features for input in form (N,num_features)
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        elif isinstance(normalized_shape, tuple):
            self.normalized_shape = normalized_shape
        
        # i think this is correct but might not be
        self.axis_tuple = tuple([i for i in range(1, len(self.normalized_shape)+1)])
        #self.axis_tuple = tuple(range(-len(self.normalized_shape), 0))
        
        self.gamma = Tensor.ones(normalized_shape)
        self.beta = Tensor.zeros(normalized_shape)
        
    def __call__(self, x):
        # x is of shape normalized_shape
        m = Tensor.mean(x, axis=self.axis_tuple, keepdims=True)
        v = Tensor.var(x, m, axis=self.axis_tuple, keepdims=True) + 1e-5
        
        
        
        
        
        return ((x - m)/v.sqrt())*self.gamma + self.beta
        
        
    def parameters(self):
        return [self.gamma, self.beta]
    
    
#TODO: make Conv2d more flexible (stride,pad,...) and maybe faster
class Conv2d(Module):
    # only valid only stride 1
    def __init__(self, in_channels, out_channels, kernel_size):
        # input_shape - shape of input image (batch_size, channel_dim, height, width)
        # kernel_size - square kernel size only, int
        # depth - num of kernels, num of channels in output image
        #batch_size, in_channels, input_height, input_width = input_shape
        self.kernel_size = kernel_size
        #self.num_filters = out_channels # num of kernels
        #self.input_shape = input_shape
        self.input_depth = in_channels
        self.num_filters = out_channels
        #self.output_shape = (batch_size, num_filters, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (out_channels, in_channels, kernel_size, kernel_size)
        
        self.kernels = Tensor.rand(self.kernels_shape)
        #self.kernels = Tensor(np.random.randn(*self.kernels_shape))
    def __call__(self, x):
        # x is a Tensor of shape (batch_size, channel_dim, height, width)
        #out = Tensor(np.copy(self.bias))
        output_shape = (x.shape[0],self.num_filters,x.shape[2]-self.kernel_size+1,x.shape[3]-self.kernel_size+1)
        return x.conv2d(self.kernels, output_shape)
    
    def parameters(self):
        return [self.kernels]
    
class MaxPool2d(Module):
    
    def __init__(self, kernel_size, stride):
        if isinstance(kernel_size, int):
            self.kernel_height = self.kernel_width = kernel_size
        elif isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size
        
        self.stride = stride
        
    def __call__(self, x):
        # x is (N,C,H,W)
        return x.maxpool2d(self.kernel_height, self.kernel_width, self.stride)

    def parameters(self):
        return []
        

    
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
        self.Wx = Tensor.rand(D,H)
        self.Wh = Tensor.rand(H,H)
        self.b = Tensor.rand(H)
    
    def rnn_step(self, x):
        # x is of shape (N,D)
        return (x @ self.Wx + self.prev_h @ self.Wh + self.b).tanh()
    
    
    def __call__(self, x):
        # x is of shape (N,T,D)
        seq = []
        
        for i in range(self.T):
            step_out = self.rnn_step(x[:,i,:])
            seq.append(step_out)
            self.prev_h = step_out
        
        return Tensor.concatenate(seq).reshape((self.N,self.T,self.H))
    
    def parameters(self):
        return [self.Wx, self.Wh, self.b]
    

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
        return [*self.proj.parameters()] + [p for head in self.heads for p in head.parameters()]
    
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


# hl functions
def softmax(logits):
    # logits is a 2D Tensor
    # each row is one sample, each column is one feature
    logits -= logits.max(axis=1,keepdims=True)
    ex = logits.exp()
    ex_sum = ex.sum(axis=1, keepdims=True)
    sigma = ex / ex_sum
    return sigma

def log_softmax(logits):
    return logits / logits.exp().sum(axis=1,keepdims=True).log()

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
    log_probs = (probs + 10e-20).log()
    return -(targets * log_probs).sum()

def hinge_loss(logits, targets):
    # logits is not a prob vector (columns dont sum to 1)
    num_samples = logits.data.shape[0]
    return 1./num_samples * (1. - targets * logits).relu().sum()
    
def line_plot(tensor_data, ylabel='loss', xlabel='epochs'):
    import matplotlib.pyplot as plt
    
    plt.plot([p.data for p in tensor_data])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()