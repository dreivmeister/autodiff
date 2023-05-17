from refactor import *
from optim import AdamW
import numpy as np

num_data_rows = 100 # rows
batch_size = 25
num_features = 10 # cols

def get_minibatch(batch_size,X,Y):
    indices = np.random.permutation(X.shape[0])[:batch_size]
    return (Tensor(X[indices]), Tensor(Y[indices]))

# single value regression
regr_X = np.random.rand(num_data_rows,num_features)
regr_Y = np.random.rand(num_data_rows,1) * 100


#mlp = MLP(num_features, [16,16,1])

class Model(Module):
    
    def __init__(self, num_feat):
        self.l1 = LinearLayer(num_feat, 16, nonlin='relu')
        self.l2 = LinearLayer(16, 16, nonlin='relu')
        self.l3 = LinearLayer(16, 1)
        self.d1 = Dropout(0.5)
        self.bn1 = BatchNorm1D(16)
        self.ln1 = LayerNorm(16)
    
    def __call__(self, x):
        x1 = self.d1(self.ln1(self.l1(x)))
        y = self.l3(self.bn1(self.l2(x1)))
        return y
    
    def parameters(self):
        return [*self.l1.parameters(),
                *self.l2.parameters(),
                *self.l3.parameters(),
                *self.bn1.parameters(),
                *self.ln1.parameters()]
        

mlp = Model(num_features)
adam = AdamW(mlp.parameters())
losses = []
# single batch
inp,target = get_minibatch(batch_size,regr_X,regr_Y)
for i in range(1000):
    # or multibatch
    #inp,target = get_minibatch(batch_size,regr_X,regr_Y)
    pred = mlp(inp)
    loss = (pred-target).l2()
    losses.append(loss)
    mlp.zero_grad()
    loss.backward()
    adam.step()
line_plot(losses)

