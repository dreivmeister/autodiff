import numpy as np
import torch
from micro_grad import *

def test_function(x, f1, f2, name=""):
    x1 = torch.Tensor(x)
    x1.requires_grad = True
    y1 = f1(x1)
    y1.sum().backward()
    
    
    x2 = Tensor(x)
    y2 = f2(x2)
    y2.sum().backward()
    
    if np.allclose(y2.data, y1.data.numpy()) and np.allclose(x2.grad,x1.grad.numpy()):
        print(f"{name} SUCCESS!")
        return True # test success
    
    print(f"{name} FAILED!")
    # forward pass went well
    if np.allclose(y2.data, y1.data.numpy()):
        print(f"{name} forward pass went well")
    else:
        print(f"{name} forward pass failed")
        print(f"data\n mg:\n {y2.data}\ntorch:\n {y1.data.numpy()}")
    # backward pass went well
    if np.allclose(x2.grad,x1.grad.numpy()):
        print(f"{name} backward pass went well")
    else:
        print(f"{name} backward pass failed")
        print(f"grad\n mg:\n {x2.grad}\ntorch:\n {x1.grad.numpy()}")
    
    return False

batch_size = 5
num_features = 3
x = np.random.rand(batch_size,num_features)
y = np.random.rand(batch_size,num_features)

# test_function(x, torch.nn.BatchNorm1d(num_features), BatchNorm1D(num_features), name="batchnorm")
# test_function(x, torch.nn.Linear(3, 6), LinearLayer(3,6), name="linearlayer")


# test Attention
# query = np.random.rand(1,128,64)
# key = np.random.rand(1,128,64)
# value = np.random.rand(1,128,64)

# mg = Attention(Tensor(query[0]),Tensor(key[0]),Tensor(value[0]))

# q1 = torch.Tensor(query)
# q1.requires_grad = True
# k1 = torch.Tensor(key)
# k1.requires_grad = True
# v1 = torch.Tensor(value)
# v1.requires_grad = True

# to = torch.nn.functional.scaled_dot_product_attention(q1,k1,v1)


# mg_l2 = l2_loss(Tensor(x),Tensor(y))

# x1 = torch.Tensor(x)
# x1.requires_grad = True
# y1 = torch.Tensor(y)
# y1.requires_grad = True
# torch_l2 = torch.nn.functional.mse_loss(x1,y1)

# print(mg_l2)
# print(torch_l2)












#still to test:
"""
- Convolution
- MultiHeadAttention
- Attention
- MLP
- softmax
- l2_loss
- sigmoid
- crossentropy
- nll
- hinge_loss
"""