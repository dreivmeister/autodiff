import torch
from micro_grad import *

a = 5
b = 3
x = np.random.rand(a,b)
y = np.random.rand(b,a)


def test_prim_one(x, f, name=""):
    x1 = torch.Tensor(x)
    x1.requires_grad = True
    y1 = f(x1)
    #print(y1)
    y1.sum().backward()

    x2 = Tensor(x)
    y2 = f(x2)
    #print(y2)
    y2.sum().backward()
    
    
    if np.allclose(y2.data, y1.data.numpy()) and np.allclose(x2.grad,x1.grad.numpy()):
        print(f"{name} SUCCESS!")
        return True # test success
    
    print(f"{name} FAILED!")
    #print(loss.data, loss1.data)
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
    

# one input
test_prim_one(x, lambda x: x+x, name='add')
test_prim_one(x, lambda x: x*x, name='mul')
power = np.random.randint(1,10)
test_prim_one(x, lambda x: x**power, name='pow')
# sub, div are compositions of add and mul and power


test_prim_one(x, lambda x:x.sum(),name='sum')
test_prim_one(x, lambda x:x.sum(0),name='sum')
test_prim_one(x, lambda x:x.sum(1),name='sum')

test_prim_one(x, lambda x:x.sum(0, keepdims=True),name='sum')
test_prim_one(x, lambda x:x.sum(1, keepdims=True),name='sum')

# test_prim_one(x, lambda x:x.max(),name='max')
# test_prim_one(x, lambda x:x.max(0),name='max')
# test_prim_one(x, lambda x:x.max(1),name='max')

test_prim_one(x, lambda x:x.mean(),name='mean')
test_prim_one(x, lambda x:x.mean(0),name='mean')
test_prim_one(x, lambda x:x.mean(1),name='mean')

test_prim_one(x, lambda x:x.mean(0, keepdims=True),name='mean')
test_prim_one(x, lambda x:x.mean(1, keepdims=True),name='mean')

    
# have to check that one out
# try with bessels correction in torch =0
test_prim_one(x, lambda x:x.var(),name='var')
test_prim_one(x, lambda x:x.var(0),name='var')
test_prim_one(x, lambda x:x.var(1),name='var')

test_prim_one(x, lambda x: x.relu() if type(x) == Tensor else torch.nn.functional.relu(x), name='relu')

# might not work because of var difference
test_prim_one(x, lambda x: x.softmax() if type(x) == Tensor else torch.nn.functional.softmax(x, dim=1), name='softmax') 

x1 = Tensor(x)
out1 = x1.softmax()
print(out1)
out1.backward()
print(x1)

x2 = Tensor(x)
out2 = softmax(x2)
print(out2)
out2.backward()
print(x2)




# mult input
#matmul
def test_prim_two(x,y,f,name=""):
    x1 = torch.Tensor(x)
    x1.requires_grad = True
    x11 = torch.Tensor(y)
    x11.requires_grad = True
    y1 = f(x1,x11)
    y1.sum().backward()

    x2 = Tensor(x)
    x22 = Tensor(y)
    y2 = f(x2,x22)
    y2.sum().backward()


    if np.allclose(y2.data, y1.data.numpy()) and np.allclose(x2.grad,x1.grad.numpy()):
        print(f"{name} SUCCESS!")
        return True # test success

    print(f"{name} FAILED!")
    #print(loss.data, loss1.data)
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

test_prim_two(x,y,lambda x1,x2: x1 @ x2, name='matmul')


