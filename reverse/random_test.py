from micro_grad import *
import numpy as np

num_data_rows = 100 # rows
batch_size = 25
num_features = 10 # cols


# X = np.random.rand(num_data_rows,num_features)
# # single value classification, binary classification
# bin_class_Y = np.random.rand(num_data_rows,1)
# #or: Y = np.random.rand(1,num_data_rows)

# # single value regression
# regr_Y = np.random.rand(num_data_rows,1) * 100

# # multiple value classification
# num_classes = 10
# mult_class_Y = np.eye(num_classes)[np.random.choice(num_classes, num_data_rows)]


# cos regression
# random x between [0,2*pi)

def get_minibatch(batch_size,X,Y):
    indices = np.random.permutation(X.shape[0])[:batch_size]
    return (Tensor(X[indices]), Tensor(Y[indices]))
    



num_feat = 1
cos_X = np.random.rand(num_data_rows,num_feat) * 2*np.pi
cos_Y = np.cos(cos_X)

mlp = MLP(num_feat, [16,1])
#lay = LinearLayer(1,1,nonlin='relu')

losses = []
for i in range(10000):
    inp,target = get_minibatch(batch_size,cos_X,cos_Y)
    pred = mlp(inp)
    loss = (pred-target).l2()
    losses.append(loss)
    mlp.zero_grad()
    loss.backward()
    mlp.step(3e-4)
line_plot(losses)



