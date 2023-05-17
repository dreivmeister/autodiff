from engine import *
from nn import *
from optim import *

def get_minibatch(batch_size,X,Y):
    indices = np.random.permutation(X.shape[0])[:batch_size]
    return (Tensor(X[indices]), Tensor(Y[indices]))


num_images = 100
num_channels = 1
height = width = 28
num_classes = 5

X = np.random.uniform(-3,3,(num_images,num_channels,height,width))
# random one hot vector
Y = np.eye(num_classes)[np.random.choice(num_classes, num_images)]


class Model(Module):
    
    def __init__(self, input_channels, num_classes):
        self.cn1 = Conv2d(input_channels,6,3)
        self.ln1 = LayerNorm((6,26,26))
        self.cn2 = Conv2d(6,16,3)
        self.mp1 = MaxPool2d(2,2)
        self.mp2 = MaxPool2d(2,2)
        
        self.ll1 = LinearLayer(400,240,nonlin='relu')
        self.ll2 = LinearLayer(240,120,nonlin='relu')
        self.ll3 = LinearLayer(120,num_classes)
    
    def __call__(self, x, training=True):
        block1_out = self.mp1(self.ln1(self.cn1(x))).relu()
        block2_out = self.mp2(self.cn2(block1_out)).relu()
        
        conv_out_flatten = block2_out.flatten()
        ll_out = self.ll3(self.ll2(self.ll1(conv_out_flatten)))
        return ll_out

    def parameters(self):
        return [*self.cn1.parameters(),
                *self.cn2.parameters(),
                *self.ll1.parameters(),
                *self.ll2.parameters(),
                *self.ll3.parameters(),
                *self.ln1.parameters()]


conv_net = Model(1,num_classes)


# batch_X, batch_Y = get_minibatch(5, X, Y)
# print(batch_X.shape)
# print(batch_Y.shape)

# out = conv_net(batch_X)
# print(out.shape)
# probs = softmax(out)
# print(probs.shape)

# loss = cross_entropy(probs, batch_Y)
# print(loss)
# loss.backward()



# FEEL LIKE IT CANT REALLY LEARN
batch_size = 20
sgd = Adam(conv_net.parameters(),3e-4)
losses = []
# single batch
inp,target = get_minibatch(batch_size, X, Y)
for i in range(50):
    # or multibatch
    #inp,target = get_minibatch(batch_size,regr_X,regr_Y)
    pred = conv_net(inp)
    probs = softmax(pred)
    loss = cross_entropy(probs, target)
    losses.append(loss)
    conv_net.zero_grad()
    loss.backward()
    sgd.step()
    
    
#test batch size must be equal to train batch size !!!!!!!!!!!!!!!!!!!
test_inp,test_target = get_minibatch(20, X, Y)

probs = softmax(conv_net(test_inp, training=False))
test_loss = cross_entropy(probs, test_target)
print(test_loss)
print(losses[-1])
line_plot(losses)




#TODO: correct cross_entropy, softmax, log_softmax, nll implementation