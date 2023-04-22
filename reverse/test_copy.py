from micro_grad import *
"""
A = Tensor(np.array([[1,2],[3,4]]))
b = A[0] # b = [1,2]
"""


# A = Tensor(np.array([[1,2],
#                      [3,4]]))


# # cant do self refrencing like this
# #A[0] = A[1] ** 2

# # A.backward()
# # A.generate_graph().render()
# # print(A)


# print(A[:,0])


# E = Tensor.concatenate((
#         A[0] ** 2,
#         A[1] ** 2,
#         A[0] * A[1]
#         )).reshape((3,2))
# print(E)
# E.backward()
# print(A)
# print(E)
# E.generate_graph().render()



# C = A ** 2
# C.backward()

#print(B)

#A.generate_graph().render()



# N = 1
# T = 2
# D = 3
# H = 4

# x = Tensor(np.random.rand(N,T,D))
# h0 = Tensor(np.random.rand(N,H))
# Wx = Tensor(np.random.rand(D,H))
# Wh = Tensor(np.random.rand(H,H))
# b = Tensor(np.random.rand(H))
# # or: b = np.random.rand(1,H)


# out = vanilla_rnn(x,h0,Wx,Wh,b)
# out.sum().backward()
# print(out.shape) # expecting: (N,T,H)



batch_size = 4
block_size = 8

n_embd = 32
n_head = 4
head_size = n_embd // n_head
x = Tensor(np.random.rand(batch_size,block_size,n_embd))
#MH = MHA(block_size,n_embd,n_head,head_size,do_mask=True)
B = Block(block_size,n_embd,n_head,do_mask=True)
out = B(x)
print(out.shape)
out.sum().backward()
out.generate_graph().render()
#print(x.grad.shape)






