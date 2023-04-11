import numpy as np

class Node:
    def __init__(self, data):
        if isinstance(data, (int,float)):
            self.data = data
        elif isinstance(data, (np.ndarray, list)):
            self.data = np.array(data)
        self.children = []
        self.grad = None
        
    def __repr__(self):
        if self.data.ndim > 1:
            return f"{self.__class__.__name__}\n{self.data}"
        return f"{self.__class__.__name__}({self.data})"
        
    def compute_grad(self):
        if self.children is not None and len(self.children) == 0:
            return 1.0
        if self.grad is None:
            self.grad = sum(np.dot(weight,node.compute_grad()) for weight, node in self.children)
        return self.grad

    @staticmethod
    def zero_grad(*args):
        for arg in args:
            try:
                arg.children = []
                arg.grad = None
            except AttributeError:
                raise AttributeError(f'Cannot set gradient to zero for type {arg.__class__.__name__}')
    
    @staticmethod
    def constant(data):
        node = Node(data)
        node.children = None
        node.grad = 0
        return node
    
    @staticmethod
    def from_array(X):
        # if np.ndim(X) != 1:
        #     raise Exception(f"array must be 1-dimensional")
        if len(X) == 1:
            return Node(X[0])

        return list(iter(Node(x) for x in X))
    
    def _isConstant(self, other, operand=None):
        if isinstance(other, (int, float)):
            return Node.constant(other)
        elif isinstance(other, Node):
            return other
        raise TypeError(f"unsupported operand type(s) for {operand}: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def _addChildren(self, new_weight, new_child):
        if self.children is not None:
            self.children.append((new_weight, new_child))
    
    
    def __add__(self,other):
        if other := self._isConstant(other):
            child = Node(self.data + other.data)
            self._addChildren(1.0, child)
            other._addChildren(1.0, child)
            return child
    def __radd__(self,other):
        return self + other
    
    def __mul__(self,other):
        if other := self._isConstant(other):
            child = Node(self.data * other.data)
            self._addChildren(other.data, child)
            other._addChildren(self.data, child)
            return child
    def __rmul__(self,other):
        return self * other
    
    def __sub__(self,other):
        if other := self._isConstant(other):
            child = Node(self.data - other.data)
            self._addChildren(1.0,child)
            other._addChildren(-1.0,child)
            return child
        
    def __rsub__(self,other):
        if other := self._isConstant(other):
            child = Node(other.data - self.data)
            self._addChildren(-1.0,child)
            other._addChildren(1.0,child)
            return child
        
    def __truediv__(self, other):
        if other := self._isConstant(other):
            child = Node(self.data/other.data)
            self._addChildren(1/other.data,child)
            other._addChildren(-self.data/(other.data**2),child)
            return child

    def __rtruediv__(self, other):
        if other := self._isConstant(other):
            child = Node(other.data/self.data)
            self._addChildren(-other.data/(self.data**2),child)
            other._addChildren(1/self.data,child)
            return child
        
    def __pow__(self, other):
        try:
            val = self.data**other.data
            child = Node(val)
            self._addChildren(val * other.data / self.data, child)
            other._addChildren(val * np.log(self.data), child)
            return child
        except AttributeError:
            child = Node(self.data**other)
            self._addChildren(other*self.data**(other-1),child)
            return child
    
    def __neg__(self):
        if self.children == None:
            child = Node.constant(-self.data)
        else:
            child = Node(-self.data)
            self._addChildren(-1.0,child)
        return child

def dot_product(a, b):
    pass

def matrix_vector_product(A, x):
    # A*x as matrix vector product A (2D), x (1D)
    if not isinstance(A, Node) or not isinstance(x, Node):
        raise AttributeError('wrong type mvp')

    
    if A.data.ndim != 2 or x.data.ndim != 1:
        raise AttributeError('wrong ndims in mvp')
    if A.data.shape[1] != x.data.shape[0]:
        raise AttributeError('wrong shape in mvp')
    
    child = Node(np.dot(A.data,x.data))
    # this is weird
    x._addChildren(A.data.T,child)
    A._addChildren(x.data.T,child)
    return child

def matrix_matrix_product(A, B):
    # A*x as matrix matrix product A (2D), B (2D)
    if not isinstance(A, Node) or not isinstance(x, Node):
        raise AttributeError('wrong type mmp')
    
    
    if A.data.ndim != 2 or B.data.ndim != 2:
        raise AttributeError('wrong ndims in mmp')
    if A.data.shape[1] != B.data.shape[0]:
        raise AttributeError('wrong shape in mmp')
    
    child = Node(np.matmul(A.data,B.data))
    # this is weird
    A._addChildren(B.data.T,child)
    B._addChildren(A.data.T,child)
    return child

    
def jacobian(results):
    return np.vstack(results)

def val_and_grad(function, eval_point):
    # eval_point - list of scalar(s)
    # like: [2,4,6] or [2] or [2.3,234.2]
    if not isinstance(eval_point, list):
        raise AttributeError('eval_point must be a list of scalars or a of one scalar')
    v = Node.from_array(eval_point)
    # only one output possible
    # one input
    if isinstance(v, Node):
        function_out = function(v)
        return function_out, v.compute_grad()
    # mult input
    else:
        function_out = function(*v)
        return function_out, jacobian([x.compute_grad() for x in v])
    
    

def f(x):
    return -(x ** 3)
    


if __name__=="__main__":
    # x = Node([[5,2],
    #           [2,3]])
    # print(x)
    # y = Node([3,7])
    # print(y)
    # z = f(x, y)
    # f1_grad = [x.compute_grad(),y.compute_grad()]
    # Node.zero_grad(x,y)
    # print(z)
    
    #print(val_and_grad(f,[2]))
    
    
    # print(jacobian(f1_grad))
    
    
    # v = Node.from_array([1,2,3])
    # print(list(v))
    
    A = Node([[1,2],
              [3,4]])
    B = Node([[4,9],
              [3,2]])
    x = Node([1,2])
    #print(A)
    #print(x)
    
    #C = matrix_matrix_product(A,B)
    
    #Node.zero_grad(A)
    z = matrix_vector_product(A,x)
    
    
    print(A.compute_grad())
    #print(B.compute_grad())
    print(x.compute_grad())