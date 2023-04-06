import numpy as np


# class DDual:
#     def __init__(self, val, der=1.0):
#         if np.isscalar(val):
#             self.val = np.array([val])
#         else:
#             self.val = np.array(val)
#         self.der = der
    

class Dual:
    """
    Primary data structure for forward mode automatic differentiation.

    Dual numbers can be used as a data structure to store the value and
    derivative of a function. The value and derivative can be stored as
    the real and "dual" part of a dual number, respectively. The properties
    of a dual number lend itself nicely to a straightforward implementation of
    forward mode automatic differentiation.

    Parameters
    ----------
    val : float
        The value of the Dual number.
    der : ndarray
        The derivative of the Dual number.

    Examples
    --------
    Construct a Dual number for a univariate function:

    >>> x = ad.Dual(42)
    >>> x
    Dual(42, array([1]))

    Construct a Dual number for a multivariate function with user-defined seed vector:

    >>> x = ad.Dual(42, [1, 2])
    >>> x
    Dual(42, array([1, 2]))

    Construct multiple Dual numbers from array of scalars:

    >>> x, y, z = ad.Dual.from_array([1, 2, 4])
    >>> x
    Dual(1, array([1, 0, 0]))
    >>> y
    Dual(2, array([0, 1, 0]))

    Create a function from multiple Dual numbers:

    >>> x, y, z = ad.Dual.from_array([1, 2, 4])
    >>> f = (x * y)/z
    >>> f.val
    0.5
    >>> f.der
    array([0.5, 0.25, -0.125])

    See Also
    --------
    Dual.constant
    Dual.from_array

    """
    def __init__(self, val, der=1.0):
        self.val = val
        self.der = np.array(der, ndmin=1)
            
    @property
    def ndim(self):
        return len(self.der)

    @staticmethod
    def constant(val, ndim=1):
        zeros = np.zeros(ndim)
        return Dual(val, zeros)
    
    @staticmethod
    def constant_vector(val):
        zeros = np.zeros_like(val)
        return Dual(val, zeros)

    @staticmethod
    def from_array(X, var_out=True):
        if np.ndim(X) != 1:
            raise Exception(f"array must be 1-dimensional")
        if len(X) == 1:
            return Dual(X[0], 1.0)
        
        X = np.array(X)
        
        if var_out:
            # variable output
            I = np.identity(len(X))
            return iter(Dual(x, I[i]) for i, x in enumerate(X))
        else:
            # vector output
            d = np.ones_like(X)
            return Dual(X, d)

    def _compatible(self, other, operand=None):
        if isinstance(other, (int, float)):
            return Dual.constant(other, ndim=self.ndim)
        elif isinstance(other, (np.ndarray, list)):
            return Dual.constant_vector(other)
        elif isinstance(other, Dual):
            if self.ndim == other.ndim:
                return other
            raise ArithmeticError(
                f"Dimensionality mismatch between {self} and {other}")
        raise TypeError(
            f"unsupported operand type(s) for {operand}: '{type(self).__name__}' and '{type(other).__name__}'"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.val}, {np.array_repr(self.der)})"

    def __add__(self, other):
        if other := self._compatible(other, "+"):
            return Dual(self.val + other.val, self.der + other.der)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if other := self._compatible(other, "-"):
            return Dual(self.val - other.val, self.der - other.der)

    def __rsub__(self, other):
        if other := self._compatible(other, "-"):
            return Dual(other.val - self.val, other.der - self.der)

    def __mul__(self, other):
        if other := self._compatible(other, "*"):
            return Dual(self.val * other.val,
                        self.val * other.der + self.der * other.val)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if other := self._compatible(other, "/"):
            return Dual(self.val / other.val,
                        (other.val * self.der - self.val * other.der) /
                        (other.val**2))

    def __rtruediv__(self, other):
        if other := self._compatible(other, "/"):
            return Dual(other.val / self.val,
                        (self.val * other.der - other.val * self.der) /
                        (self.val**2))

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            if self.val < 0 and (other != int(other)):
                raise ValueError(
                    f"{self.val} cannot be raised to the power of {other}; only integer powers are allowed if base is negative"
                )
            elif self.val == 0 and other < 1:
                raise ZeroDivisionError(
                    f"0.0 cannot be raised to a negative power")
        elif isinstance(other, Dual):
            if self.val <= 0:
                raise ValueError(
                    f"{self.val} cannot be raised to the power of {other.val}; log is undefined for x = {self.val}"
                )
        try:
            der_comp_2 = other.der * np.log(
                self.val) + other.val * (self.der / self.val)
            return Dual(self.val**other.val,
                        (self.val**other.val) * der_comp_2)
        except AttributeError:
            return Dual(self.val**other,
                        other * self.val**(other - 1) * self.der)

    def __rpow__(self, other):
        if other <= 0:
            raise ValueError(
                f"{other} cannot be raised to the power of {self.val}; log is undefined for x = {other}"
            )

        val = other**self.val
        der = val * np.log(other) * self.der
        return Dual(val, der)

    def __neg__(self):
        return Dual(-self.val, -self.der)

    def __lt__(self, other):
        if other := self._compatible(other, "<"):
            return self.val < other.val, self.der < other.der

    def __gt__(self, other):
        if other := self._compatible(other, ">"):
            return self.val > other.val, self.der > other.der

    def __le__(self, other):
        if other := self._compatible(other, "<="):
            return self.val <= other.val, self.der <= other.der

    def __ge__(self, other):
        if other := self._compatible(other, ">="):
            return self.val >= other.val, self.der >= other.der

    def __eq__(self, other):
        if other := self._compatible(other, "=="):
            return self.val == other.val, self.der == other.der

    def __ne__(self, other):
        if other := self._compatible(other, "!="):
            return self.val != other.val, self.der != other.der


def f(x):
    return x**4 + 3*x

def pushforward(f, primal, tangent):
    if isinstance(primal, float):
        inp = Dual(primal, tangent)
    elif isinstance(primal, np.ndarray):
        inp = Dual.from_array(primal, var_out=False)
    output = f(inp)
    return output.val, output.der

def derivative(f, x):
    if isinstance(x, float):
        v = 1.0
    elif isinstance(x, np.ndarray):
        v = np.ones_like(x)
    f_x, df_dx = pushforward(f, x, v)
    return f_x, df_dx




if __name__=="__main__":
    
    
    x_p = np.array([10.,8.])
    print(derivative(f,x_p))
    
    
    
    # t = Dual.constant_vector([1.,2.])
    # print(t)
    
    # a1 = np.array([1.,2.])
    # a2 = np.array([2.,3.])
    
    
    # d = Dual.from_array(a1, var_out=False)
    # d1 = Dual.from_array(a2, var_out=False)
    
    # d2 = d + a2
    # print(d2.val, d2.der)
    # print(d2.val, d.val + d1.val)
    # print(d2.der, d.der + d1.der)