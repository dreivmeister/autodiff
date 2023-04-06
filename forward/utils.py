import numpy as np
from dual import Dual


def _equal(x, val, der, eval_der=None):
    if eval_der is None:
        return np.isclose(x.val, val).all() and np.isclose(x.der, der).all()
    else:
        return np.isclose(x.val, val).all() and np.isclose(eval_der, der).all()


def _compare(comparison, val, der):
    x = Dual(*comparison)
    return _equal(x, val, der)