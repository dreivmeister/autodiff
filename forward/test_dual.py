import numpy as np
import pytest

from dual import Dual

from utils import _equal, _compare


@pytest.mark.parametrize("val", [1, -6.2])
def test_dual_constant(val):
    x = Dual.constant(val)
    assert _equal(x, val, 0)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [-2, 4.2])
def test_dual_univariate(val, der):
    x = Dual(val, der)
    assert _equal(x, val, der)


@pytest.mark.parametrize("val", [0.7, -64])
@pytest.mark.parametrize("der", [np.array([-3.4, 6]), np.array([-1, 6])])
def test_dual_multivariate(val, der):
    x = Dual(val, der)
    assert _equal(x, val, der)


@pytest.mark.parametrize("vals", [np.array([-3.4, 6]), np.array([-1, 6])])
def test_dual_from_array(vals):
    xs = list(Dual.from_array(vals))

    for x, val, der in zip(xs, vals, np.identity(len(vals))):
        assert _equal(x, val, der)


@pytest.mark.parametrize("val", [np.array([0.7, -64])])
def test_dual_from_array_vector_out(val):
    x = Dual.from_array(val, var_out=False)
    assert _equal(x, val, np.ones_like(val))


def test_dual_from_non_1d_array():
    with pytest.raises(Exception):
        Dual.from_array([[1, 2], [3, 4]])


@pytest.mark.parametrize("val", [[0.7], [-64]])
def test_dual_from_array_single_val(val):
    x = Dual.from_array(val)
    assert _equal(x, val[0], 1)


def test_dual_compat_mismatch_dims_error():
    with pytest.raises(ArithmeticError):
        x = Dual(1)
        y = Dual(1, [1, 2])
        return x + y


def test_dual_compat_type_error():
    with pytest.raises(TypeError):
        x = Dual(1)
        y = "autodiff"
        return x + y


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_add_constant(val1, val2):
    x = val1
    y = Dual.constant(val2)
    assert _equal(x + y, val1 + val2, 0)

    x = Dual.constant(val1)
    y = val2
    assert _equal(x + y, val1 + val2, 0)

    x = Dual.constant(val1)
    y = Dual.constant(val2)
    assert _equal(x + y, val1 + val2, 0)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [-3.4, 6])
@pytest.mark.parametrize("der2", [-1, 5])
def test_add_univariate(val1, der1, val2, der2):
    x = val1
    y = Dual(val2, der2)
    assert _equal(x + y, val1 + val2, der2)

    x = Dual(val1, der1)
    y = val2
    assert _equal(x + y, val1 + val2, der1)

    x = Dual(val1, der1)
    y = Dual(val2, der2)
    assert _equal(x + y, val1 + val2, der1 + der2)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [np.array([-3.4, 6]), np.array([-1, 24.2])])
@pytest.mark.parametrize("der2", [np.array([-4, 2]), np.array([-1.1, 32])])
def test_add_multivariate(val1, der1, val2, der2):
    x = val1
    y = Dual(val2, der2)
    assert _equal(x + y, val1 + val2, der2)

    x = Dual(val1, der1)
    y = val2
    assert _equal(x + y, val1 + val2, der1)

    x = Dual(val1, der1)
    y = Dual(val2, der2)
    assert _equal(x + y, val1 + val2, der1 + der2)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_sub_constant(val1, val2):
    x = val1
    y = Dual.constant(val2)
    assert _equal(x - y, val1 - val2, 0)

    x = Dual.constant(val1)
    y = val2
    assert _equal(x - y, val1 - val2, 0)

    x = Dual.constant(val1)
    y = Dual.constant(val2)
    assert _equal(x - y, val1 - val2, 0)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [-3.4, 6])
@pytest.mark.parametrize("der2", [-1, 5])
def test_sub_univariate(val1, der1, val2, der2):
    x = val1
    y = Dual(val2, der2)
    assert _equal(x - y, val1 - val2, 0 - der2)

    x = Dual(val1, der1)
    y = val2
    assert _equal(x - y, val1 - val2, der1 - 0)

    x = Dual(val1, der1)
    y = Dual(val2, der2)
    assert _equal(x - y, val1 - val2, der1 - der2)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [np.array([-3.4, 6]), np.array([-1, 24.2])])
@pytest.mark.parametrize("der2", [np.array([-4, 2]), np.array([-1.1, 32])])
def test_sub_multivariate(val1, der1, val2, der2):
    x = val1
    y = Dual(val2, der2)
    assert _equal(x - y, val1 - val2, 0 - der2)

    x = Dual(val1, der1)
    y = val2
    assert _equal(x - y, val1 - val2, der1 - 0)

    x = Dual(val1, der1)
    y = Dual(val2, der2)
    assert _equal(x - y, val1 - val2, der1 - der2)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_mul_constant(val1, val2):
    x = val1
    y = Dual.constant(val2)
    assert _equal(x * y, val1 * val2, val1 * 0 + val2 * 0)

    x = Dual.constant(val1)
    y = val2
    assert _equal(x * y, val1 * val2, val1 * 0 + val2 * 0)

    x = Dual.constant(val1)
    y = Dual.constant(val2)
    assert _equal(x * y, val1 * val2, val1 * 0 + val2 * 0)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [-3.4, 6])
@pytest.mark.parametrize("der2", [-1, 5])
def test_mul_univariate(val1, der1, val2, der2):
    x = val1
    y = Dual(val2, der2)
    assert _equal(x * y, val1 * val2, val1 * der2 + val2 * 0)

    x = Dual(val1, der1)
    y = val2
    assert _equal(x * y, val1 * val2, val1 * 0 + val2 * der1)

    x = Dual(val1, der1)
    y = Dual(val2, der2)
    assert _equal(x * y, val1 * val2, val1 * der2 + val2 * der1)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [np.array([-3.4, 6]), np.array([-1, 24.2])])
@pytest.mark.parametrize("der2", [np.array([-4, 2]), np.array([-1.1, 32])])
def test_mul_multivariate(val1, der1, val2, der2):
    x = val1
    y = Dual(val2, der2)
    assert _equal(x * y, val1 * val2, val1 * der2 + val2 * 0)

    x = Dual(val1, der1)
    y = val2
    assert _equal(x * y, val1 * val2, val1 * 0 + val2 * der1)

    x = Dual(val1, der1)
    y = Dual(val2, der2)
    assert _equal(x * y, val1 * val2, val1 * der2 + val2 * der1)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_truediv_constant(val1, val2):
    x = val1
    y = Dual.constant(val2)
    assert _equal(x / y, val1 / val2, 0)

    x = Dual.constant(val1)
    y = val2
    assert _equal(x / y, val1 / val2, 0)

    x = Dual.constant(val1)
    y = Dual.constant(val2)
    assert _equal(x / y, val1 / val2, 0)


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [-3.4, 6])
@pytest.mark.parametrize("der2", [-1, 5])
def test_truediv_univariate(val1, der1, val2, der2):
    x = val1
    y = Dual(val2, der2)
    assert _equal(x / y, val1 / val2, (-val1 * der2) / (val2**2))

    x = Dual(val1, der1)
    y = val2
    assert _equal(x / y, val1 / val2, (der1 * val2) / (val2**2))

    x = Dual(val1, der1)
    y = Dual(val2, der2)
    assert _equal(x / y, val1 / val2, (val2 * der1 - val1 * der2) / (val2**2))


@pytest.mark.parametrize("val1", [0.7, -64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [np.array([-3.4, 6]), np.array([-1, 24.2])])
@pytest.mark.parametrize("der2", [np.array([-4, 2]), np.array([-1.1, 32])])
def test_truediv_multivariate(val1, der1, val2, der2):
    x = val1
    y = Dual(val2, der2)
    assert _equal(x / y, val1 / val2, (-val1 * der2) / (val2**2))

    x = Dual(val1, der1)
    y = val2
    assert _equal(x / y, val1 / val2, (der1 * val2) / (val2**2))

    x = Dual(val1, der1)
    y = Dual(val2, der2)
    assert _equal(x / y, val1 / val2, (val2 * der1 - val1 * der2) / (val2**2))


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
def test_pow_constants(val1, val2):
    x = val1
    y = Dual.constant(val2)
    assert _equal(x**y, val1**val2, 0)

    x = Dual.constant(val1)
    y = val2
    assert _equal(x**y, val1**val2, 0)

    x = Dual.constant(val1)
    y = Dual.constant(val2)
    assert _equal(x**y, val1**val2, 0)


@pytest.mark.parametrize("val1", [-0.7, -64])
@pytest.mark.parametrize("val2", [-2.1, 4.2])
def test_pow_invalid(val1, val2):
    with pytest.raises(ValueError):
        x = val1
        y = Dual.constant(val2)
        _ = x**y

    with pytest.raises(ValueError):
        x = Dual.constant(val1)
        y = val2
        _ = x**y

    with pytest.raises(ValueError):
        x = Dual.constant(val1)
        y = Dual.constant(val2)
        _ = x**y

    with pytest.raises(ZeroDivisionError):
        x = Dual.constant(0)
        y = -2.1
        _ = x**y


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [-3.4, 6])
@pytest.mark.parametrize("der2", [-1, 5])
def test_pow_univariate(val1, der1, val2, der2):
    x = val1
    y = Dual(val2, der2)
    assert _equal(x**y, val1**val2, val1**val2 * np.log(val1) * der2)

    x = Dual(val1, der1)
    y = val2
    assert _equal(x**y, val1**val2, val2 * val1**(val2 - 1) * der1)

    x = Dual(val1, der1)
    y = Dual(val2, der2)
    int_der = der2 * np.log(val1) + val2 * (der1 / val1)
    assert _equal(x**y, val1**val2, val1**val2 * int_der)


@pytest.mark.parametrize("val1", [0.7, 64])
@pytest.mark.parametrize("val2", [-2, 4.2])
@pytest.mark.parametrize("der1", [np.array([-3.4, 6]), np.array([-1, 24.2])])
@pytest.mark.parametrize("der2", [np.array([-4, 2]), np.array([-1.1, 32])])
def test_pow_multivariate(val1, der1, val2, der2):
    x = val1
    y = Dual(val2, der2)
    assert _equal(x**y, val1**val2, val1**val2 * np.log(val1) * der2)

    x = Dual(val1, der1)
    y = val2
    assert _equal(x**y, val1**val2, val2 * val1**(val2 - 1) * der1)

    x = Dual(val1, der1)
    y = Dual(val2, der2)
    int_der = der2 * np.log(val1) + val2 * (der1 / val1)
    assert _equal(x**y, val1**val2, val1**val2 * int_der)


def test_neg_constants():
    x = Dual.constant(2)
    y = Dual.constant(-2)
    val = True
    der = True
    assert _compare((-x == y), val, der)


def test_neg_univariate():
    x, y = Dual(-1, 11), Dual(1, 11)
    val = True
    der = False
    assert _compare((-x == y), val, der)


def test_lt_constants():
    x = Dual.constant(2)
    y = Dual.constant(3)
    val = True
    der = False
    assert _compare((x < y), val, der)


def test_lt_univariate():
    x, y = Dual(1, 11), Dual(2, 20)
    val = True
    der = True
    assert _compare((x < y), val, der)


def test_lt_multivariate():
    x, y = Dual.from_array([6, -6])
    val = False
    der = [False, True]
    assert _compare((x < y), val, der)


def test_gt_constants():
    x = Dual.constant(1)
    y = Dual.constant(2)
    val = False
    der = False
    assert _compare((x > y), val, der)


def test_gt_univariate():
    x, y = Dual(1, 11), Dual(2, -5)
    val = False
    der = True
    assert _compare((x > y), val, der)


def test_gt_multivariate():
    x, y = Dual.from_array([8, 2])
    val = True
    der = [True, False]
    assert _compare((x > y), val, der)


def test_le_constants():
    x = Dual.constant(1)
    y = Dual.constant(2)
    val = True
    der = True
    assert _compare((x <= y), val, der)


def test_le_univariate():
    x, y = Dual(1, 11), Dual(2, 2)
    val = True
    der = False
    assert _compare((x <= y), val, der)


def test_le_multivariate():
    x, y = Dual.from_array([6, 4])
    val = False
    der = [False, True]
    assert _compare((x <= y), val, der)


def test_ge_constants():
    x = Dual.constant(2.6)
    y = Dual.constant(1.2)
    val = True
    der = True
    assert _compare((x >= y), val, der)


def test_ge_univariate():
    x, y = Dual(1, 11), Dual(2, -8)
    val = False
    der = True
    assert _compare((x >= y), val, der)


def test_ge_multivariate():
    x, y = Dual.from_array([6, 2])
    val = True
    der = [True, False]
    assert _compare((x >= y), val, der)


def test_eq_constants():
    x = Dual.constant(-6.4)
    y = Dual.constant(3)
    val = False
    der = True
    assert _compare((x == y), val, der)


def test_eq_univariate():
    x, y = Dual(5, -9), Dual(20, -9)
    val = False
    der = True
    assert _compare((x == y), val, der)


def test_eq_multivariate():
    x, y = Dual.from_array([2.8, 2.8])
    val = True
    der = [False, False]
    assert _compare((x == y), val, der)


def test_ne_constants():
    x = Dual.constant(8)
    y = Dual.constant(8)
    val = False
    der = False
    assert _compare((x != y), val, der)


def test_ne_univariate():
    x, y = Dual(7, -6), Dual(7, -6)
    val = False
    der = False
    assert _compare((x != y), val, der)


def test_ne_multivariate():
    x, y = Dual.from_array([1, 2])
    val = True
    der = [True, True]
    assert _compare((x != y), val, der)


if __name__ == "__main__":
    pytest.main([__file__])