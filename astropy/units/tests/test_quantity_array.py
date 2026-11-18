# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests of the Array API quantity and its namespace, for all back-ends."""

import importlib.util
import pickle

import array_api_compat
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astropy import units as u
from astropy.units._quantity_array import Quantity
from astropy.utils.compat.optional_deps import HAS_ARRAY_API_STRICT


def _numpy():
    return array_api_compat.numpy


def _strict():
    import array_api_strict

    return array_api_strict


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)

    return jax.numpy


BACKENDS = [
    pytest.param(_numpy, id="numpy"),
    pytest.param(
        _strict,
        id="array_api_strict",
        marks=pytest.mark.skipif(
            not HAS_ARRAY_API_STRICT, reason="array-api-strict not installed"
        ),
    ),
    pytest.param(
        _jax,
        id="jax",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("jax") is None, reason="jax not installed"
        ),
    ),
]


@pytest.fixture(params=BACKENDS)
def xp(request):
    return request.param()


def to_numpy(x):
    """A plain array, or the value of a quantity, as a numpy array."""
    if isinstance(x, Quantity):
        x = x.value

    return np.from_dlpack(x) if hasattr(x, "__dlpack__") else np.asarray(x)


def q(xp, value, unit):
    return Quantity(xp.asarray(value, dtype=xp.float64), unit)


def assert_quantity(result, value, unit, xp):
    assert isinstance(result, Quantity)
    assert result.unit == unit
    assert array_api_compat.array_namespace(result.value) is (
        array_api_compat.array_namespace(xp.asarray(0.0))
    )

    assert_allclose(to_numpy(result), value)


class MyQuantity(Quantity):
    __slots__ = ()


class TestConstruction:
    def test_python_values_are_numpy(self):
        result = Quantity([1, 2], u.m)

        assert isinstance(result.value, np.ndarray)
        assert result.dtype == np.float64  # integers become floating point
        assert result.unit is u.m

    def test_explicit_dtype_is_kept(self):
        assert Quantity([1, 2], u.m, dtype=np.int32).dtype == np.int32

    def test_scalar_is_0d_array(self):
        result = Quantity(1.5, u.s)

        assert isinstance(result.value, np.ndarray)
        assert result.shape == ()

    def test_default_unit(self):
        assert Quantity(1.0).unit is u.dimensionless_unscaled

    def test_unit_string(self):
        assert Quantity(1.0, "km/s").unit == u.km / u.s

    def test_copy(self):
        value = np.array([1.0, 2.0])

        assert Quantity(value, u.m).value is not value

        result = Quantity(value, u.m, copy=None)

        assert np.shares_memory(result.value, value)

    def test_function_units_are_not_supported(self):
        with pytest.raises(NotImplementedError):
            Quantity(1.0, u.dex(u.cm / u.s**2))

    def test_array_backends(self, xp):
        value = xp.asarray([1, 2])
        result = Quantity(value, u.m)

        assert_quantity(result, [1.0, 2.0], u.m, xp)
        assert xp.isdtype(result.dtype, "real floating")

    def test_subclass(self, xp):
        a = MyQuantity(xp.asarray([[1.0, 2.0]]), u.m)

        for result in (a[0, 1], a.T, a.to_device(a.device)):
            assert type(result) is MyQuantity


class TestAttributes:
    def test_array_attributes(self, xp):
        a = q(xp, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], u.m)
        assert a.shape == (2, 3)
        assert a.ndim == 2
        assert a.size == 6
        assert a.dtype == xp.float64
        assert a.device == array_api_compat.device(a.value)
        assert_quantity(a.T, to_numpy(a).T, u.m, xp)
        assert_quantity(a.to_device(a.device), to_numpy(a), u.m, xp)

    def test_value_and_unit_are_read_only(self):
        a = Quantity(1.0, u.m)
        with pytest.raises(AttributeError):
            a.unit = u.s
        with pytest.raises(AttributeError):
            a.value = np.array(2.0)

    def test_indexing_and_iteration(self, xp):
        a = q(xp, [1.0, 2.0, 3.0], u.m)
        assert_quantity(a[1], 2.0, u.m, xp)
        assert_quantity(a[1:], [2.0, 3.0], u.m, xp)
        assert len(a) == 3
        assert [float(to_numpy(x)) for x in a] == [1.0, 2.0, 3.0]
        with pytest.raises(TypeError, match="no len"):
            len(a[0])
        with pytest.raises(TypeError, match="not iterable"):
            iter(a[0])

    def test_numpy_scalars_become_0d_arrays(self):
        a = Quantity([1.0, 2.0], u.m)
        assert type(a[0].value) is np.ndarray

    def test_pickle(self):
        a = Quantity([1.0, 2.0], u.m)
        result = pickle.loads(pickle.dumps(a))
        assert result.unit == u.m
        assert_array_equal(result.value, a.value)

    def test_repr(self):
        assert repr(Quantity([1.0, 2.0], u.m)) == "<Quantity [1. 2.] m>"
        assert str(Quantity(1.5)) == "1.5"
