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

    def test_from_quantity(self):
        km = Quantity([1.0], u.km)

        assert Quantity(km).unit is u.km
        assert_array_equal(Quantity(km, u.m).value, [1000.0])

    def test_from_legacy_quantity(self):
        result = Quantity(u.Quantity([1.0, 2.0], u.km), u.m)

        assert type(result.value) is np.ndarray
        assert_array_equal(result.value, [1000.0, 2000.0])

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

        for result in (a[0, 1], a.T, a.to_device(a.device), a.to(u.cm), a.decompose()):
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


class TestConversion:
    def test_to(self, xp):
        a = q(xp, [1.0, 2.0], u.km)

        assert_quantity(a.to(u.m), [1000.0, 2000.0], u.m, xp)
        assert_quantity(a << u.m, [1000.0, 2000.0], u.m, xp)
        assert_allclose(to_numpy(a.to_value(u.m)), [1000.0, 2000.0])
        assert a.to_value() is a.value
        assert a.to(u.km).value is a.value

        with pytest.raises(u.UnitConversionError):
            a.to(u.s)

    def test_to_keeps_dtype(self, xp):
        a = Quantity(xp.asarray([1.0, 2.0], dtype=xp.float32), u.km)

        assert a.to(u.m).dtype == xp.float32

    def test_integer_values_are_scaled_as_float(self, xp):
        a = Quantity(xp.asarray([1, 2]), u.km, dtype=xp.int64)
        result = a.to(u.m)

        assert xp.isdtype(result.dtype, "real floating")
        assert_quantity(result, [1000.0, 2000.0], u.m, xp)

    def test_to_with_equivalencies(self, xp):
        a = q(xp, [1.0, 2.0], u.m)
        result = a.to(u.Hz, equivalencies=u.spectral())

        assert_quantity(result, 299792458.0 / to_numpy(a), u.Hz, xp)

        with u.set_enabled_equivalencies(u.spectral()):
            assert_quantity(a.to(u.Hz), 299792458.0 / to_numpy(a), u.Hz, xp)

            with pytest.raises(u.UnitConversionError):
                a.to(u.Hz, equivalencies=None)

    def test_decompose(self, xp):
        a = q(xp, [1.0, 2.0], u.km / u.ms)
        result = a.decompose()

        assert result.unit == u.m / u.s
        assert_allclose(to_numpy(result), [1e6, 2e6])

    def test_python_scalars(self):
        assert float(Quantity(3.0, u.m / u.cm)) == 300.0
        assert int(Quantity(3.0)) == 3
        assert complex(Quantity(3.0)) == 3 + 0j
        assert [10, 20, 30][Quantity(1, dtype=np.int64)] == 20

        with pytest.raises(TypeError, match="dimensionless"):
            float(Quantity(3.0, u.m))

        with pytest.raises(TypeError, match="index"):
            [1, 2][Quantity(1.0, u.m)]

        with pytest.raises(ValueError, match="truthiness"):
            bool(Quantity(1.0))

    def test_numpy_conversion(self):
        assert_array_equal(np.asarray(Quantity([1.0, 2.0], u.m / u.cm)), [100, 200])

        with pytest.raises(TypeError, match="drop the unit"):
            np.asarray(Quantity([1.0, 2.0], u.m))

    def test_numpy_ufuncs_do_not_drop_units(self):
        with pytest.raises(TypeError):
            np.add(Quantity(1.0, u.m), 1.0)
