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
from astropy.units.quantity_helper.namespace import QuantityNamespace
from astropy.units.quantity_helper.rules import PASSTHROUGH, _cached_plan
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

    def test_dtype_none_keeps_integers(self):
        assert Quantity([1, 2], u.m, dtype=None).dtype == np.int64

    def test_booleans_are_kept(self):
        assert Quantity([True, False]).dtype == bool

    def test_integer_conversion_configuration(self):
        with u.preserve_dtype_by_default():
            assert Quantity([1, 2], u.m).dtype == np.int64
            assert Quantity([1, 2], u.m, dtype=np.inexact).dtype == np.float64

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

        ns = a.__array_namespace__()
        assert type(ns.negative(a)) is MyQuantity
        assert type(ns.multiply(2.0, a)) is MyQuantity
        assert type(2.0 * a) is MyQuantity
        assert type(a + a) is MyQuantity
        assert type(u.s / a) is MyQuantity


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
        assert_quantity(a.mT, to_numpy(a).T, u.m, xp)

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
        b = Quantity(1.0, u.m)
        assert type(b.__array_namespace__().add(b, b).value) is np.ndarray

    def test_pickle(self):
        a = Quantity([1.0, 2.0], u.m)
        result = pickle.loads(pickle.dumps(a))
        assert result.unit == u.m
        assert_array_equal(result.value, a.value)

    def test_repr(self):
        assert repr(Quantity([1.0, 2.0], u.m)) == "<Quantity [1. 2.] m>"
        assert str(Quantity(1.5)) == "1.5"


class TestSetitem:
    def test_numpy(self):
        a = Quantity([1.0, 2.0], u.km)
        a[0] = Quantity(500.0, u.m)

        assert_array_equal(a.value, [0.5, 2.0])

        a[1] = 0

        assert_array_equal(a.value, [0.5, 0.0])

        with pytest.raises(u.UnitConversionError):
            a[0] = 1.0

        with pytest.raises(u.UnitConversionError):
            a[0] = Quantity(1.0, u.s)

    def test_integer_precision(self):
        a = Quantity([1, 2], u.m, dtype=np.int64)
        a[0] = Quantity(300.0, u.cm)
        a[1] = Quantity(4, u.m, dtype=np.int64)

        assert_array_equal(a.value, [3, 4])

        with pytest.raises(TypeError, match="precision loss"):
            a[0] = Quantity(50.0, u.cm)

        assert_array_equal(a.value, [3, 4])

    def test_mutability_follows_backend(self, xp):
        a = q(xp, [1.0, 2.0], u.km)

        try:
            a[0] = Quantity(500.0, u.m)
        except (TypeError, ValueError):
            # Back-ends with immutable arrays (e.g. JAX).
            assert_quantity(a, [1.0, 2.0], u.km, xp)
        else:
            assert_quantity(a, [0.5, 2.0], u.km, xp)

    def test_augmented_assignment_rebinds(self):
        a = Quantity([1.0, 2.0], u.km)
        b = a
        a *= u.s

        assert a.unit == u.km * u.s
        assert b.unit is u.km

    def test_not_hashable(self):
        with pytest.raises(TypeError):
            hash(Quantity(1.0, u.m))


class TestOperators:
    def test_arithmetic(self, xp):
        a = q(xp, [1.0, 2.0], u.km)
        b = q(xp, [500.0, 1000.0], u.m)

        assert_quantity(a + b, [1.5, 3.0], u.km, xp)
        assert_quantity(b - a, [-500.0, -1000.0], u.m, xp)
        assert_quantity(a * b, [500.0, 2000.0], u.km * u.m, xp)
        assert_quantity(a / b, [0.002, 0.002], u.km / u.m, xp)
        assert_quantity(a // b, [2.0, 2.0], u.dimensionless_unscaled, xp)
        assert_quantity(a % b, [0.0, 0.0], u.km, xp)
        assert_quantity(-a, [-1.0, -2.0], u.km, xp)
        assert_quantity(+a, [1.0, 2.0], u.km, xp)
        assert_quantity(abs(-a), [1.0, 2.0], u.km, xp)

        with pytest.raises(u.UnitConversionError):
            a + q(xp, 1.0, u.s)

    def test_matmul(self, xp):
        a = q(xp, [[1.0, 2.0], [3.0, 4.0]], u.m)
        b = q(xp, [1.0, 1.0], u.s)

        assert_quantity(a @ b, [3.0, 7.0], u.m * u.s, xp)

    def test_python_numbers(self, xp):
        a = q(xp, [1.0, 2.0], u.km)

        assert_quantity(2 * a, [2.0, 4.0], u.km, xp)
        assert_quantity(a / 2, [0.5, 1.0], u.km, xp)
        assert_quantity(1 / a, [1.0, 0.5], 1 / u.km, xp)
        # 0, inf and nan can have any unit; other numbers are dimensionless.
        assert_quantity(a + 0, [1.0, 2.0], u.km, xp)

        with pytest.raises(u.UnitConversionError):
            a + 1

        dimensionless = q(xp, [1.0, 2.0], u.m / u.cm)

        assert_quantity(dimensionless + 1, [101.0, 201.0], u.dimensionless_unscaled, xp)

    def test_plain_arrays(self, xp):
        a = q(xp, [1.0, 2.0], u.km)
        plain = xp.asarray([2.0, 3.0])

        assert_quantity(a * plain, [2.0, 6.0], u.km, xp)

        if xp.__name__ != "array_api_strict":
            # array-api-strict arrays raise.
            assert_quantity(plain * a, [2.0, 6.0], u.km, xp)

    def test_units(self, xp):
        a = q(xp, [1.0, 2.0], u.km)

        assert_quantity(a * u.s, [1.0, 2.0], u.km * u.s, xp)
        assert_quantity(a / u.s, [1.0, 2.0], u.km / u.s, xp)
        assert_quantity(u.s * a, [1.0, 2.0], u.s * u.km, xp)
        assert_quantity(u.s / a, [1.0, 0.5], u.s / u.km, xp)

    def test_comparisons(self, xp):
        a = q(xp, [1.0, 2.0], u.km)
        b = q(xp, [1000.0, 1000.0], u.m)

        assert_array_equal(to_numpy(a == b), [True, False])
        assert_array_equal(to_numpy(a != b), [False, True])
        assert_array_equal(to_numpy(a < b), [False, False])
        assert_array_equal(to_numpy(a <= b), [True, False])
        assert_array_equal(to_numpy(a > b), [False, True])
        assert_array_equal(to_numpy(a >= b), [True, True])
        assert (a == q(xp, 1.0, u.s)) is False
        assert (a != q(xp, 1.0, u.s)) is True

        with pytest.raises(u.UnitConversionError):
            a < q(xp, 1.0, u.s)  # noqa: B015

    def test_unsupported_operands(self):
        a = Quantity(1.0, u.m)

        with pytest.raises(TypeError):
            a + "1 m"

        with pytest.raises(TypeError):
            a + u.m

    def test_enabled_equivalencies(self, xp):
        with u.set_enabled_equivalencies(u.temperature()):
            result = q(xp, 0.0, u.deg_C) + q(xp, 10.0, u.K)

        assert_quantity(result, -263.15, u.deg_C, xp)


class TestConversion:
    def test_to(self, xp):
        a = q(xp, [1.0, 2.0], u.km)

        assert_quantity(a.to(u.m), [1000.0, 2000.0], u.m, xp)
        assert_quantity(a << u.m, [1000.0, 2000.0], u.m, xp)
        assert_allclose(to_numpy(a.to_value(u.m)), [1000.0, 2000.0])
        assert a.to_value() is a.value
        assert (a << u.km).value is a.value
        assert a.to(u.km, copy=False).value is a.value

        with pytest.raises(u.UnitConversionError):
            a.to(u.s)

    def test_to_copies(self):
        a = Quantity([1.0, 2.0], u.km)
        b = a.to(u.km)
        b[0] = Quantity(5.0, u.km)

        assert_array_equal(a.value, [1.0, 2.0])

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

        # Only units that are dimensionless without equivalencies.
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            with pytest.raises(TypeError, match="drop the unit"):
                np.asarray(Quantity(1.0, u.rad))

    def test_numpy_ufuncs_do_not_drop_units(self):
        with pytest.raises(TypeError):
            np.add(Quantity(1.0, u.m), 1.0)


class TestNamespace:
    def test_namespace(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()

        assert isinstance(ns, QuantityNamespace)
        assert ns is q(xp, 2.0, u.s).__array_namespace__()

    def test_functions_are_kept(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()

        assert ns.add is ns.add
        assert ns.add.__name__ == "add"

    def test_unknown_names(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()

        with pytest.raises(AttributeError, match="not supported"):
            ns.from_dlpack

        with pytest.raises(AttributeError):
            ns.no_such_function

    def test_mixed_namespaces(self, xp):
        a = q(xp, 1.0, u.m)

        with pytest.raises(TypeError, match="Multiple namespaces"):
            array_api_compat.array_namespace(a, a.value)

    def test_passthrough(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()

        for name in PASSTHROUGH:
            if hasattr(xp, name):
                assert getattr(ns, name) is getattr(xp, name)

    def test_plain_arguments(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()
        result = ns.add(xp.asarray([1.0]), xp.asarray([2.0]))

        assert not isinstance(result, Quantity)
        assert_allclose(to_numpy(result), [3.0])


class TestElementwise:
    def test_unary(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()
        a = q(xp, [1.0, 4.0], u.m)

        assert_quantity(ns.sqrt(a), [1.0, 2.0], u.m**0.5, xp)
        assert_quantity(ns.square(a), [1.0, 16.0], u.m**2, xp)
        assert_quantity(ns.abs(ns.negative(a)), [1.0, 4.0], u.m, xp)
        assert_quantity(ns.sin(q(xp, 90.0, u.deg)), 1.0, u.dimensionless_unscaled, xp)
        assert_quantity(ns.acos(q(xp, 1.0, u.one)), 0.0, u.rad, xp)
        assert_allclose(to_numpy(ns.isnan(a)), [False, False])

        with pytest.raises(u.UnitTypeError):
            ns.exp(a)

    def test_binary(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()
        a = q(xp, [1.0, 2.0], u.km)
        b = q(xp, [3000.0, 1000.0], u.m)

        assert_quantity(ns.add(a, b), [4.0, 3.0], u.km, xp)
        assert_quantity(ns.multiply(a, b), [3000.0, 2000.0], u.km * u.m, xp)
        assert_quantity(ns.maximum(a, b), [3.0, 2.0], u.km, xp)
        assert_quantity(ns.hypot(a, b), [10**0.5, 5**0.5], u.km, xp)
        assert_quantity(ns.atan2(a, b), np.arctan2([1, 2], [3, 1]), u.rad, xp)
        assert_quantity(ns.copysign(a, ns.negative(b)), [-1.0, -2.0], u.km, xp)
        assert_allclose(to_numpy(ns.less(a, b)), [True, False])

        with pytest.raises(u.UnitConversionError):
            ns.add(a, q(xp, 1.0, u.s))

    def test_python_numbers(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()
        a = q(xp, [1.0, 2.0], u.km)

        assert_quantity(ns.multiply(2.0, a), [2.0, 4.0], u.km, xp)
        # 0, inf and nan can have any unit. Other numbers are dimensionless.
        assert_quantity(ns.add(a, 0), [1.0, 2.0], u.km, xp)

        with pytest.raises(u.UnitConversionError):
            ns.add(a, 1.0)

        dimensionless = q(xp, [1.0, 2.0], u.m / u.cm)
        result = ns.add(dimensionless, 1.0)

        assert_quantity(result, [101.0, 201.0], u.dimensionless_unscaled, xp)

    def test_float32_is_kept(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()
        a = Quantity(xp.asarray([1.0, 2.0], dtype=xp.float32), u.km)
        b = Quantity(xp.asarray([1.0, 2.0], dtype=xp.float32), u.m)

        assert ns.add(a, b).dtype == xp.float32

    def test_integer_values_are_scaled_as_float(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()
        a = Quantity(xp.asarray([1, 2]), u.km, dtype=xp.int64)
        result = ns.add(q(xp, 1.0, u.m), a)

        assert_quantity(result, [1001.0, 2001.0], u.m, xp)


class TestRejected:
    def test_reject_quantity(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()

        with pytest.raises(TypeError, match="does not accept quantities"):
            ns.logical_and(q(xp, 1.0, u.m), xp.asarray(True))

        result = ns.logical_and(xp.asarray(True), xp.asarray(False))

        assert not bool(to_numpy(result))

    def test_keywords(self):
        a = Quantity([1.0, 2.0], u.km)
        ns = a.__array_namespace__()
        out = np.zeros(2)

        # Results written to plain arrays would lose the unit.
        with pytest.raises(TypeError):
            ns.add(a, a, out=out)

        with pytest.raises(TypeError, match="'out'"):
            ns.sum(a, out=out[0])

        assert_array_equal(out, [0.0, 0.0])


class TestReductionsAndShapes:
    def test_reductions(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()
        a = q(xp, [[1.0, 2.0], [3.0, 4.0]], u.m)

        assert_quantity(ns.sum(a), 10.0, u.m, xp)
        assert_quantity(ns.mean(a, axis=0), [2.0, 3.0], u.m, xp)
        assert_quantity(ns.max(a), 4.0, u.m, xp)
        assert_quantity(ns.min(a, axis=1, keepdims=True), [[1.0], [3.0]], u.m, xp)
        assert_quantity(ns.std(a), np.std([1, 2, 3, 4]), u.m, xp)
        assert_quantity(ns.cumulative_sum(a, axis=1), [[1, 3], [3, 7]], u.m, xp)
        assert int(to_numpy(ns.argmax(a))) == 3
        assert bool(to_numpy(ns.all(ns.greater(a, 0))))

    def test_shapes(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()
        a = q(xp, [[1.0, 2.0], [3.0, 4.0]], u.m)

        assert_quantity(ns.reshape(a, (4,)), [1, 2, 3, 4], u.m, xp)
        assert_quantity(ns.flip(a, axis=0), [[3, 4], [1, 2]], u.m, xp)
        assert_quantity(ns.expand_dims(a, axis=0), [[[1, 2], [3, 4]]], u.m, xp)
        assert_quantity(ns.matrix_transpose(a), [[1, 3], [2, 4]], u.m, xp)
        assert_quantity(ns.sort(-a, axis=1), [[-2, -1], [-4, -3]], u.m, xp)
        assert_quantity(ns.take(a, xp.asarray([1]), axis=0), [[3, 4]], u.m, xp)
        assert_allclose(to_numpy(ns.argsort(-a, axis=1)), [[1, 0], [1, 0]])

        with pytest.raises(TypeError, match="does not accept quantities"):
            ns.take(a, q(xp, [0], u.m))

    def test_plain_arrays(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()
        result = ns.sum(xp.asarray([1.0, 2.0]))

        assert not isinstance(result, Quantity)
        assert float(to_numpy(result)) == 3.0

    def test_subclass(self, xp):
        a = MyQuantity(xp.asarray([[1.0, 2.0]]), u.m)
        ns = a.__array_namespace__()

        assert type(ns.sum(a)) is MyQuantity
        assert type(ns.reshape(a, (2,))) is MyQuantity
        assert type(a.mT) is MyQuantity


class TestEquivalencies:
    def test_enabled_equivalencies(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()

        with u.set_enabled_equivalencies(u.temperature()):
            result = ns.add(q(xp, 0.0, u.deg_C), q(xp, 10.0, u.K))

        assert_quantity(result, -263.15, u.deg_C, xp)

    def test_results_are_not_cached(self, xp):
        ns = q(xp, 1.0, u.m).__array_namespace__()
        _cached_plan.cache_clear()

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            assert_quantity(ns.exp(q(xp, 0.0, u.rad)), 1.0, u.one, xp)

        assert _cached_plan.cache_info().currsize == 0

        with pytest.raises(u.UnitTypeError):
            ns.exp(q(xp, 0.0, u.rad))
