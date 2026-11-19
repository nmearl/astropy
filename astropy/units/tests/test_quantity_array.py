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
from astropy.units.quantity_helper import rules
from astropy.units.quantity_helper.namespace import QuantityNamespace, units_namespace
from astropy.units.quantity_helper.rules import PASSTHROUGH
from astropy.utils.compat.array_api import array_namespace
from astropy.utils.compat.optional_deps import HAS_ARRAY_API_STRICT

BACKENDS = [
    "numpy",
    pytest.param(
        "array_api_strict",
        marks=pytest.mark.skipif(
            not HAS_ARRAY_API_STRICT, reason="array-api-strict not installed"
        ),
    ),
    pytest.param(
        "jax",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("jax") is None, reason="jax not installed"
        ),
    ),
]


@pytest.fixture(params=BACKENDS)
def xp(request):
    if request.param == "numpy":
        yield array_api_compat.numpy
    elif request.param == "array_api_strict":
        import array_api_strict

        yield array_api_strict
    else:
        import jax

        # Double precision, as for the other back-ends, for these tests only.
        enabled = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)

        try:
            yield jax.numpy
        finally:
            jax.config.update("jax_enable_x64", enabled)


@pytest.fixture
def ns(xp):
    return units_namespace(xp)


def to_numpy(x):
    """A plain array as a numpy array."""
    # Quantities would be unwrapped, hiding results that should be plain.
    assert not isinstance(x, Quantity), "expected a plain array"

    return np.from_dlpack(x) if hasattr(x, "__dlpack__") else np.asarray(x)


def q(xp, value, unit):
    return Quantity(xp.asarray(value, dtype=xp.float64), unit)


def assert_quantity(result, value, unit, xp):
    assert isinstance(result, Quantity)
    assert result.unit == unit
    assert array_api_compat.array_namespace(result.value) is (
        array_api_compat.array_namespace(xp.asarray(0.0))
    )

    assert_allclose(to_numpy(result.value), value)


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
        value = to_numpy(a.value)
        assert_quantity(a.T, value.T, u.m, xp)
        assert_quantity(a.to_device(a.device), value, u.m, xp)
        assert_quantity(a.mT, value.T, u.m, xp)

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
        assert [float(to_numpy(x.value)) for x in a] == [1.0, 2.0, 3.0]
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

        if xp.__name__ == "jax.numpy":
            # JAX arrays are immutable.
            with pytest.raises(TypeError):
                a[0] = q(xp, 500.0, u.m)

            assert_quantity(a, [1.0, 2.0], u.km, xp)
        else:
            a[0] = q(xp, 500.0, u.m)

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
        assert_quantity(a**2, [1.0, 4.0], u.km**2, xp)

        with pytest.raises(u.UnitConversionError):
            a + q(xp, 1.0, u.s)

    def test_matmul(self, xp):
        a = q(xp, [[1.0, 2.0], [3.0, 4.0]], u.m)
        b = q(xp, [1.0, 1.0], u.s)

        assert_quantity(a @ b, [3.0, 7.0], u.m * u.s, xp)

    @pytest.mark.parametrize(
        "name", ["__add__", "__rsub__", "__le__", "__neg__", "__abs__"]
    )
    def test_names(self, name):
        assert getattr(Quantity, name).__name__ == name

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

        assert_quantity(result, 299792458.0 / to_numpy(a.value), u.Hz, xp)

        with u.set_enabled_equivalencies(u.spectral()):
            assert_quantity(a.to(u.Hz), 299792458.0 / to_numpy(a.value), u.Hz, xp)

            with pytest.raises(u.UnitConversionError):
                a.to(u.Hz, equivalencies=None)

    def test_decompose(self, xp):
        a = q(xp, [1.0, 2.0], u.km / u.ms)
        result = a.decompose()

        assert_quantity(result, [1e6, 2e6], u.m / u.s, xp)

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
    def test_namespace(self, xp, ns):
        assert isinstance(ns, QuantityNamespace)
        assert q(xp, 1.0, u.m).__array_namespace__() is ns
        assert q(xp, 2.0, u.s).__array_namespace__() is ns

    def test_function_names(self, ns):
        assert ns.add.__name__ == "add"
        assert ns.sum.__name__ == "sum"

    def test_unknown_names(self, xp, ns):
        with pytest.raises(AttributeError, match="not supported"):
            ns.from_dlpack

        with pytest.raises(AttributeError):
            ns.no_such_function

    def test_mixed_namespaces(self, xp):
        a = q(xp, 1.0, u.m)

        with pytest.raises(TypeError, match="Multiple namespaces"):
            array_api_compat.array_namespace(a, a.value)

    def test_passthrough(self, xp, ns):
        for name in PASSTHROUGH:
            if hasattr(xp, name):
                assert getattr(ns, name) is getattr(xp, name)

    def test_plain_arguments(self, xp, ns):
        result = ns.add(xp.asarray([1.0]), xp.asarray([2.0]))

        assert not isinstance(result, Quantity)
        assert_allclose(to_numpy(result), [3.0])


class TestElementwise:
    def test_unary(self, xp, ns):
        a = q(xp, [1.0, 4.0], u.m)

        assert_quantity(ns.sqrt(a), [1.0, 2.0], u.m**0.5, xp)
        assert_quantity(ns.square(a), [1.0, 16.0], u.m**2, xp)
        assert_quantity(ns.abs(ns.negative(a)), [1.0, 4.0], u.m, xp)
        assert_quantity(ns.sin(q(xp, 90.0, u.deg)), 1.0, u.dimensionless_unscaled, xp)
        assert_quantity(ns.acos(q(xp, 1.0, u.one)), 0.0, u.rad, xp)
        assert_allclose(to_numpy(ns.isnan(a)), [False, False])

        with pytest.raises(u.UnitTypeError):
            ns.exp(a)

    def test_binary(self, xp, ns):
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

    def test_python_numbers(self, xp, ns):
        a = q(xp, [1.0, 2.0], u.km)

        assert_quantity(ns.multiply(2.0, a), [2.0, 4.0], u.km, xp)
        # 0, inf and nan can have any unit. Other numbers are dimensionless.
        assert_quantity(ns.add(a, 0), [1.0, 2.0], u.km, xp)

        with pytest.raises(u.UnitConversionError):
            ns.add(a, 1.0)

        dimensionless = q(xp, [1.0, 2.0], u.m / u.cm)
        result = ns.add(dimensionless, 1.0)

        assert_quantity(result, [101.0, 201.0], u.dimensionless_unscaled, xp)

    def test_float32_is_kept(self, xp, ns):
        a = Quantity(xp.asarray([1.0, 2.0], dtype=xp.float32), u.km)
        b = Quantity(xp.asarray([1.0, 2.0], dtype=xp.float32), u.m)

        assert ns.add(a, b).dtype == xp.float32

    def test_integer_values_are_scaled_as_float(self, xp, ns):
        a = Quantity(xp.asarray([1, 2]), u.km, dtype=xp.int64)
        result = ns.add(q(xp, 1.0, u.m), a)

        assert_quantity(result, [1001.0, 2001.0], u.m, xp)


class TestRejected:
    def test_reject_quantity(self, xp, ns):
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
    def test_reductions(self, xp, ns):
        a = q(xp, [[1.0, 2.0], [3.0, 4.0]], u.m)

        assert_quantity(ns.sum(a), 10.0, u.m, xp)
        assert_quantity(ns.mean(a, axis=0), [2.0, 3.0], u.m, xp)
        assert_quantity(ns.max(a), 4.0, u.m, xp)
        assert_quantity(ns.min(a, axis=1, keepdims=True), [[1.0], [3.0]], u.m, xp)
        assert_quantity(ns.std(a), np.std([1, 2, 3, 4]), u.m, xp)
        assert_quantity(ns.cumulative_sum(a, axis=1), [[1, 3], [3, 7]], u.m, xp)
        assert int(to_numpy(ns.argmax(a))) == 3
        assert bool(to_numpy(ns.all(ns.greater(a, 0))))

    def test_shapes(self, xp, ns):
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

    def test_plain_arrays(self, xp, ns):
        result = ns.sum(xp.asarray([1.0, 2.0]))

        assert not isinstance(result, Quantity)
        assert float(to_numpy(result)) == 3.0

    def test_subclass(self, xp):
        a = MyQuantity(xp.asarray([[1.0, 2.0]]), u.m)
        ns = a.__array_namespace__()

        assert type(ns.sum(a)) is MyQuantity
        assert type(ns.reshape(a, (2,))) is MyQuantity
        assert type(a.mT) is MyQuantity


class TestUnitFromArguments:
    def test_pow(self, xp, ns):
        a = q(xp, [1.0, 2.0], u.m)

        assert_quantity(ns.pow(a, 3), [1.0, 8.0], u.m**3, xp)
        assert_quantity(ns.pow(a, np.float64(0.5)), [1.0, 2**0.5], u.m**0.5, xp)
        assert_quantity(a**-1, [1.0, 0.5], 1 / u.m, xp)

        dimensionless = q(xp, [1.0, 2.0], u.m / u.cm)
        result = ns.pow(dimensionless, xp.asarray([1.0, 2.0]))

        assert_quantity(result, [100.0, 40000.0], u.one, xp)
        assert_quantity(2.0**dimensionless, [2.0**100, 2.0**200], u.one, xp)

        with pytest.raises(u.UnitTypeError, match="Python number"):
            ns.pow(a, xp.asarray([1.0, 2.0]))

        with pytest.raises(u.UnitTypeError, match="exponent of 'pow' must be"):
            ns.pow(dimensionless, q(xp, [1.0, 2.0], u.m))

    def test_prod(self, xp, ns):
        a = q(xp, [[1.0, 2.0], [3.0, 4.0]], u.m)

        assert_quantity(ns.prod(a), 24.0, u.m**4, xp)
        assert_quantity(ns.prod(a, axis=1), [2.0, 12.0], u.m**2, xp)
        assert_quantity(ns.prod(a, axis=(0, 1)), 24.0, u.m**4, xp)

        dimensionless = q(xp, [1.0, 2.0], u.m / u.cm)

        assert_quantity(ns.cumulative_prod(dimensionless), [100, 20000], u.one, xp)

        with pytest.raises(u.UnitTypeError):
            ns.cumulative_prod(a)

    @pytest.mark.parametrize("size", [None, np.nan])
    def test_prod_unknown_size(self, size):
        class Lazy:
            # Just enough of a lazy array with an unknown size for prod.
            shape = (size,)

        ns = Quantity(1.0, u.m).__array_namespace__()

        with pytest.raises(u.UnitsError, match="not known"):
            ns.prod(Quantity._new(Lazy(), u.m))

    def test_var(self, xp, ns):
        a = q(xp, [1.0, 2.0, 3.0, 4.0], u.m)

        assert_quantity(ns.var(a), np.var([1, 2, 3, 4]), u.m**2, xp)
        assert_quantity(
            ns.var(a, correction=1), np.var([1, 2, 3, 4], ddof=1), u.m**2, xp
        )

    def test_diff(self, xp, ns):
        a = q(xp, [1.0, 2.0, 4.0], u.m)

        assert_quantity(ns.diff(a), [1.0, 2.0], u.m, xp)

        result = ns.diff(a, prepend=q(xp, [0.0], u.cm), append=q(xp, [0.005], u.km))

        assert_quantity(result, [1.0, 1.0, 2.0, 1.0], u.m, xp)

        with pytest.raises(u.UnitConversionError):
            ns.diff(a, append=q(xp, [1.0], u.s))

    def test_clip(self, xp, ns):
        a = q(xp, [1.0, 2.0, 3.0], u.m)

        assert_quantity(ns.clip(a, min=q(xp, 150.0, u.cm)), [1.5, 2.0, 3.0], u.m, xp)
        assert_quantity(ns.clip(a, max=q(xp, 2.0, u.m)), [1.0, 2.0, 2.0], u.m, xp)

        with pytest.raises(u.UnitConversionError):
            ns.clip(a, min=1.0)

    def test_tensordot_and_vecdot(self, xp, ns):
        a = q(xp, [[1.0, 2.0], [3.0, 4.0]], u.m)
        b = q(xp, [[1.0, 0.0], [0.0, 1.0]], u.s)

        assert_quantity(ns.tensordot(a, b), 5.0, u.m * u.s, xp)
        assert_quantity(ns.tensordot(a, b, axes=1), [[1, 2], [3, 4]], u.m * u.s, xp)
        assert_quantity(ns.vecdot(a, b), [1.0, 4.0], u.m * u.s, xp)
        assert_quantity(ns.vecdot(a, b, axis=-2), [1.0, 4.0], u.m * u.s, xp)
        assert_quantity(ns.vecdot(a, xp.asarray([1.0, 1.0])), [3, 7], u.m, xp)

    def test_where(self, xp, ns):
        a = q(xp, [1.0, 2.0], u.m)
        condition = xp.asarray([True, False])

        assert_quantity(ns.where(condition, a, q(xp, 0.5, u.km)), [1.0, 500.0], u.m, xp)
        assert_quantity(ns.where(condition, a, 0.0), [1.0, 0.0], u.m, xp)

        with pytest.raises(TypeError, match="does not accept quantities"):
            ns.where(q(xp, [1.0, 0.0], u.one), a, a)

        with pytest.raises(u.UnitConversionError):
            ns.where(condition, a, q(xp, 1.0, u.s))


class TestCombining:
    def test_concat_and_stack(self, xp, ns):
        a = q(xp, [1.0, 2.0], u.km)
        b = q(xp, [500.0], u.m)

        assert_quantity(ns.concat([a, b]), [1.0, 2.0, 0.5], u.km, xp)
        assert_quantity(ns.concat((b, a)), [500.0, 1000.0, 2000.0], u.m, xp)
        assert_quantity(ns.stack([a, a], axis=1), [[1, 1], [2, 2]], u.km, xp)

        with pytest.raises(u.UnitConversionError):
            ns.concat([a, q(xp, [1.0], u.s)])

    def test_unstack(self, xp, ns):
        first, second = ns.unstack(q(xp, [[1.0, 2.0], [3.0, 4.0]], u.km))

        assert_quantity(first, [1.0, 2.0], u.km, xp)
        assert_quantity(second, [3.0, 4.0], u.km, xp)

    def test_broadcast_arrays_and_meshgrid(self, xp, ns):
        a = q(xp, [1.0, 2.0], u.km)
        b = MyQuantity(xp.asarray([[3.0]]), u.s)
        plain = xp.asarray([5.0])
        x, y, z = ns.broadcast_arrays(a, b, plain)

        assert_quantity(x, [[1.0, 2.0]], u.km, xp)
        assert_quantity(y, [[3.0, 3.0]], u.s, xp)
        assert not isinstance(z, Quantity)
        # Each result has the class of its own input.
        assert type(x) is Quantity
        assert type(y) is MyQuantity

        x, y = ns.meshgrid(a, q(xp, [1.0, 2.0, 3.0], u.s))

        assert x.shape == (3, 2)
        assert (x.unit, y.unit) == (u.km, u.s)


class TestSearchingAndSets:
    def test_searchsorted(self, xp, ns):
        a = q(xp, [1.0, 2.0, 3.0], u.km)
        result = ns.searchsorted(a, q(xp, [1500.0, 2500.0], u.m))

        assert not isinstance(result, Quantity)
        assert_array_equal(to_numpy(result), [1, 2])

        with pytest.raises(u.UnitConversionError):
            ns.searchsorted(a, q(xp, [1.0], u.s))

    def test_isin(self, xp, ns):
        if not hasattr(xp, "isin"):
            pytest.skip("isin not available")

        # array-api-strict only allows integers for isin.
        a = Quantity(xp.asarray([1, 2]), u.km, dtype=None)
        result = ns.isin(a, Quantity(xp.asarray([2]), u.km, dtype=None))

        assert_array_equal(to_numpy(result), [False, True])

        with pytest.raises(u.UnitConversionError):
            ns.isin(a, Quantity(xp.asarray([2]), u.s, dtype=None))

        if xp.__name__ != "array_api_strict":
            result = ns.isin(q(xp, [1.0, 2.0], u.km), q(xp, [1000.0], u.m))

            assert_array_equal(to_numpy(result), [True, False])

    def test_unique(self, xp, ns):
        a = q(xp, [2.0, 1.0, 2.0], u.m)

        assert_quantity(ns.unique_values(a), [1.0, 2.0], u.m, xp)

        result = ns.unique_counts(a)

        assert_quantity(result.values, [1.0, 2.0], u.m, xp)
        assert_array_equal(to_numpy(result.counts), [1, 2])

        result = ns.unique_inverse(a)

        assert_quantity(result.values, [1.0, 2.0], u.m, xp)
        assert_array_equal(to_numpy(result.inverse_indices), [1, 0, 1])
        assert ns.unique_all(a).values.unit is u.m


class TestDtypeFunctions:
    def test_dtype_functions(self, xp, ns):
        a = q(xp, 1.0, u.m)

        assert ns.result_type(a, xp.float32) == xp.result_type(a.value, xp.float32)
        assert ns.can_cast(a.dtype, xp.float32) == xp.can_cast(xp.float64, xp.float32)
        assert ns.finfo(a).eps == xp.finfo(xp.float64).eps
        assert ns.iinfo(xp.int32).max == 2**31 - 1


class TestCreation:
    def test_asarray(self, xp, ns):
        assert_quantity(ns.asarray([1.0, 2.0], unit=u.m), [1.0, 2.0], u.m, xp)
        assert_quantity(ns.asarray(q(xp, 1.0, u.km), unit=u.m), 1000.0, u.m, xp)
        assert_quantity(ns.asarray(q(xp, 1.0, u.km)), 1.0, u.km, xp)
        # Without a unit, plain values give plain arrays, e.g. for masks.
        mask = ns.asarray([True, False])

        assert not isinstance(mask, Quantity)
        assert mask.dtype == xp.bool

    def test_from_shape(self, xp, ns):
        assert_quantity(ns.zeros(2, unit=u.s), [0.0, 0.0], u.s, xp)
        assert_quantity(ns.ones((1, 2), unit=u.s), [[1.0, 1.0]], u.s, xp)
        assert_quantity(ns.eye(2, unit=u.s), np.eye(2), u.s, xp)
        assert ns.empty((2, 3), unit=u.s).shape == (2, 3)
        assert not isinstance(ns.zeros(2), Quantity)

    def test_full(self, xp, ns):
        assert_quantity(ns.full(2, q(xp, 1.0, u.km)), [1.0, 1.0], u.km, xp)
        assert_quantity(ns.full(2, q(xp, 1.0, u.km), unit=u.m), [1e3, 1e3], u.m, xp)
        # A plain fill value is in the unit given.
        assert_quantity(ns.full(2, 3.0, unit=u.m), [3.0, 3.0], u.m, xp)
        assert not isinstance(ns.full(2, 3.0), Quantity)

        if xp.__name__ != "jax.numpy":
            # Results from array fill values can be written to.
            result = ns.full((2, 2), q(xp, 1.0, u.km))
            result[0, 0] = q(xp, 2.0, u.km)

            assert_quantity(result, [[2.0, 1.0], [1.0, 1.0]], u.km, xp)

    def test_full_boolean(self, xp, ns):
        fill = Quantity(xp.asarray(True), u.one, dtype=None)
        result = ns.full(2, fill)

        assert result.unit is u.one
        assert result.dtype == xp.bool
        assert_array_equal(to_numpy(result.value), [True, True])

    def test_like(self, xp, ns):
        a = q(xp, [1.0, 2.0], u.km)

        assert_quantity(ns.zeros_like(a), [0.0, 0.0], u.km, xp)
        assert_quantity(ns.ones_like(a, unit=u.s), [1.0, 1.0], u.s, xp)
        assert ns.empty_like(a).unit is u.km
        assert not isinstance(ns.zeros_like(xp.asarray([1.0])), Quantity)
        assert type(ns.zeros_like(MyQuantity(xp.asarray([1.0]), u.m))) is MyQuantity

    def test_full_like(self, xp, ns):
        a = q(xp, [1.0, 2.0], u.km)

        assert_quantity(ns.full_like(a, q(xp, 500.0, u.m)), [0.5, 0.5], u.km, xp)
        assert_quantity(ns.full_like(a, 0), [0.0, 0.0], u.km, xp)
        assert_quantity(ns.full_like(a, 2.0, unit=u.s), [2.0, 2.0], u.s, xp)

        plain = xp.asarray([1.0, 2.0])

        assert_quantity(ns.full_like(plain, q(xp, 1.0, u.m)), [1.0, 1.0], u.m, xp)

        # As in arithmetic, other plain numbers are dimensionless.
        with pytest.raises(u.UnitConversionError):
            ns.full_like(a, 2.0)

    def test_arange(self, xp, ns):
        result = ns.arange(0.0, q(xp, 3.0, u.km), q(xp, 1000.0, u.m))

        assert_quantity(result, [0.0, 1.0, 2.0], u.km, xp)
        assert_quantity(ns.arange(3.0, unit=u.s), [0.0, 1.0, 2.0], u.s, xp)
        # Without a step, it is 1 in the unit of the result.
        assert_quantity(ns.arange(q(xp, 3.0, u.m)), [0.0, 1.0, 2.0], u.m, xp)
        assert not isinstance(ns.arange(3), Quantity)

        with pytest.raises(u.UnitConversionError):
            ns.arange(q(xp, 3.0, u.m), step=1.0)

    def test_arange_keeps_integers(self):
        ns = Quantity(1.0, u.m).__array_namespace__()
        result = ns.arange(Quantity(np.int64(3), u.m, dtype=None))

        assert result.dtype == np.int64

    def test_linspace(self, xp, ns):
        result = ns.linspace(q(xp, 1.0, u.m), q(xp, 1.0, u.km), 3)

        assert_quantity(result, [1.0, 500.5, 1000.0], u.m, xp)
        assert_quantity(ns.linspace(0.0, 1.0, 3, unit=u.s), [0, 0.5, 1], u.s, xp)

        with pytest.raises(u.UnitConversionError):
            ns.linspace(q(xp, 1.0, u.m), q(xp, 1.0, u.s), 3)


# Names the namespace handles, from the rule tables and its methods.
HANDLED = (
    set(rules.UNARY_RULES)
    | set(rules.BINARY_RULES)
    | rules.REJECT_QUANTITY
    | rules.REDUCTION_SAME
    | rules.REDUCTION_PLAIN
    | rules.SHAPE_SAME
    | rules.SHAPE_PLAIN
    | PASSTHROUGH
    | {name for name in vars(QuantityNamespace) if not name.startswith("_")}
)
# Names of the standard that are deliberately not supported.
UNSUPPORTED = {"from_dlpack", "fft", "linalg"}


def test_standard_coverage():
    # array-api-strict has the names of the standard and few others.
    xp = pytest.importorskip("array_api_strict")
    names = {name for name in dir(xp) if not name.startswith("_")}
    names -= {"ArrayAPIStrictFlags", "Device", "ModuleType"}
    names -= {name for name in names if "array_api_strict_flags" in name}
    missing = names - HANDLED - UNSUPPORTED

    assert not missing, f"not handled: {sorted(missing)}"


def test_handled_names_resolve(xp, ns):
    for name in HANDLED:
        if hasattr(xp, name):
            getattr(ns, name)


def _generic(x, mask_values):
    """Code written against the standard only, with no knowledge of units."""
    xp = array_namespace(x)
    mask = xp.asarray(mask_values)
    centred = x - xp.mean(x)
    spread = xp.sqrt(xp.sum(centred**2) / x.shape[0])
    grid = xp.linspace(xp.min(x), xp.max(x), 3)

    return xp.where(mask, centred, xp.zeros_like(x)), spread, grid


class TestGenericCode:
    def test_namespace_lookup(self, xp):
        a = q(xp, [1.0, 2.0], u.m)

        assert array_namespace(a) is a.__array_namespace__()
        assert array_api_compat.array_namespace(a) is a.__array_namespace__()

    def test_expression(self, xp):
        a = q(xp, [1.0, 2.0, 3.0], u.km)
        result, spread, grid = _generic(a, [True, False, True])

        assert_quantity(result, [-1.0, 0.0, 1.0], u.km, xp)
        assert_quantity(spread, (2 / 3) ** 0.5, u.km, xp)
        assert_quantity(grid, [1.0, 2.0, 3.0], u.km, xp)

    def test_same_code_on_plain_arrays(self, xp):
        result, spread, grid = _generic(
            xp.asarray([1.0, 2.0, 3.0]), [True, False, True]
        )

        assert not isinstance(result, Quantity)
        assert_allclose(to_numpy(result), [-1.0, 0.0, 1.0])


class TestEquivalencies:
    def test_enabled_equivalencies(self, xp, ns):
        with u.set_enabled_equivalencies(u.temperature()):
            result = ns.add(q(xp, 0.0, u.deg_C), q(xp, 10.0, u.K))

        assert_quantity(result, -263.15, u.deg_C, xp)

    def test_results_are_not_reused_outside_context(self, xp, ns):
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            assert_quantity(ns.exp(q(xp, 0.0, u.rad)), 1.0, u.one, xp)

        with pytest.raises(u.UnitTypeError):
            ns.exp(q(xp, 0.0, u.rad))
