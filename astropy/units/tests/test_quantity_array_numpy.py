# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests of numpy functions on the Array API quantity with numpy arrays as values."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astropy import units as u
from astropy.units._quantity_array import Quantity
from astropy.utils.compat.optional_deps import HAS_ARRAY_API_STRICT

pytestmark = pytest.mark.numpy_compat


class MyQuantity(Quantity):
    pass


class TestUfuncsWithRules:
    """Ufuncs with a rule give the same results as the namespace functions."""

    def test_same_as_operators(self):
        a = Quantity([1.0, 2.0], u.m)
        b = Quantity([10.0, 20.0], u.cm)

        result = np.add(a, b)

        assert type(result) is Quantity
        assert result.unit is u.m
        assert_allclose(result.value, [1.1, 2.2])
        assert_array_equal((a + b).value, result.value)

        result = np.multiply(a, b)

        assert result.unit == u.m * u.cm
        assert_allclose(result.value, [10.0, 40.0])

    def test_one_argument(self):
        result = np.sin(Quantity([30.0, 90.0], u.deg))

        assert result.unit is u.dimensionless_unscaled
        assert_allclose(result.value, [0.5, 1.0])

        result = np.sqrt(Quantity([4.0, 9.0], u.m**2))

        assert result.unit == u.m
        assert_allclose(result.value, [2.0, 3.0])

    def test_comparison_gives_plain_array(self):
        result = np.less(Quantity([1.0, 2.0], u.m), Quantity([150.0, 150.0], u.cm))

        assert type(result) is np.ndarray
        assert_array_equal(result, [True, False])

    def test_power(self):
        result = np.power(Quantity([1.0, 2.0], u.m), 3)

        assert result.unit == u.m**3
        assert_allclose(result.value, [1.0, 8.0])

        result = np.power(2.0, Quantity([100.0, 200.0], u.cm / u.m))

        assert result.unit is u.dimensionless_unscaled
        assert_allclose(result.value, [2.0, 4.0])

        with pytest.raises(u.UnitTypeError, match="Python number"):
            np.power(Quantity([1.0, 2.0], u.m), np.array([2.0, 3.0]))

        with pytest.raises(u.UnitTypeError, match="raise something to a"):
            np.power(3.0, Quantity(4.0, u.m))

    def test_errors_give_ufunc_name(self):
        with pytest.raises(u.UnitTypeError, match="'arcsin'"):
            np.arcsin(Quantity(1.0, u.m))

        with pytest.raises(u.UnitConversionError, match="'add'"):
            np.add(Quantity(1.0, u.m), Quantity(1.0, u.s))

    def test_numbers_that_can_have_any_unit(self):
        a = Quantity([-1.0, 2.0], u.m)

        assert_array_equal(np.less(a, 0), [True, False])
        assert_array_equal(np.less(a, np.zeros(2)), [True, False])
        assert_array_equal(np.less(a, np.array([np.inf, np.nan])), [True, False])

        with pytest.raises(u.UnitConversionError):
            np.less(a, np.array([0.0, 1.0]))


class TestUfuncsWithoutRules:
    """Other ufuncs are evaluated as for the numpy Quantity class."""

    def test_one_argument(self):
        result = np.deg2rad(Quantity(180.0, u.deg))

        assert result.unit is u.rad
        assert_allclose(result.value, np.pi)

    def test_two_outputs(self):
        fractional, integral = np.modf(Quantity([1.5, 2.25], u.m / u.cm))

        assert fractional.unit is u.dimensionless_unscaled
        assert integral.unit is u.dimensionless_unscaled
        assert_allclose(fractional.value, [0.0, 0.0])
        assert_allclose(integral.value, [150.0, 225.0])

    def test_reduce_and_accumulate(self):
        a = Quantity([1.0, 2.0, 3.0], u.m)

        result = np.add.reduce(a)

        assert result.unit is u.m
        assert result.value == 6.0

        result = np.add.reduce(a, initial=Quantity(100.0, u.cm))

        assert result.value == 7.0

        result = np.add.accumulate(a)

        assert result.unit is u.m
        assert_array_equal(result.value, [1.0, 3.0, 6.0])

        with pytest.raises(u.UnitsError, match="change the unit"):
            np.multiply.reduce(a)

    def test_at(self):
        a = Quantity([1.0, 2.0, 3.0], u.m)

        np.add.at(a, [0, 2], Quantity(50.0, u.cm))

        assert a.unit is u.m
        assert_array_equal(a.value, [1.5, 2.0, 3.5])


class TestOutput:
    def test_quantity_output(self):
        a = Quantity([1.0, 2.0], u.m)
        out = Quantity(np.zeros(2), u.s)

        result = np.multiply(a, a, out=out)

        assert result is out
        assert out.unit == u.m**2
        assert_array_equal(out.value, [1.0, 4.0])

    def test_plain_input(self):
        # Plain inputs are dimensionless.
        out = Quantity(np.zeros(2), u.m)

        result = np.absolute([-1.0, 2.0], out=out)

        assert result is out
        assert out.unit is u.dimensionless_unscaled
        assert_array_equal(out.value, [1.0, 2.0])

    def test_two_outputs(self):
        out = (Quantity(np.zeros(2), u.m), Quantity(np.zeros(2), u.m))

        result = np.modf(Quantity([1.5, 2.25], u.one), out=out)

        assert result[0] is out[0]
        assert result[1] is out[1]
        assert_array_equal(out[0].value, [0.5, 0.25])
        assert_array_equal(out[1].value, [1.0, 2.0])

    def test_plain_array_output(self):
        out = np.zeros(2)

        result = np.sin(Quantity([30.0, 90.0], u.deg), out=out)

        assert result is out
        assert_allclose(out, [0.5, 1.0])

        with pytest.raises(u.UnitTypeError):
            np.multiply(Quantity([1.0, 2.0], u.m), 2.0, out=out)

    def test_nothing_written_if_output_is_unsuitable(self):
        a = Quantity([1.0, 2.0], u.m)
        out = Quantity([3.0, 4.0], u.s)

        with pytest.raises(TypeError, match="non-quantity"):
            np.less(a, a, out=out)

        with pytest.raises(TypeError, match="cast"):
            np.add(a, a, out=Quantity(np.zeros(2, int), u.m, dtype=None))

        assert out.unit is u.s
        assert_array_equal(out.value, [3.0, 4.0])

    def test_where_cannot_be_quantity(self):
        a = np.ones(2)

        with pytest.raises(TypeError, match="all returned NotImplemented"):
            np.add(a, a, out=a, where=Quantity([True, False], u.m))


class TestResultClass:
    def test_subclass(self):
        result = np.add(MyQuantity([1.0, 2.0], u.m), Quantity(1.0, u.m))

        assert type(result) is MyQuantity

    def test_numpy_quantity_class(self):
        # Results have the Array API class, whatever the order.
        a = Quantity([1.0, 2.0], u.m)
        b = u.Quantity([1.0, 2.0], u.s)

        assert type(np.multiply(a, b)) is Quantity
        assert type(np.multiply(b, a)) is Quantity
        assert type(b * a) is Quantity
        assert type(u.s * a) is Quantity
        assert type(u.s / a) is Quantity


@pytest.mark.skipif(not HAS_ARRAY_API_STRICT, reason="array-api-strict not installed")
def test_other_array_libraries():
    import array_api_strict

    a = Quantity(array_api_strict.asarray([1.0, 2.0]), u.m)

    with pytest.raises(TypeError, match="numpy arrays as values"):
        np.add(a, a)

    with pytest.raises(TypeError, match="numpy arrays as values"):
        np.add(np.ones(2), np.ones(2), out=Quantity(array_api_strict.zeros(2), u.one))
