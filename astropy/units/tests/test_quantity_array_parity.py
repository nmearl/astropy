# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Parity of the Array API quantity namespace with numpy functions on Quantity."""

import array_api_compat.numpy as numpy_namespace
import numpy as np
import pytest

from astropy import units as u
from astropy.units._quantity_array import Quantity
from astropy.units.quantity_helper import rules
from astropy.units.quantity_helper.namespace import units_namespace

xp = units_namespace(numpy_namespace)

UNITS = [u.m, u.km, u.deg, u.rad, u.one, u.m / u.cm, u.s]
VALUE = np.array([0.5, 1.0, 2.0])
VALUE_2D = np.array([[0.5, 1.0], [2.0, 3.0]])

# numpy names of functions named differently in the Array API standard.
NUMPY_NAMES = {
    "acos": "arccos",
    "asin": "arcsin",
    "atan": "arctan",
    "atan2": "arctan2",
    "acosh": "arccosh",
    "asinh": "arcsinh",
    "atanh": "arctanh",
    "bitwise_invert": "invert",
    "bitwise_left_shift": "left_shift",
    "bitwise_right_shift": "right_shift",
}


def assert_same(new, old):
    """Check that a result of the namespace equals that of numpy on Quantity."""
    if isinstance(old, (tuple, list)):
        assert len(new) == len(old)
        for new_item, old_item in zip(new, old):
            assert_same(new_item, old_item)
    elif isinstance(old, u.Quantity):
        assert isinstance(new, Quantity)
        assert new.unit == old.unit
        assert np.shape(new.value) == np.shape(old.value)
        np.testing.assert_allclose(new.value, old.value, equal_nan=True)
    else:
        assert not isinstance(new, Quantity)
        np.testing.assert_array_equal(new, old)


def assert_parity(new_function, old_function, new_args, old_args, **kwargs):
    """Check that new_function(*new_args) behaves as old_function(*old_args).

    If the old function raises, the new one should raise the same kind of
    error (or a subclass, e.g. UnitTypeError for UnitsError); otherwise, both
    should give the same result.
    """
    try:
        expected = old_function(*old_args, **kwargs)
    except Exception as exc:
        with pytest.raises(type(exc)):
            new_function(*new_args, **kwargs)
    else:
        assert_same(new_function(*new_args, **kwargs), expected)


class TestElementwise:
    @pytest.mark.parametrize("unit", UNITS, ids=str)
    @pytest.mark.parametrize("name", sorted(rules.UNARY_RULES))
    def test_quantity(self, name, unit):
        assert_parity(
            getattr(xp, name),
            getattr(np, NUMPY_NAMES.get(name, name)),
            [Quantity(VALUE, unit)],
            [u.Quantity(VALUE, unit)],
        )

    @pytest.mark.parametrize("unit2", UNITS, ids=str)
    @pytest.mark.parametrize("unit1", UNITS, ids=str)
    @pytest.mark.parametrize("name", sorted(rules.BINARY_RULES))
    def test_two_quantities(self, name, unit1, unit2):
        assert_parity(
            getattr(xp, name),
            getattr(np, NUMPY_NAMES.get(name, name)),
            [Quantity(VALUE, unit1), Quantity(VALUE, unit2)],
            [u.Quantity(VALUE, unit1), u.Quantity(VALUE, unit2)],
        )

    @pytest.mark.parametrize("unit", UNITS, ids=str)
    @pytest.mark.parametrize("name", sorted(rules.BINARY_RULES))
    def test_quantity_and_plain_array(self, name, unit):
        new_function = getattr(xp, name)
        old_function = getattr(np, NUMPY_NAMES.get(name, name))
        assert_parity(
            new_function,
            old_function,
            [Quantity(VALUE, unit), VALUE],
            [u.Quantity(VALUE, unit), VALUE],
        )
        assert_parity(
            new_function,
            old_function,
            [VALUE, Quantity(VALUE, unit)],
            [VALUE, u.Quantity(VALUE, unit)],
        )

    # Scaled dimensionless units are in test_literal_with_scaled_dimensionless.
    @pytest.mark.parametrize("literal", [0, 2.0, np.inf])
    @pytest.mark.parametrize("unit", [u.m, u.km, u.deg, u.rad, u.one, u.s], ids=str)
    @pytest.mark.parametrize("name", sorted(set(rules.BINARY_RULES) - {"matmul"}))
    def test_quantity_and_literal(self, name, unit, literal):
        new_function = getattr(xp, name)
        old_function = getattr(np, NUMPY_NAMES.get(name, name))
        assert_parity(
            new_function,
            old_function,
            [Quantity(VALUE, unit), literal],
            [u.Quantity(VALUE, unit), literal],
        )
        assert_parity(
            new_function,
            old_function,
            [literal, Quantity(VALUE, unit)],
            [literal, u.Quantity(VALUE, unit)],
        )

    @pytest.mark.parametrize("name", ["add", "maximum", "hypot", "remainder"])
    def test_literal_with_scaled_dimensionless(self, name):
        # Literals 0 and inf can have any unit, so the quantity keeps its unit.
        new = getattr(xp, name)(Quantity(VALUE, u.m / u.cm), 0)
        assert new.unit == u.m / u.cm
        # numpy functions on Quantity convert to dimensionless_unscaled instead.
        old = getattr(np, name)(u.Quantity(VALUE, u.m / u.cm), 0)
        assert old.unit == u.dimensionless_unscaled
        np.testing.assert_allclose(new.to_value(u.one), old.value, equal_nan=True)

    @pytest.mark.parametrize("name", sorted(rules.REJECT_QUANTITY))
    def test_reject_quantity(self, name):
        x = np.array([1, 0, 2])
        new_function = getattr(xp, name)
        old_function = getattr(np, NUMPY_NAMES.get(name, name))
        if name in ("logical_not", "bitwise_invert"):
            assert_parity(
                new_function, old_function, [Quantity(x, u.one)], [u.Quantity(x, u.one)]
            )
        else:
            assert_parity(
                new_function,
                old_function,
                [Quantity(x, u.one), x],
                [u.Quantity(x, u.one), x],
            )


class TestPow:
    @pytest.mark.parametrize("exponent", [2, 0.5, -1, 0])
    @pytest.mark.parametrize("unit", UNITS, ids=str)
    def test_python_exponent(self, unit, exponent):
        assert_parity(
            xp.pow,
            np.power,
            [Quantity(VALUE, unit), exponent],
            [u.Quantity(VALUE, unit), exponent],
        )

    @pytest.mark.parametrize("unit", [u.one, u.m / u.cm], ids=str)
    def test_dimensionless_array_exponent(self, unit):
        assert_parity(
            xp.pow,
            np.power,
            [Quantity(VALUE, unit), VALUE],
            [u.Quantity(VALUE, unit), VALUE],
        )
        assert_parity(
            xp.pow,
            np.power,
            [VALUE, Quantity(VALUE, unit)],
            [VALUE, u.Quantity(VALUE, unit)],
        )

    @pytest.mark.parametrize("exponent", [np.inf, np.nan])
    def test_dimensionless_non_finite_exponent(self, exponent):
        assert_parity(
            xp.pow,
            np.power,
            [Quantity(VALUE, u.one), exponent],
            [u.Quantity(VALUE, u.one), exponent],
        )

    def test_array_exponent_with_units(self):
        with pytest.raises(u.UnitTypeError, match="exponent"):
            xp.pow(Quantity(VALUE, u.m), VALUE)
        # numpy functions on Quantity raise a ValueError.
        with pytest.raises(ValueError, match="scalar power"):
            np.power(u.Quantity(VALUE, u.m), VALUE)

    def test_non_finite_exponent_with_units(self):
        with pytest.raises(u.UnitTypeError, match="finite"):
            xp.pow(Quantity(VALUE, u.m), np.inf)
        # numpy functions on Quantity raise an OverflowError.
        with pytest.raises(OverflowError):
            np.power(u.Quantity(VALUE, u.m), np.inf)


class TestReductions:
    @pytest.mark.parametrize("axis", [None, 1])
    @pytest.mark.parametrize("unit", UNITS, ids=str)
    @pytest.mark.parametrize(
        "name", ["sum", "mean", "max", "min", "std", "var", "argmax", "argmin"]
    )
    def test_reduction(self, name, unit, axis):
        assert_parity(
            getattr(xp, name),
            getattr(np, name),
            [Quantity(VALUE_2D, unit)],
            [u.Quantity(VALUE_2D, unit)],
            axis=axis,
        )

    @pytest.mark.parametrize("unit", UNITS, ids=str)
    @pytest.mark.parametrize("name", ["cumulative_sum", "count_nonzero"])
    def test_along_axis(self, name, unit):
        assert_parity(
            getattr(xp, name),
            getattr(np, name),
            [Quantity(VALUE_2D, unit)],
            [u.Quantity(VALUE_2D, unit)],
            axis=0,
        )

    @pytest.mark.parametrize("name", ["prod", "cumulative_prod"])
    def test_products_of_dimensionless(self, name):
        assert_parity(
            getattr(xp, name),
            getattr(np, name),
            [Quantity(VALUE_2D, u.one)],
            [u.Quantity(VALUE_2D, u.one)],
            axis=0,
        )

    @pytest.mark.parametrize("unit", [u.m, u.deg, u.m / u.cm], ids=str)
    def test_product_with_unit(self, unit):
        x = Quantity(VALUE_2D, unit)
        assert xp.prod(x).unit == unit**4
        assert xp.prod(x, axis=1).unit == unit**2
        np.testing.assert_allclose(xp.prod(x).value, np.prod(VALUE_2D))
        # numpy functions on Quantity raise, since the unit depends on the shape.
        with pytest.raises(u.UnitsError):
            np.prod(u.Quantity(VALUE_2D, unit))

    def test_cumulative_product_of_scaled_dimensionless(self):
        result = xp.cumulative_prod(Quantity(VALUE_2D, u.m / u.cm), axis=0)
        assert result.unit == u.one
        np.testing.assert_allclose(result.value, np.cumprod(100 * VALUE_2D, axis=0))
        # numpy functions on Quantity raise.
        with pytest.raises(u.UnitsError):
            np.cumulative_prod(u.Quantity(VALUE_2D, u.m / u.cm), axis=0)

    @pytest.mark.parametrize("name", ["all", "any"])
    def test_all_any(self, name):
        result = getattr(xp, name)(Quantity([0.0, 1.0], u.m))
        assert result == getattr(np, name)([0.0, 1.0])
        # numpy functions on Quantity do not support these.
        with pytest.raises(TypeError):
            getattr(np, name)(u.Quantity([0.0, 1.0], u.m))


class TestShape:
    @pytest.mark.parametrize("unit", [u.m, u.one], ids=str)
    @pytest.mark.parametrize(
        "name, numpy_name, args, kwargs",
        [
            ("reshape", "reshape", [(4,)], {}),
            ("permute_dims", "transpose", [(1, 0)], {}),
            ("expand_dims", "expand_dims", [], {"axis": 0}),
            ("flip", "flip", [], {}),
            ("roll", "roll", [1], {}),
            ("moveaxis", "moveaxis", [0, 1], {}),
            ("repeat", "repeat", [2], {}),
            ("tile", "tile", [(2, 1)], {}),
            ("matrix_transpose", "matrix_transpose", [], {}),
            ("sort", "sort", [], {}),
            ("unique_values", "unique_values", [], {}),
            ("astype", "astype", [np.float32], {}),
            ("take", "take", [np.array([1, 0])], {"axis": 0}),
            ("take_along_axis", "take_along_axis", [np.array([[1, 0], [0, 1]])], {"axis": 1}),
            ("tril", "tril", [], {}),
            ("triu", "triu", [], {}),
            ("argsort", "argsort", [], {}),
            ("nonzero", "nonzero", [], {}),
        ],
    )  # fmt: skip
    def test_shape(self, name, numpy_name, args, kwargs, unit):
        assert_parity(
            getattr(xp, name),
            getattr(np, numpy_name),
            [Quantity(VALUE_2D, unit), *args],
            [u.Quantity(VALUE_2D, unit), *args],
            **kwargs,
        )

    @pytest.mark.parametrize("unit", [u.m, u.one], ids=str)
    def test_squeeze(self, unit):
        assert_parity(
            xp.squeeze,
            np.squeeze,
            [Quantity(VALUE_2D[np.newaxis], unit)],
            [u.Quantity(VALUE_2D[np.newaxis], unit)],
            axis=0,
        )

    def test_broadcast(self):
        assert xp.broadcast_to(Quantity(VALUE, u.m), (2, 3)).unit == u.m
        x, y = xp.broadcast_arrays(
            Quantity(VALUE, u.m), Quantity(VALUE[:, np.newaxis], u.km)
        )
        assert x.unit == u.m
        assert y.unit == u.km
        # numpy functions on Quantity give plain arrays, unless subok=True.
        assert type(np.broadcast_to(u.Quantity(VALUE, u.m), (2, 3))) is np.ndarray


PAIRS_OF_UNITS = [(u.m, u.m), (u.m, u.km), (u.km, u.m), (u.m, u.deg), (u.one, u.one)]


class TestSeveralArrays:
    @pytest.mark.parametrize("unit1, unit2", PAIRS_OF_UNITS, ids=str)
    def test_where(self, unit1, unit2):
        condition = VALUE > 1
        assert_parity(
            xp.where,
            np.where,
            [condition, Quantity(VALUE, unit1), Quantity(VALUE, unit2)],
            [condition, u.Quantity(VALUE, unit1), u.Quantity(VALUE, unit2)],
        )

    @pytest.mark.parametrize("other", [0, 2.0, VALUE])
    @pytest.mark.parametrize("unit", [u.m, u.one], ids=str)
    def test_where_with_non_quantity(self, unit, other):
        condition = VALUE > 1
        assert_parity(
            xp.where,
            np.where,
            [condition, Quantity(VALUE, unit), other],
            [condition, u.Quantity(VALUE, unit), other],
        )

    @pytest.mark.parametrize("unit1, unit2", PAIRS_OF_UNITS, ids=str)
    def test_clip(self, unit1, unit2):
        x, bound = Quantity(VALUE, unit1), Quantity(1.0, unit2)
        old_x, old_bound = u.Quantity(VALUE, unit1), u.Quantity(1.0, unit2)
        assert_parity(xp.clip, np.clip, [x, bound, None], [old_x, old_bound, None])
        assert_parity(xp.clip, np.clip, [x, None, bound], [old_x, None, old_bound])

    @pytest.mark.parametrize("unit1, unit2", PAIRS_OF_UNITS, ids=str)
    @pytest.mark.parametrize(
        "name, numpy_name", [("concat", "concatenate"), ("stack", "stack")]
    )
    def test_join(self, name, numpy_name, unit1, unit2):
        assert_parity(
            getattr(xp, name),
            getattr(np, numpy_name),
            [[Quantity(VALUE, unit1), Quantity(VALUE, unit2)]],
            [[u.Quantity(VALUE, unit1), u.Quantity(VALUE, unit2)]],
        )

    @pytest.mark.parametrize("unit1, unit2", PAIRS_OF_UNITS, ids=str)
    @pytest.mark.parametrize("name", ["isin", "searchsorted", "vecdot", "meshgrid"])
    def test_two_arrays(self, name, unit1, unit2):
        assert_parity(
            getattr(xp, name),
            getattr(np, name),
            [Quantity(VALUE, unit1), Quantity(VALUE, unit2)],
            [u.Quantity(VALUE, unit1), u.Quantity(VALUE, unit2)],
        )

    @pytest.mark.parametrize("unit1, unit2", PAIRS_OF_UNITS, ids=str)
    def test_tensordot(self, unit1, unit2):
        assert_parity(
            xp.tensordot,
            np.tensordot,
            [Quantity(VALUE, unit1), Quantity(VALUE, unit2)],
            [u.Quantity(VALUE, unit1), u.Quantity(VALUE, unit2)],
            axes=1,
        )

    @pytest.mark.parametrize("unit", [u.m, u.one], ids=str)
    def test_diff(self, unit):
        assert_parity(
            xp.diff, np.diff, [Quantity(VALUE, unit)], [u.Quantity(VALUE, unit)]
        )

    @pytest.mark.parametrize("unit1, unit2", PAIRS_OF_UNITS, ids=str)
    def test_diff_prepend(self, unit1, unit2):
        x, prepend = Quantity(VALUE, unit1), Quantity([1.0], unit2)
        old_x, old_prepend = u.Quantity(VALUE, unit1), u.Quantity([1.0], unit2)
        try:
            expected = np.diff(old_x, prepend=old_prepend)
        except u.UnitsError:
            with pytest.raises(u.UnitsError):
                xp.diff(x, prepend=prepend)
        else:
            assert_same(xp.diff(x, prepend=prepend), expected)

    def test_diff_prepend_zero(self):
        result = xp.diff(Quantity(VALUE, u.m), prepend=0)
        assert result.unit == u.m
        np.testing.assert_allclose(result.value, [0.5, 0.5, 1.0])
        # numpy functions on Quantity take 0 to be dimensionless.
        with pytest.raises(u.UnitConversionError):
            np.diff(u.Quantity(VALUE, u.m), prepend=0)

    def test_searchsorted_plain_in_quantity(self):
        with pytest.raises(u.UnitConversionError):
            xp.searchsorted(VALUE, Quantity(VALUE, u.m))
        # numpy functions on Quantity ignore the unit.
        assert type(np.searchsorted(VALUE, u.Quantity(VALUE, u.m))) is np.ndarray

    @pytest.mark.parametrize("unit", [u.m, u.one], ids=str)
    @pytest.mark.parametrize(
        "name", ["unique_all", "unique_counts", "unique_inverse", "unstack"]
    )
    def test_one_array(self, name, unit):
        assert_parity(
            getattr(xp, name),
            getattr(np, name),
            [Quantity(VALUE, unit)],
            [u.Quantity(VALUE, unit)],
        )


class TestCreation:
    @pytest.mark.parametrize("unit", [u.m, u.one], ids=str)
    @pytest.mark.parametrize("name", ["zeros_like", "ones_like"])
    def test_like(self, name, unit):
        assert_parity(
            getattr(xp, name),
            getattr(np, name),
            [Quantity(VALUE, unit)],
            [u.Quantity(VALUE, unit)],
        )

    def test_empty_like(self):
        result = xp.empty_like(Quantity(VALUE, u.m))
        expected = np.empty_like(u.Quantity(VALUE, u.m))
        assert result.shape == expected.shape
        assert result.unit == expected.unit

    @pytest.mark.parametrize("fill_value", [3.0, 0])
    @pytest.mark.parametrize("unit", [u.m, u.one], ids=str)
    def test_full_like_plain_fill_value(self, unit, fill_value):
        assert_parity(
            xp.full_like,
            np.full_like,
            [Quantity(VALUE, unit), fill_value],
            [u.Quantity(VALUE, unit), fill_value],
        )

    @pytest.mark.parametrize("fill_unit", [u.km, u.one], ids=str)
    @pytest.mark.parametrize("unit", [u.m, u.one], ids=str)
    def test_full_like_quantity_fill_value(self, unit, fill_unit):
        assert_parity(
            xp.full_like,
            np.full_like,
            [Quantity(VALUE, unit), Quantity(3.0, fill_unit)],
            [u.Quantity(VALUE, unit), u.Quantity(3.0, fill_unit)],
        )

    def test_full_like_plain_with_quantity(self):
        assert xp.full_like(VALUE, Quantity(3.0, u.m)).unit == u.m
        # numpy functions on Quantity raise.
        with pytest.raises(u.UnitConversionError):
            np.full_like(VALUE, 3.0 * u.m)

    def test_full_with_quantity(self):
        assert xp.full((2,), Quantity(3.0, u.m)).unit == u.m
        # numpy functions on Quantity give a plain array.
        assert type(np.full((2,), 3.0 * u.m)) is np.ndarray

    @pytest.mark.parametrize("unit", [u.m, u.one], ids=str)
    def test_linspace(self, unit):
        assert_parity(
            xp.linspace,
            np.linspace,
            [Quantity(1.0, unit), Quantity(5.0, unit), 3],
            [u.Quantity(1.0, unit), u.Quantity(5.0, unit), 3],
        )

    def test_linspace_unit_of_start(self):
        result = xp.linspace(Quantity(1.0, u.m), Quantity(1.0, u.km), 3)
        assert result.unit == u.m
        np.testing.assert_allclose(result.value, [1.0, 500.5, 1000.0])
        # numpy functions on Quantity give the unit of stop.
        assert np.linspace(1.0 * u.m, 1.0 * u.km, 3).unit == u.km

    def test_arange(self):
        result = xp.arange(Quantity(1.0, u.m), Quantity(0.005, u.km))
        assert result.unit == u.m
        np.testing.assert_allclose(result.value, [1.0, 2.0, 3.0, 4.0])
        # numpy functions on Quantity do not support quantities.
        with pytest.raises(TypeError):
            np.arange(1.0 * u.m, 5.0 * u.m)


@pytest.mark.parametrize("name", ["add", "maximum", "less"])
def test_temperature_equivalency(name):
    with u.set_enabled_equivalencies(u.temperature()):
        assert_parity(
            getattr(xp, name),
            getattr(np, name),
            [Quantity([0.0, 10.0], u.deg_C), Quantity([273.15, 300.0], u.K)],
            [u.Quantity([0.0, 10.0], u.deg_C), u.Quantity([273.15, 300.0], u.K)],
        )
