# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests of the unit rules, including parity with the ufunc helpers."""

import numpy as np
import pytest

from astropy import units as u
from astropy.units.quantity_helper import UFUNC_HELPERS, rules
from astropy.units.quantity_helper.rules import (
    ARBITRARY,
    BINARY_RULES,
    UNARY_RULES,
    _cached_plan,
    plan,
)

# Array API names that differ from the numpy ufunc names.
NUMPY_NAMES = {
    "acos": "arccos",
    "asin": "arcsin",
    "atan": "arctan",
    "atan2": "arctan2",
    "acosh": "arccosh",
    "asinh": "arcsinh",
    "atanh": "arctanh",
    "conj": "conjugate",
}
# Rules for functions that are not ufuncs in numpy, so have no ufunc helper.
NOT_UFUNCS = {"real", "imag", "round"}

UNITS = [
    u.m,
    u.km,
    u.deg,
    u.rad,
    u.dimensionless_unscaled,
    u.m / u.cm,
    u.s,
]


def _ufunc(name):
    return getattr(np, NUMPY_NAMES.get(name, name))


def _scale(converter):
    """Scale factor applied by a converter, for comparison."""
    if converter is None:
        return 1.0
    if callable(converter):
        return float(converter(1.0))
    return converter


def _call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except u.UnitsError as e:
        return None, e


def check_parity(name, units):
    ufunc = _ufunc(name)
    rule = {**UNARY_RULES, **BINARY_RULES}[name]
    expected, expected_e = _call(UFUNC_HELPERS[ufunc], ufunc, *units)
    result, e = _call(rule, *units, name=name)

    if expected_e is not None:
        assert type(e) is type(expected_e), (name, units)
        return

    expected_converters, expected_unit = expected

    if False in expected_converters:
        # The helper defers to the values. Plain arguments are only allowed if it is 0,
        # inf, or nan. Rules raise for a plain argument, but accept an ARBITRARY
        # literal instead.
        assert isinstance(e, u.UnitConversionError), (name, units)

        index = expected_converters.index(False)
        units = list(units)
        units[index] = ARBITRARY
        result, e = _call(rule, *units, name=name)
        expected_converters = list(expected_converters)
        expected_converters[index] = None

    assert e is None, (name, units, e)

    converters, unit = result

    if expected_unit is None:
        assert unit is None
    else:
        assert unit == expected_unit, (name, units)

    assert len(converters) == len(expected_converters)

    for converter, expected_converter in zip(converters, expected_converters):
        assert _scale(converter) == pytest.approx(_scale(expected_converter))


@pytest.mark.parametrize("unit", UNITS, ids=str)
@pytest.mark.parametrize("name", sorted(set(UNARY_RULES) - NOT_UFUNCS))
def test_unary_parity(name, unit):
    check_parity(name, (unit,))


# All pairs of units, where None is a plain (non-quantity) argument.
UNIT_PAIRS = [
    pytest.param(unit1, unit2, id=f"{unit1}-{unit2}")
    for unit1 in [*UNITS, None]
    for unit2 in [*UNITS, None]
    if unit1 is not None or unit2 is not None
]


@pytest.mark.parametrize("unit1, unit2", UNIT_PAIRS)
@pytest.mark.parametrize("name", sorted(set(BINARY_RULES) - NOT_UFUNCS))
def test_binary_parity(name, unit1, unit2):
    check_parity(name, (unit1, unit2))


@pytest.mark.parametrize("unit", UNITS, ids=str)
@pytest.mark.parametrize("name", sorted(set(UNARY_RULES) - NOT_UFUNCS))
def test_unary_parity_dimensionless_angles(name, unit):
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        check_parity(name, (unit,))


@pytest.mark.parametrize("unit1, unit2", UNIT_PAIRS)
@pytest.mark.parametrize("name", sorted(set(BINARY_RULES) - NOT_UFUNCS))
def test_binary_parity_dimensionless_angles(name, unit1, unit2):
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        check_parity(name, (unit1, unit2))


def test_enabled_equivalencies_offset():
    # Converters from equivalencies need not be a simple scaling.
    with u.set_enabled_equivalencies(u.temperature()):
        (conv1, conv2), unit = plan(rules.same_unit, (u.deg_C, u.K))

    assert conv1 is None
    assert unit == u.deg_C
    assert conv2(0.0) == pytest.approx(-273.15)


def test_enabled_equivalencies_are_part_of_cache_key():
    _cached_plan.cache_clear()

    with u.set_enabled_equivalencies(u.temperature()):
        plan(rules.same_unit, (u.deg_C, u.K))

    # Outside the context, the conversion is no longer possible.
    with pytest.raises(u.UnitConversionError):
        plan(rules.same_unit, (u.deg_C, u.K))

    # Within a new context with the same equivalencies, it is again.
    with u.set_enabled_equivalencies(u.temperature()):
        plan(rules.same_unit, (u.deg_C, u.K))


def test_memoized_with_enabled_equivalencies():
    # Some equivalencies are enabled for good (e.g. on importing
    # astropy.cosmology), which should not stop memoization.
    with u.add_enabled_equivalencies(u.dimensionless_angles()):
        _cached_plan.cache_clear()

        for _ in range(3):
            assert plan(rules.radian_to_dimensionless, (u.one,))[1] == u.one

        info = _cached_plan.cache_info()

    assert (info.misses, info.hits) == (1, 2)


def test_rule_tables_cover_ufuncs():
    # Every rule for a numpy ufunc has a helper to be compared against.
    for name in (set(UNARY_RULES) | set(BINARY_RULES)) - NOT_UFUNCS:
        assert _ufunc(name) in UFUNC_HELPERS, name


@pytest.mark.parametrize("exponent", [2, 0.5, -1, 0])
@pytest.mark.parametrize("unit", UNITS, ids=str)
def test_static_power(unit, exponent):
    assert rules.static_power(unit, exponent=exponent) == ((None,), unit**exponent)


def test_array_power():
    assert rules.array_power(u.m / u.cm, None)[1] == u.dimensionless_unscaled

    with pytest.raises(u.UnitTypeError, match="Python number"):
        rules.array_power(u.m, None)

    with pytest.raises(u.UnitTypeError, match="exponent of 'pow' must be"):
        rules.array_power(None, u.m)


def test_prod():
    assert rules.prod_rule(u.m, n=3) == ((None,), u.m**3)
    assert rules.prod_rule(u.m / u.cm, n=None) == ((100.0,), u.dimensionless_unscaled)

    with pytest.raises(u.UnitsError, match="not known"):
        rules.prod_rule(u.m, n=None)


@pytest.mark.parametrize(
    "value, expected",
    [(0, True), (0.0, True), (np.inf, True), (-np.inf, True), (np.nan, True)]
    + [(1, False), (1.5, False), (np.float64(0), False), (True, False)]
    + [(10**400, False), (-(10**400), False)],
)
def test_is_literal_arbitrary(value, expected):
    assert rules.is_literal_arbitrary(value) is expected


def test_plan_is_memoized():
    _cached_plan.cache_clear()

    for _ in range(3):
        assert plan(rules.multiplication, (u.km, u.s)) == ((None, None), u.km * u.s)

    info = _cached_plan.cache_info()

    assert (info.misses, info.hits) == (1, 2)


def test_plan_keeps_equal_but_distinct_units_apart():
    thousand_m = u.CompositeUnit(1000, [u.m], [1])

    assert thousand_m == u.km

    _cached_plan.cache_clear()

    assert plan(rules.invariant, (u.km,))[1] is u.km
    assert plan(rules.invariant, (thousand_m,))[1] is thousand_m
