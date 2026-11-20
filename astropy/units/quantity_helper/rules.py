# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Rules for specifying how array functions transform units."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from astropy.units.core import dimensionless_unscaled, get_current_unit_registry
from astropy.units.errors import UnitConversionError, UnitsError, UnitTypeError
from astropy.units.si import radian

__all__ = ["ARBITRARY", "Converter", "converter", "plan"]

Converter = None | float | Callable[[Any], Any]


class _Arbitrary:
    """Marker for Python literals that can take any unit."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "ARBITRARY"


ARBITRARY = _Arbitrary()


@functools.lru_cache(maxsize=4096)
def _cached_plan(rule, units, static, equivalencies):
    return rule(*units, **dict(static))


def plan(rule: Callable, units: tuple, static: tuple = ()) -> tuple:
    """Memoized evaluation of ``rule`` for ``units``.

    Parameters
    ----------
    rule : callable
        The rule giving converters and the result unit.
    units : tuple
        The units of the arguments.
    static : tuple of tuple, optional
        Further keyword arguments for the rule, as ``(name, value)`` pairs, so
        that they can be part of the cache key (e.g. ``(("name", "add"),)``).

    Returns
    -------
    converters, unit
        A converter for each argument, and the unit of the result.

    Notes
    -----
    Since enabled equivalencies can give converters that are only valid within
    their context, they are part of the cache key.
    """
    equivalencies = frozenset(get_current_unit_registry()._equivalencies)

    return _cached_plan(rule, units, static, equivalencies)


def _unit_or_dimensionless(unit):
    """Units of non-quantities and literals are dimensionless."""
    return dimensionless_unscaled if unit is None or unit is ARBITRARY else unit


@functools.lru_cache(maxsize=4096)
def _scale(from_unit, to_unit) -> float:
    """Scale from ``from_unit`` to ``to_unit``, ignoring equivalencies."""
    return _unit_or_dimensionless(from_unit)._to(to_unit)


def converter(from_unit, to_unit, name: str = "") -> Converter:
    """Return the converter from ``from_unit`` to ``to_unit``."""
    if from_unit is ARBITRARY or from_unit is to_unit:
        return None

    try:
        scale = _scale(from_unit, to_unit)
    except UnitsError:
        try:
            # Enabled equivalencies (e.g. dimensionless_angles, temperature).
            return _unit_or_dimensionless(from_unit)._get_converter(to_unit)
        except UnitsError:
            what = f"'{name}'" if name else "function"

            if from_unit is None:
                raise UnitConversionError(
                    f"Can only apply {what} to quantities with dimensionless units "
                    f"when combined with non-quantity input (got unit '{to_unit}')."
                ) from None

            raise UnitConversionError(
                f"Can only apply {what} to quantities with compatible dimensions "
                f"(got '{from_unit}' and '{to_unit}')."
            ) from None

    return None if scale == 1.0 else float(scale)


def _require(unit, target, name, kind):
    """Converter to ``target``. Raises a UnitTypeError if not possible."""
    try:
        return converter(unit, target, name)
    except UnitConversionError:
        raise UnitTypeError(f"Can only apply '{name}' function to {kind}") from None


# ======================================================================================
# Rules for functions with one argument.
# ======================================================================================
def invariant(unit, name=""):
    """f(x) has the unit of x."""
    return (None,), unit


def to_plain(unit, name=""):
    """f(x) is a plain array."""
    return (None,), None


def dimensionless_to_dimensionless(unit, name=""):
    conv = _require(unit, dimensionless_unscaled, name, "dimensionless quantities")
    return (conv,), dimensionless_unscaled


def radian_to_dimensionless(unit, name=""):
    conv = _require(unit, radian, name, "quantities with angle units")
    return (conv,), dimensionless_unscaled


def dimensionless_to_radian(unit, name=""):
    conv = _require(unit, dimensionless_unscaled, name, "dimensionless quantities")
    return (conv,), radian


def power_of(p):
    """f(x) has the unit of x to the power ``p``."""

    def rule(unit, name=""):
        return (None,), _unit_or_dimensionless(unit) ** p

    rule.__name__ = f"power_of({p})"

    return rule


def static_power(unit, *, exponent):
    """x ** p for a Python number p, so the result unit is known."""
    return (None,), _unit_or_dimensionless(unit) ** exponent


def prod_rule(unit, *, n):
    """Product over ``n`` elements has unit**n (``n`` from the static shape)."""
    if n is None:
        if unit.physical_type == "dimensionless":
            return (converter(unit, dimensionless_unscaled),), dimensionless_unscaled

        raise UnitsError(
            "Cannot compute 'prod' of a quantity with units when the number of "
            "elements reduced over is not known (e.g., lazy arrays with unknown "
            "shape)."
        )
    return (None,), unit**n


# ======================================================================================
# Rules for functions with two or more arguments.
# ======================================================================================
def same_unit(*units, name=""):
    """Convert all to a common unit consistent with the result."""
    if None in units:
        target = dimensionless_unscaled
    else:
        target = next(
            (unit for unit in units if unit is not ARBITRARY), dimensionless_unscaled
        )

    try:
        return tuple(converter(unit, target, name) for unit in units), target
    except UnitConversionError:
        if target is not dimensionless_unscaled or None not in units:
            raise

        res_unit = [
            str(unit) for unit in units if unit is not None and unit is not ARBITRARY
        ]

        raise UnitConversionError(
            f"Can only apply '{name}' to dimensionless quantities when other "
            f"arguments are not quantities (got units {res_unit})."
        ) from None


def same_unit_to_plain(*units, name=""):
    """Convert all to a common unit; the result is plain (e.g. comparisons, ...)."""
    converters, _ = same_unit(*units, name=name)
    return converters, None


def same_unit_to_dimensionless(*units, name=""):
    converters, _ = same_unit(*units, name=name)
    return converters, dimensionless_unscaled


def same_unit_to_radian(*units, name=""):
    converters, _ = same_unit(*units, name=name)
    return converters, radian


def first_unit(unit1, unit2, name=""):
    """
    The result has the unit of the first argument. If the first argument is not a
    quantity, the result is a plain array.
    """
    return (None, None), None if unit1 is None or unit1 is ARBITRARY else unit1


def both_dimensionless(unit1, unit2, name=""):
    kind = "dimensionless quantities"

    return (
        _require(unit1, dimensionless_unscaled, name, kind),
        _require(unit2, dimensionless_unscaled, name, kind),
    ), dimensionless_unscaled


def multiplication(unit1, unit2, name=""):
    return (None, None), _unit_or_dimensionless(unit1) * _unit_or_dimensionless(unit2)


def division(unit1, unit2, name=""):
    return (None, None), _unit_or_dimensionless(unit1) / _unit_or_dimensionless(unit2)


def array_power(unit1, unit2, name="pow"):
    """Evaluate x ** y for an array y, which is only possible if x is dimensionless."""
    kind = "dimensionless quantities"

    try:
        conv1 = _require(unit1, dimensionless_unscaled, name, kind)
    except UnitTypeError:
        raise UnitTypeError(
            "Raising a quantity with units to an array power requires the "
            "exponent to be a finite Python number (so that the resulting unit is "
            f"known without inspecting values); got base unit '{unit1}'."
        ) from None

    try:
        conv2 = _require(unit2, dimensionless_unscaled, name, kind)
    except UnitTypeError:
        raise UnitTypeError(
            f"The exponent of '{name}' must be dimensionless (got unit '{unit2}')."
        ) from None

    return (conv1, conv2), dimensionless_unscaled


# ======================================================================================
# Rule tables.
# ======================================================================================
UNARY_RULES: Mapping[str, Callable] = MappingProxyType(
    {
        **dict.fromkeys(
            ["abs", "negative", "positive", "conj", "real", "imag"]
            + ["ceil", "floor", "trunc", "round"],
            invariant,
        ),
        **dict.fromkeys(["isfinite", "isinf", "isnan", "sign", "signbit"], to_plain),
        **dict.fromkeys(
            ["exp", "expm1", "log", "log1p", "log2", "log10"],
            dimensionless_to_dimensionless,
        ),
        # Hyperbolic functions taking radians, as for the np.sinh, etc., helpers.
        **dict.fromkeys(
            ["sin", "cos", "tan", "sinh", "cosh", "tanh"], radian_to_dimensionless
        ),
        **dict.fromkeys(
            ["acos", "asin", "atan", "acosh", "asinh", "atanh"], dimensionless_to_radian
        ),
        "sqrt": power_of(0.5),
        "square": power_of(2),
        "reciprocal": power_of(-1),
    }
)

BINARY_RULES: Mapping[str, Callable] = MappingProxyType(
    {
        **dict.fromkeys(
            [
                "add",
                "subtract",
                "maximum",
                "minimum",
                "hypot",
                "nextafter",
                "remainder",
            ],
            same_unit,
        ),
        **dict.fromkeys(
            ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"],
            same_unit_to_plain,
        ),
        "floor_divide": same_unit_to_dimensionless,
        "atan2": same_unit_to_radian,
        "copysign": first_unit,
        "logaddexp": both_dimensionless,
        "multiply": multiplication,
        "divide": division,
        "matmul": multiplication,
    }
)

# Functions for which any quantity input is an error.
REJECT_QUANTITY = frozenset(
    [
        "logical_and", "logical_or", "logical_xor", "logical_not",
        "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_invert",
        "bitwise_left_shift", "bitwise_right_shift",
    ]
)  # fmt: skip

# Reductions and statistics.
REDUCTION_SAME = frozenset(["sum", "mean", "max", "min", "std", "cumulative_sum"])
REDUCTION_PLAIN = frozenset(["argmax", "argmin", "all", "any", "count_nonzero"])

# Shape manipulation, etc.: f(x, ...) has the unit of x, and other arguments
# are not arrays (or are integer index arrays).
SHAPE_SAME = frozenset(
    [
        "reshape", "permute_dims", "expand_dims", "squeeze", "flip", "roll",
        "broadcast_to", "moveaxis", "repeat", "tile", "matrix_transpose",
        "sort", "unique_values", "astype", "take", "take_along_axis",
        "tril", "triu",
    ]
)  # fmt: skip
SHAPE_PLAIN = frozenset(["argsort", "nonzero"])

# Namespace members that do not act on arrays and pass through unchanged.
PASSTHROUGH = frozenset(
    [
        "e", "inf", "nan", "pi", "newaxis",
        "bool", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float32", "float64", "complex64", "complex128",
        "isdtype", "broadcast_shapes", "__array_api_version__",
        "__array_namespace_info__",
    ]
)  # fmt: skip


def is_literal_arbitrary(x) -> bool:
    """
    Check whether ``x`` is 0, inf, or nan, which can have any unit. Safe under
    tracing and lazy evaluation.
    """
    if type(x) is int:
        return x == 0

    if type(x) is float:
        return x == 0 or math.isinf(x) or math.isnan(x)

    return False
