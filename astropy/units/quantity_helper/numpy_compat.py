# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Support for numpy functions on quantities with numpy arrays as values."""

from __future__ import annotations

import math
import numbers

import numpy as np

from astropy.units import _quantity_array
from astropy.units.core import dimensionless_unscaled
from astropy.units.errors import UnitConversionError
from astropy.units.quantity import Quantity as _LegacyQuantity

from .converters import (
    can_have_arbitrary_unit,
    check_output,
    check_partial_output,
    converters_and_unit,
    writes_partially,
)
from .namespace import _is_quantity_unit, _wrap, apply_converter, split
from .rules import (
    ARBITRARY,
    BINARY_RULES,
    UNARY_RULES,
    array_power,
    plan,
    static_power,
)

__all__ = ["array_ufunc"]

# The ufuncs that have a rule, with the name of the rule. Others use the
# ufunc helpers of the old Quantity class.
UFUNC_RULES = {
    getattr(np, name): name
    for name in (*UNARY_RULES, *BINARY_RULES)
    if isinstance(getattr(np, name, None), np.ufunc)
}
UFUNC_RULES[np.power] = UFUNC_RULES[np.float_power] = "pow"


def _arbitrary(x, unit):
    """ARBITRARY for plain inputs that are all 0, inf or nan, else ``unit``."""
    if unit is not None:
        return unit

    try:
        return ARBITRARY if can_have_arbitrary_unit(x) else None
    except TypeError:
        return None


def _converters_and_unit(function, method, inputs):
    """Converters for the inputs and the unit of the result.

    For calls of ufuncs with a rule, these are given by the rule, so that the
    result is the same as for the namespace function.
    """
    name = UFUNC_RULES.get(function)

    if method != "__call__" or name is None:
        return converters_and_unit(function, method, *inputs)

    units = tuple(split(x)[1] for x in inputs)

    if not any(map(_is_quantity_unit, units)):
        # Only an output is a quantity; inputs are taken to be dimensionless.
        units = (dimensionless_unscaled,) * len(units)

    if name == "pow":
        exponent = inputs[1]

        if (
            _is_quantity_unit(units[0])
            and isinstance(exponent, numbers.Real)
            and not isinstance(exponent, bool)
            and math.isfinite(exponent)
        ):
            # A Python exponent, so the result unit is known.
            (_,), unit = plan(static_power, units[:1], (("exponent", exponent),))

            return (None, None), unit

        rule = array_power
    else:
        rule = UNARY_RULES[name] if function.nin == 1 else BINARY_RULES[name]

    # Messages should give the name of the ufunc.
    static = (("name", function.__name__),)

    try:
        return plan(rule, units, static)
    except UnitConversionError:
        # As for any number, plain arrays of 0, inf or nan can have any unit,
        # e.g. for comparisons like np.less(q, np.zeros(3)).
        arbitrary = tuple(map(_arbitrary, inputs, units))

        if arbitrary == units:
            raise

        return plan(rule, arbitrary, static)


def _output_value(output, unit, inputs, function):
    """The array to store a result with ``unit`` in.

    Raises if the result cannot be stored in ``output``, before anything is
    written.
    """
    if not isinstance(output, _quantity_array.Quantity):
        return check_output(output, unit, inputs, function)

    if unit is None:
        raise TypeError(
            f"Cannot store non-quantity output from {function.__name__} function "
            f"in {type(output)} instance"
        )

    if not isinstance(output._value, np.ndarray):
        raise TypeError(
            "Can only store output in quantities with numpy arrays as values."
        )

    # Python numbers are kept, since they do not set the type of the result.
    values = [split(x)[0] for x in inputs]
    result_type = np.result_type(
        *[
            v if isinstance(v, (numbers.Number, np.ndarray)) else np.asarray(v)
            for v in values
        ]
    )

    if not (
        result_type.names
        or output.dtype.names
        or np.can_cast(result_type, output.dtype, casting="same_kind")
    ):
        raise TypeError(
            "Arguments cannot be cast safely to inplace output with "
            f"dtype={output.dtype}"
        )

    return output._value


def _result(result, unit, output, inputs):
    """The result with ``unit``, in ``output`` if given."""
    if output is None:
        return _wrap(result, unit, inputs)

    if isinstance(output, _quantity_array.Quantity):
        output._unit = unit
    elif isinstance(output, _LegacyQuantity) and unit is not None:
        output._set_unit(unit)

    return output


def _defers(x):
    """Whether ``x`` has an override that numpy should try instead."""
    return getattr(type(x), "__array_ufunc__", None) not in (
        None,
        np.ndarray.__array_ufunc__,
        _quantity_array.Quantity.__array_ufunc__,
        _LegacyQuantity.__array_ufunc__,
    )


def array_ufunc(function, method, *inputs, **kwargs):
    """Evaluate a numpy ufunc for quantities.

    The ufuncs that correspond to functions of the namespace use the same
    rules, so that, e.g., ``np.add(q1, q2)`` and ``q1 + q2`` are the same.
    """
    outputs = kwargs.get("out")

    for x in (*inputs, *(outputs or ())):
        if isinstance(x, _quantity_array.Quantity) and not isinstance(
            x._value, np.ndarray
        ):
            raise TypeError(
                "numpy functions are only supported for quantities with numpy "
                "arrays as values; use the namespace of the quantity instead."
            )

    if isinstance(kwargs.get("where"), _quantity_array.Quantity) or any(
        map(_defers, (*inputs, *(outputs or ())))
    ):
        # Let other overrides handle their own types; the mask should be a
        # plain boolean array.
        return NotImplemented

    converters, unit = _converters_and_unit(function, method, inputs)
    units = unit if function.nout > 1 else (unit,)

    if outputs is not None:
        if writes_partially(method, kwargs):
            for output, unit_ in zip(outputs, units):
                check_partial_output(output, unit_, function)

        kwargs["out"] = tuple(
            _output_value(output, unit_, inputs, function)
            for output, unit_ in zip(outputs, units)
        )

    if method == "reduce" and "initial" in kwargs and unit is not None:
        kwargs["initial"] = _quantity_array._to_unit(kwargs["initial"], unit)

    values = [
        apply_converter(split(x)[0], conv)
        for x, conv in zip(inputs, converters, strict=False)
    ]
    values += [split(x)[0] for x in inputs[len(values) :]]
    result = getattr(function, method)(*values, **kwargs)

    if result is None or result is NotImplemented:
        # Method 'at', or another override could not handle the inputs.
        return result

    outputs = outputs or (None,) * function.nout

    if function.nout == 1:
        return _result(result, unit, outputs[0], inputs)

    return tuple(map(_result, result, units, outputs, (inputs,) * function.nout))
