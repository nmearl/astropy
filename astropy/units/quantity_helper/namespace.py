# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Array API namespaces for quantities.

``units_namespace(xp)`` wraps an Array API namespace ``xp`` (e.g. numpy, JAX,
array-api-strict) in a namespace whose functions accept and return quantities.
Each function splits its arguments into values and units, gets the converters
and the result unit from the rules for the units only, calls the function of
``xp`` on the converted values, and attaches the result unit. Since values are
never inspected for the unit handling, the same code works for any array
library, including under tracing.
"""

from __future__ import annotations

import functools

import array_api_compat
import numpy as np

from astropy.units import _quantity_array
from astropy.units.quantity import Quantity as _LegacyQuantity
from astropy.utils.compat.array_api import default_floating_dtype

from .rules import (
    ARBITRARY,
    BINARY_RULES,
    PASSTHROUGH,
    REDUCTION_PLAIN,
    REDUCTION_SAME,
    REJECT_QUANTITY,
    SHAPE_PLAIN,
    SHAPE_SAME,
    UNARY_RULES,
    is_literal_arbitrary,
    plan,
    same_unit,
)

__all__ = ["namespace_for", "units_namespace"]


def split(x):
    """The value of ``x`` and its unit, as the rules see it."""
    if isinstance(x, _quantity_array.Quantity):
        return x._value, x._unit

    if isinstance(x, _LegacyQuantity):
        return x.view(np.ndarray), x.unit

    if is_literal_arbitrary(x):
        return x, ARBITRARY

    return x, None


def _is_quantity_unit(unit):
    return unit is not None and unit is not ARBITRARY


def _wrap(result, unit, args):
    """Attach ``unit`` to ``result``, as an instance of the first quantity's type."""
    if unit is None:
        return result

    for arg in args:
        if isinstance(arg, _quantity_array.Quantity):
            return arg._new(result, unit)

    return _quantity_array.Quantity._new(result, unit)


def apply_converter(value, conv):
    """Apply a converter given by the rules to a plain value.

    Integer and boolean arrays are converted to floating point before they
    are scaled, since the Array API standard does not define mixing them
    with Python floats.
    """
    if conv is None:
        return value

    if type(conv) is not float:
        return conv(value)

    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biu":
            value = value.astype(np.float64)
    elif hasattr(value, "dtype") and not isinstance(value, np.generic):
        xp = array_api_compat.array_namespace(value)

        if xp.isdtype(value.dtype, ("integral", "bool")):
            device = array_api_compat.device(value)
            value = xp.astype(value, default_floating_dtype(xp, device))

    return value * conv


def _check_arguments(name, args, kwargs):
    """Reject arguments other than the arrays acted on that are quantities.

    Also reject ``out``, which some array libraries take (the result would
    be written without any unit in such cases).
    """
    if "out" in kwargs:
        raise TypeError(f"'{name}' does not support 'out' for quantities.")

    for arg in (*args, *kwargs.values()):
        if isinstance(arg, (_quantity_array.Quantity, _LegacyQuantity)):
            raise TypeError(f"'{name}' does not accept quantities for this argument.")


def _converted(name, args, rule=same_unit):
    """Values of ``args``, converted following ``rule``, and the result unit."""
    values, units = zip(*map(split, args))

    if not any(map(_is_quantity_unit, units)):
        return list(args), None

    converters, unit = plan(rule, units, (("name", name),))

    return list(map(apply_converter, values, converters)), unit


def _elementwise(xp, name, rule):
    """f(*arrays) with converters and result unit given by ``rule``."""
    f = getattr(xp, name)

    # The standard defines no keyword arguments for element-wise functions.
    def func(*args):
        values, unit = _converted(name, args, rule)
        return _wrap(f(*values), unit, args)

    return func


def _reject(xp, name):
    """f(*arrays), where no argument can be a quantity (e.g. logical_and)."""
    f = getattr(xp, name)

    def func(*args, **kwargs):
        _check_arguments(name, args, kwargs)
        return f(*args, **kwargs)

    return func


def _same(xp, name):
    """f(x, ...) on the value of x, with the unit of x (e.g. sum, reshape)."""
    f = getattr(xp, name)

    def func(x, /, *args, **kwargs):
        _check_arguments(name, args, kwargs)
        value, unit = split(x)

        if not _is_quantity_unit(unit):
            return f(x, *args, **kwargs)

        return _wrap(f(value, *args, **kwargs), unit, (x,))

    return func


def _plain(xp, name):
    """f(x, ...) on the value of x, giving a plain array (e.g. argmax, argsort)."""
    f = getattr(xp, name)

    def func(x, /, *args, **kwargs):
        _check_arguments(name, args, kwargs)
        return f(split(x)[0], *args, **kwargs)

    return func


def _function(xp, name):
    """The units-aware version of ``xp.<name>``, made from the rule tables."""
    if name in PASSTHROUGH:
        return getattr(xp, name)

    # Only make functions that the wrapped namespace has.
    getattr(xp, name)

    if name in UNARY_RULES:
        function = _elementwise(xp, name, UNARY_RULES[name])
    elif name in BINARY_RULES:
        function = _elementwise(xp, name, BINARY_RULES[name])
    elif name in REJECT_QUANTITY:
        function = _reject(xp, name)
    elif name in REDUCTION_SAME or name in SHAPE_SAME:
        function = _same(xp, name)
    elif name in REDUCTION_PLAIN or name in SHAPE_PLAIN:
        function = _plain(xp, name)
    else:
        raise AttributeError(
            f"'{name}' is not supported for quantities, since how it transforms "
            "units has not been defined."
        )

    function.__name__ = name
    function.__qualname__ = f"QuantityNamespace.{name}"

    return function


class QuantityNamespace:
    """An Array API namespace for quantities, wrapping that of their values.

    Functions are made from the rule tables when they are first used, and kept.
    """

    def __init__(self, xp):
        self._xp = xp
        # Used by array_api_compat.
        self.__name__ = f"{type(self).__name__}({xp.__name__})"

    def __getattr__(self, name):
        # Only called for names without a function made earlier.
        if name.startswith("_") and name not in PASSTHROUGH:
            raise AttributeError(name)

        function = _function(self._xp, name)
        setattr(self, name, function)

        return function

    def __repr__(self):
        return f"<{type(self).__name__} for {self._xp.__name__}>"


@functools.cache
def units_namespace(xp):
    """The Array API namespace for quantities with values in namespace ``xp``.

    Its functions accept quantities as well as plain arrays, and return
    quantities when the result has a unit. Functions for which it has not been
    defined how they transform units raise `AttributeError` rather than drop
    units.
    """
    return QuantityNamespace(xp)


_NAMESPACE_BY_TYPE = {}


def namespace_for(value, api_version=None):
    """The namespace for quantities with ``value``, cached by its type."""
    if api_version is None:
        try:
            return _NAMESPACE_BY_TYPE[type(value)]
        except KeyError:
            pass

    namespace = units_namespace(
        array_api_compat.array_namespace(value, api_version=api_version)
    )

    if api_version is None:
        _NAMESPACE_BY_TYPE[type(value)] = namespace

    return namespace
