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
import math
import numbers
import operator

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
    array_power,
    converter,
    dimensionless_to_dimensionless,
    is_literal_arbitrary,
    multiplication,
    plan,
    prod_rule,
    same_unit,
    same_unit_to_plain,
    static_power,
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


def _values(arrays):
    return [split(array)[0] for array in arrays]


def _with_own_unit(result, array):
    """``result`` with the unit of ``array``, as an instance of its class."""
    unit = split(array)[1]

    if not _is_quantity_unit(unit):
        return result

    if isinstance(array, _quantity_array.Quantity):
        return array._new(result, unit)

    return _quantity_array.Quantity._new(result, unit)


def _each_own_unit(results, arrays):
    """Results that each have the unit and class of the corresponding input."""
    return type(results)(map(_with_own_unit, results, arrays))


def _with_values_unit(result, x):
    """A named tuple from unique_counts, etc, with ``values`` in the unit of x."""
    unit = split(x)[1]

    if not _is_quantity_unit(unit):
        return result

    return result._replace(values=_wrap(result.values, unit, (x,)))


def _created(value, unit):
    """A quantity in ``unit``, or the plain array if no unit is given."""
    return value if unit is None else _quantity_array.Quantity._new(value, unit)


def _like_unit(unit, x_unit, fill_unit=None):
    """The unit for a ``*_like`` function."""
    if unit is not None:
        return _quantity_array._as_unit(unit)

    for other in (x_unit, fill_unit):
        if _is_quantity_unit(other):
            return other

    return None


def _fill_value(name, fill_value, fill_unit, unit, explicit):
    """fill_value converted to ``unit``.

    A plain fill value is taken to be in ``unit`` if that is given explicitly;
    otherwise, it is treated as for any other function.
    """
    if explicit and not _is_quantity_unit(fill_unit):
        return fill_value

    return apply_converter(fill_value, converter(fill_unit, unit, name))


def _full(xp, shape, fill_value, *, dtype=None, device=None):
    """Like ``xp.full``, but also for array fill values.

    The standard only defines Python scalars as fill values. Arrays (e.g. the
    value of a quantity) are broadcast instead.
    """
    if isinstance(fill_value, (bool, int, float, complex)):
        return xp.full(shape, fill_value, dtype=dtype, device=device)

    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    fill_value = xp.astype(fill_value, fill_value.dtype if dtype is None else dtype)

    if device is not None:
        fill_value = array_api_compat.to_device(fill_value, device)

    # Copied, since broadcast arrays may not be writeable.
    return xp.asarray(xp.broadcast_to(fill_value, shape), copy=True)


def _python_scalar(value):
    """``value`` as a Python scalar."""
    if isinstance(value, (int, float)):
        return value

    try:
        return operator.index(value)
    except TypeError:
        pass

    try:
        return float(value)
    except Exception:
        return value


def _range_values(name, args, unit):
    """start, stop, step (any of which can be `None`) in a common unit.

    That unit is ``unit`` if given, else that of the first quantity. If no
    argument is a quantity, the values are taken to be in ``unit``.
    """
    unit = None if unit is None else _quantity_array._as_unit(unit)
    present = [arg for arg in args if arg is not None]
    values, units = zip(*map(split, present))
    quantity_units = [unit for unit in units if _is_quantity_unit(unit)]

    if not quantity_units:
        return list(args), unit

    if unit is None:
        unit = quantity_units[0]

    converted = iter(
        _python_scalar(apply_converter(value, converter(arg_unit, unit, name)))
        for value, arg_unit in zip(values, units)
    )

    return [None if arg is None else next(converted) for arg in args], unit


class QuantityNamespace:
    """An Array API namespace for quantities, wrapping that of their values.

    Functions whose unit handling depends on more than the units of their
    arguments are methods. All others are made from the rule tables when
    they are first used, and kept.
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

    def pow(self, x1, x2, /):
        """x1 ** x2; for an array exponent, x1 must be dimensionless."""
        v1, u1 = split(x1)

        if (
            _is_quantity_unit(u1)
            and isinstance(x2, numbers.Real)
            and not isinstance(x2, bool)
        ):
            # A Python exponent, so the result unit is known.
            (_,), unit = plan(static_power, (u1,), (("exponent", x2),))
            exponent = int(x2) if isinstance(x2, numbers.Integral) else float(x2)

            return _wrap(self._xp.pow(v1, exponent), unit, (x1,))

        values, unit = _converted("pow", (x1, x2), array_power)

        return _wrap(self._xp.pow(*values), unit, (x1, x2))

    def clip(self, x, /, min=None, max=None):
        """x clipped to min and max, which are converted to the unit of x."""
        bounds = [bound for bound in (min, max) if bound is not None]
        values, unit = _converted("clip", (x, *bounds))
        bounds = iter(values[1:])
        min = None if min is None else next(bounds)
        max = None if max is None else next(bounds)

        return _wrap(self._xp.clip(values[0], min=min, max=max), unit, (x,))

    def where(self, condition, x1, x2, /):
        """Elements of x1 or x2, converted to a common unit.

        The condition is a plain boolean array.
        """
        _check_arguments("where", (condition,), {})
        (v1, v2), unit = _converted("where", (x1, x2))

        return _wrap(self._xp.where(condition, v1, v2), unit, (x1, x2))

    def prod(self, x, /, *, axis=None, **kwargs):
        """Product, with unit to the power of the number of elements reduced over.

        If that number is not known (e.g. for lazy arrays), x has to be
        dimensionless.
        """
        _check_arguments("prod", (), kwargs)
        value, unit = split(x)

        if not _is_quantity_unit(unit):
            return self._xp.prod(x, axis=axis, **kwargs)

        shape = value.shape

        if axis is None:
            axes = range(len(shape))
        else:
            axes = (axis,) if isinstance(axis, int) else axis

        sizes = [shape[i] for i in axes]
        # Unknown sizes are None, or nan for dask.
        known = all(isinstance(size, int) for size in sizes)
        n = math.prod(sizes) if known else None
        (conv,), unit = plan(prod_rule, (unit,), (("n", n),))
        result = self._xp.prod(apply_converter(value, conv), axis=axis, **kwargs)

        return _wrap(result, unit, (x,))

    def cumulative_prod(self, x, /, **kwargs):
        """Cumulative product, which is only possible for dimensionless x."""
        _check_arguments("cumulative_prod", (), kwargs)
        (value,), unit = _converted(
            "cumulative_prod", (x,), dimensionless_to_dimensionless
        )

        return _wrap(self._xp.cumulative_prod(value, **kwargs), unit, (x,))

    def var(self, x, /, **kwargs):
        """Variance, with the unit of x squared."""
        _check_arguments("var", (), kwargs)
        value, unit = split(x)

        if not _is_quantity_unit(unit):
            return self._xp.var(x, **kwargs)

        (_,), unit = plan(static_power, (unit,), (("exponent", 2),))

        return _wrap(self._xp.var(value, **kwargs), unit, (x,))

    def diff(self, x, /, *, axis=-1, n=1, prepend=None, append=None):
        """Differences, with prepend and append converted to the unit of x."""
        extra = [array for array in (prepend, append) if array is not None]
        values, unit = _converted("diff", (x, *extra))
        extra = iter(values[1:])
        kwargs = {}

        if prepend is not None:
            kwargs["prepend"] = next(extra)

        if append is not None:
            kwargs["append"] = next(extra)

        result = self._xp.diff(values[0], axis=axis, n=n, **kwargs)

        return _wrap(result, unit, (x,))

    def tensordot(self, x1, x2, /, *, axes=2):
        """Tensor contraction, with the product of the units of x1 and x2."""
        _check_arguments("tensordot", (axes,), {})
        (v1, v2), unit = _converted("tensordot", (x1, x2), multiplication)

        return _wrap(self._xp.tensordot(v1, v2, axes=axes), unit, (x1, x2))

    def vecdot(self, x1, x2, /, *, axis=-1):
        """Vector dot product, with the product of the units of x1 and x2."""
        (v1, v2), unit = _converted("vecdot", (x1, x2), multiplication)

        return _wrap(self._xp.vecdot(v1, v2, axis=axis), unit, (x1, x2))

    def concat(self, arrays, /, *, axis=0):
        """Arrays joined along an existing axis, converted to a common unit."""
        arrays = tuple(arrays)
        values, unit = _converted("concat", arrays)

        return _wrap(self._xp.concat(values, axis=axis), unit, arrays)

    def stack(self, arrays, /, *, axis=0):
        """Arrays joined along a new axis, converted to a common unit."""
        arrays = tuple(arrays)
        values, unit = _converted("stack", arrays)

        return _wrap(self._xp.stack(values, axis=axis), unit, arrays)

    def unstack(self, x, /, *, axis=0):
        """Arrays split along an axis, each with the unit of x."""
        value, unit = split(x)
        results = self._xp.unstack(value, axis=axis)

        if not _is_quantity_unit(unit):
            return results

        return tuple(_wrap(result, unit, (x,)) for result in results)

    def broadcast_arrays(self, *arrays):
        """Arrays broadcast against each other, each keeping its unit."""
        results = self._xp.broadcast_arrays(*_values(arrays))

        return _each_own_unit(results, arrays)

    def meshgrid(self, *arrays, indexing="xy"):
        """Coordinate grids, each with the unit of its input."""
        results = self._xp.meshgrid(*_values(arrays), indexing=indexing)

        return _each_own_unit(results, arrays)

    def searchsorted(self, x1, x2, /, *, side="left", sorter=None):
        """Indices to insert x2 into x1, after conversion to a common unit."""
        _check_arguments("searchsorted", (sorter,), {})
        (v1, v2), _ = _converted("searchsorted", (x1, x2), same_unit_to_plain)

        return self._xp.searchsorted(v1, v2, side=side, sorter=sorter)

    def isin(self, x1, x2, /, *, invert=False):
        """Whether elements of x1 are in x2, after conversion to a common unit."""
        (v1, v2), _ = _converted("isin", (x1, x2), same_unit_to_plain)

        return self._xp.isin(v1, v2, invert=invert)

    def unique_all(self, x, /):
        """Unique elements, etc; the unique values have the unit of x."""
        return _with_values_unit(self._xp.unique_all(split(x)[0]), x)

    def unique_counts(self, x, /):
        """Unique elements and counts; the unique values have the unit of x."""
        return _with_values_unit(self._xp.unique_counts(split(x)[0]), x)

    def unique_inverse(self, x, /):
        """Unique elements and inverse indices; the values have the unit of x."""
        return _with_values_unit(self._xp.unique_inverse(split(x)[0]), x)

    def result_type(self, *arrays_and_dtypes):
        """The result data type, from the values of any quantities."""
        return self._xp.result_type(*_values(arrays_and_dtypes))

    def can_cast(self, from_, to, /):
        """Whether a cast is possible, from the value of any quantity."""
        return self._xp.can_cast(split(from_)[0], to)

    def finfo(self, type, /):
        """Floating point limits, from the value of any quantity."""
        return self._xp.finfo(split(type)[0])

    def iinfo(self, type, /):
        """Integer limits, from the value of any quantity."""
        return self._xp.iinfo(split(type)[0])

    def asarray(self, obj, /, *, unit=None, dtype=None, device=None, copy=None):
        """An array, which is a quantity if ``obj`` is one or ``unit`` is given.

        Unlike for `~astropy.units.Quantity`, the data type is that of
        ``obj``, as in the standard. Without ``unit``, plain values give a
        plain array (e.g. a boolean mask).
        """
        if isinstance(obj, (_quantity_array.Quantity, _LegacyQuantity)):
            unit = obj.unit if unit is None else _quantity_array._as_unit(unit)
            obj = obj.to_value(unit)
        elif unit is not None:
            unit = _quantity_array._as_unit(unit)

        value = self._xp.asarray(obj, dtype=dtype, device=device, copy=copy)

        return _created(value, unit)

    def arange(self, start, /, stop=None, step=None, *, unit=None, **kwargs):
        """Evenly spaced values; quantities are converted to a common unit.

        If not given, ``step`` is 1 in the unit of the result.
        """
        (start, stop, step), unit = _range_values("arange", (start, stop, step), unit)
        step = 1 if step is None else step

        return _created(self._xp.arange(start, stop, step, **kwargs), unit)

    def linspace(self, start, stop, /, num, *, unit=None, **kwargs):
        """Evenly spaced values; quantities are converted to a common unit."""
        (start, stop), unit = _range_values("linspace", (start, stop), unit)

        return _created(self._xp.linspace(start, stop, num, **kwargs), unit)

    def empty(self, shape, *, unit=None, dtype=None, device=None):
        """An uninitialized array, which is a quantity if ``unit`` is given."""
        value = self._xp.empty(shape, dtype=dtype, device=device)

        return _created(value, _like_unit(unit, None))

    def zeros(self, shape, *, unit=None, dtype=None, device=None):
        """An array of zeros, which is a quantity if ``unit`` is given."""
        value = self._xp.zeros(shape, dtype=dtype, device=device)

        return _created(value, _like_unit(unit, None))

    def ones(self, shape, *, unit=None, dtype=None, device=None):
        """An array of ones, which is a quantity if ``unit`` is given."""
        value = self._xp.ones(shape, dtype=dtype, device=device)

        return _created(value, _like_unit(unit, None))

    def eye(self, n_rows, n_cols=None, /, *, k=0, unit=None, dtype=None, device=None):
        """Ones on a diagonal; a quantity if ``unit`` is given."""
        value = self._xp.eye(n_rows, n_cols, k=k, dtype=dtype, device=device)

        return _created(value, _like_unit(unit, None))

    def full(self, shape, fill_value, *, unit=None, dtype=None, device=None):
        """An array filled with fill_value, in ``unit`` or that of fill_value."""
        value, fill_unit = split(fill_value)
        explicit = unit is not None
        unit = _like_unit(unit, fill_unit)
        value = _fill_value("full", value, fill_unit, unit, explicit)
        value = _full(self._xp, shape, value, dtype=dtype, device=device)

        return _created(value, unit)

    def empty_like(self, x, /, *, unit=None, dtype=None, device=None):
        """An uninitialized array like x, in ``unit`` or that of x."""
        value, x_unit = split(x)
        result = self._xp.empty_like(value, dtype=dtype, device=device)

        return _wrap(result, _like_unit(unit, x_unit), (x,))

    def zeros_like(self, x, /, *, unit=None, dtype=None, device=None):
        """An array of zeros like x, in ``unit`` or that of x."""
        value, x_unit = split(x)
        result = self._xp.zeros_like(value, dtype=dtype, device=device)

        return _wrap(result, _like_unit(unit, x_unit), (x,))

    def ones_like(self, x, /, *, unit=None, dtype=None, device=None):
        """An array of ones like x, in ``unit`` or that of x."""
        value, x_unit = split(x)
        result = self._xp.ones_like(value, dtype=dtype, device=device)

        return _wrap(result, _like_unit(unit, x_unit), (x,))

    def full_like(self, x, /, fill_value, *, unit=None, dtype=None, device=None):
        """An array like x filled with fill_value.

        The unit is ``unit`` if given, else that of x, else that of fill_value.
        """
        value, x_unit = split(x)
        fill_value, fill_unit = split(fill_value)
        explicit = unit is not None
        unit = _like_unit(unit, x_unit, fill_unit)
        fill_value = _fill_value("full_like", fill_value, fill_unit, unit, explicit)

        if isinstance(fill_value, (bool, int, float, complex)):
            result = self._xp.full_like(value, fill_value, dtype=dtype, device=device)
        else:
            result = _full(
                self._xp,
                value.shape,
                fill_value,
                dtype=value.dtype if dtype is None else dtype,
                device=array_api_compat.device(value) if device is None else device,
            )

        return _wrap(result, unit, (x,))


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
