# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A quantity class for any array that follows the Array API standard."""

from __future__ import annotations

import numbers

import array_api_compat
import numpy as np

from astropy.units.core import (
    CompositeUnit,
    Unit,
    UnitBase,
    dimensionless_unscaled,
)
from astropy.units.errors import UnitsError
from astropy.units.quantity import Quantity as _LegacyQuantity
from astropy.units.quantity import _upcast_int_to_float
from astropy.units.quantity_helper import namespace as _namespace
from astropy.units.quantity_helper import numpy_compat as _numpy_compat
from astropy.units.quantity_helper.rules import converter
from astropy.utils.compat.array_api import array_namespace, default_floating_dtype

__all__ = ["Quantity"]


def _as_unit(unit):
    """Convert ``unit`` to a unit.

    Function units and structured units are not yet supported.
    """
    unit = Unit(unit)

    # TODO: this is only here because the unit refactor does not yet support function
    #  or structured units.
    if not isinstance(unit, UnitBase):
        raise NotImplementedError(
            f"quantities with {type(unit).__name__} units are not yet supported."
        )

    return unit


def _is_operand(x):
    """Whether ``x`` can be combined with a quantity in arithmetic."""
    return isinstance(
        x, (Quantity, _LegacyQuantity, numbers.Number)
    ) or array_api_compat.is_array_api_obj(x)


def _binary_op(op_name, name, reflected=False):
    """The operator ``op_name``, calling the binary namespace function ``name``."""

    def op(self, other):
        if not _is_operand(other):
            return NotImplemented

        func = getattr(self.__array_namespace__(), name)

        return func(other, self) if reflected else func(self, other)

    op.__name__ = op_name
    op.__qualname__ = f"Quantity.{op_name}"

    return op


class Quantity:
    """An array with a unit, for any array that follows the Array API standard.

    Parameters
    ----------
    value : array-like or quantity
        The numerical value. Arrays are used with their own array library; Python
        numbers and sequences become numpy arrays. If ``value`` is a quantity, it
        is converted to ``unit`` if given.
    unit : unit-like, optional
        The unit; dimensionless if not given (and ``value`` is not a quantity).
    dtype : dtype, optional
        The data type. If not given, integer values are converted to the
        default floating point type of the array library, as set by the
        ``quantity_convert_int_to_float`` configuration item. If `None`, the
        data type of the value is kept. If `numpy.inexact`, integer values are
        converted whatever the configuration says.
    copy : bool, optional
        Whether to copy ``value``; by default it is copied.
    device : device, optional
        The device to place the value on.

    Notes
    -----
    Whether the value can be changed in place (e.g. ``q[0] = 1 * u.m``)
    depends on the array library. Augmented assignments like ``q *= u.s`` give
    a new quantity.
    """

    __slots__ = ("_unit", "_value")

    # Quantities are mutable, and == gives an array.
    __hash__ = None

    def __init__(self, value, unit=None, dtype=np._NoValue, copy=True, *, device=None):
        if isinstance(value, Quantity):
            if unit is not None:
                value = value.to(unit, copy=False)

            value, unit = value._value, value._unit
        elif isinstance(value, _LegacyQuantity):
            unit = value.unit if unit is None else _as_unit(unit)
            value = value.to_value(unit)
        else:
            unit = dimensionless_unscaled if unit is None else _as_unit(unit)

        upcast = dtype is np._NoValue or dtype is np.inexact
        xp = array_namespace(value)
        value = xp.asarray(
            value, dtype=None if upcast else dtype, device=device, copy=copy
        )

        if (
            upcast
            and xp.isdtype(value.dtype, "integral")
            and _upcast_int_to_float(dtype)
        ):
            device = array_api_compat.device(value)
            value = xp.astype(value, default_floating_dtype(xp, device))

        self._value = value
        self._unit = unit

    @classmethod
    def _new(cls, value, unit):
        """Create a quantity from an array and a unit, without any checks.

        All results are created through this method, ensuring subclasses get
        results of their own type.
        """
        if isinstance(value, np.generic):
            # Scalars are 0-d arrays.
            value = np.asarray(value)

        self = object.__new__(cls)
        self._value = value
        self._unit = unit

        return self

    @property
    def value(self):
        """The numerical value, as an array of the underlying array library."""
        return self._value

    @property
    def unit(self):
        """The unit."""
        return self._unit

    def __array_namespace__(self, *, api_version=None):
        return _namespace.namespace_for(self._value, api_version)

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        return _numpy_compat.array_ufunc(function, method, *inputs, **kwargs)

    def __jax_array__(self):
        # Quantities cannot be converted to JAX arrays, see open issue.
        # Returns the quantity itself to make JAX operators defer to it;
        # leaves conversion to __array__, which raises unless dimensionless.
        return self

    @property
    def dtype(self):
        return self._value.dtype

    @property
    def shape(self):
        return self._value.shape

    @property
    def ndim(self):
        return self._value.ndim

    @property
    def size(self):
        return array_api_compat.size(self._value)

    @property
    def device(self):
        return array_api_compat.device(self._value)

    def to_device(self, device, /, *, stream=None):
        """Return a copy of the quantity on ``device``.

        Parameters
        ----------
        device : device
            The device to move the value to; see the Array API standard.
        stream : int or object, optional
            A stream for the transfer, for array libraries that support it.
        """
        value = array_api_compat.to_device(self._value, device, stream=stream)
        return self._new(value, self._unit)

    @property
    def T(self):
        return self._new(self._value.T, self._unit)

    @property
    def mT(self):
        return self.__array_namespace__().matrix_transpose(self)

    def __getitem__(self, key):
        return self._new(self._value[key], self._unit)

    def __setitem__(self, key, value):
        value, unit = _namespace.split(value)
        conv = converter(unit, self._unit, "setitem")
        value = _namespace.apply_converter(value, conv)
        xp = array_namespace(self._value)

        if xp.isdtype(self.dtype, "integral"):
            # Allow values that are exactly integers, but not ones that would
            # be truncated (e.g. 50 cm into an integer array in m).
            value = xp.asarray(value)

            if not xp.isdtype(value.dtype, ("integral", "bool")) and not bool(
                xp.all(xp.astype(value, self.dtype) == value)
            ):
                raise TypeError(
                    "cannot convert value type to array type without precision loss"
                )

        self._value[key] = value

    def __len__(self):
        if self.ndim == 0:
            raise TypeError(
                f"'{type(self).__name__}' object with a scalar value has no len()"
            )
        return self.shape[0]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError(
                f"'{type(self).__name__}' object with a scalar value is not iterable"
            )
        return (self[i] for i in range(self.shape[0]))

    __add__ = _binary_op("__add__", "add")
    __radd__ = _binary_op("__radd__", "add", reflected=True)
    __sub__ = _binary_op("__sub__", "subtract")
    __rsub__ = _binary_op("__rsub__", "subtract", reflected=True)
    __floordiv__ = _binary_op("__floordiv__", "floor_divide")
    __rfloordiv__ = _binary_op("__rfloordiv__", "floor_divide", reflected=True)
    __mod__ = _binary_op("__mod__", "remainder")
    __rmod__ = _binary_op("__rmod__", "remainder", reflected=True)
    __matmul__ = _binary_op("__matmul__", "matmul")
    __rmatmul__ = _binary_op("__rmatmul__", "matmul", reflected=True)
    __pow__ = _binary_op("__pow__", "pow")
    __rpow__ = _binary_op("__rpow__", "pow", reflected=True)
    __lt__ = _binary_op("__lt__", "less")
    __le__ = _binary_op("__le__", "less_equal")
    __gt__ = _binary_op("__gt__", "greater")
    __ge__ = _binary_op("__ge__", "greater_equal")

    def __neg__(self):
        return self.__array_namespace__().negative(self)

    def __pos__(self):
        return self.__array_namespace__().positive(self)

    def __abs__(self):
        return self.__array_namespace__().abs(self)

    def __mul__(self, other):
        if isinstance(other, UnitBase):
            return self._new(self._value, self._unit * other)

        if not _is_operand(other):
            return NotImplemented

        return self.__array_namespace__().multiply(self, other)

    def __rmul__(self, other):
        if isinstance(other, UnitBase):
            return self._new(self._value, other * self._unit)

        if not _is_operand(other):
            return NotImplemented

        return self.__array_namespace__().multiply(other, self)

    def __truediv__(self, other):
        if isinstance(other, UnitBase):
            return self._new(self._value, self._unit / other)

        if not _is_operand(other):
            return NotImplemented

        return self.__array_namespace__().divide(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, UnitBase):
            result = self.__array_namespace__().divide(1.0, self)

            return result._new(result._value, other * result._unit)

        if not _is_operand(other):
            return NotImplemented

        return self.__array_namespace__().divide(other, self)

    def __eq__(self, other):
        # As for numpy arrays, incomparable quantities are simply not equal.
        if not _is_operand(other):
            return NotImplemented

        try:
            return self.__array_namespace__().equal(self, other)
        except UnitsError:
            return False

    def __ne__(self, other):
        if not _is_operand(other):
            return NotImplemented

        try:
            return self.__array_namespace__().not_equal(self, other)
        except UnitsError:
            return True

    def __bool__(self):
        raise ValueError(
            f"{type(self).__name__} truthiness is ambiguous, especially for "
            "logarithmic units and temperatures. Use explicit comparisons."
        )

    def _python_scalar(self, kind):
        try:
            return self.to_value(dimensionless_unscaled)
        except UnitsError:
            raise TypeError(
                "only dimensionless scalar quantities can be converted to "
                f"Python {kind}s"
            ) from None

    def __float__(self):
        return float(self._python_scalar("float"))

    def __int__(self):
        return int(self._python_scalar("int"))

    def __complex__(self):
        return complex(self._python_scalar("complex"))

    def __index__(self):
        # No scaling for indices.
        if self._unit.is_unity():
            return self._value.__index__()

        raise TypeError(
            "only integer dimensionless scalar quantities can be converted to an index"
        )

    def __array__(self, dtype=None, copy=None):
        # Only for units that are dimensionless without any equivalencies.
        try:
            value = self._to_value(dimensionless_unscaled, None)
        except UnitsError:
            raise TypeError(
                f"a quantity with unit '{self._unit}' cannot be converted to a "
                "numpy array, since that would drop the unit; use "
                "q.to_value(unit) to get the value in a given unit."
            ) from None

        return np.asarray(value, dtype=dtype, copy=copy)

    def to(self, unit, equivalencies=[], copy=True):
        """Return a new quantity converted to ``unit``.

        Parameters
        ----------
        unit : unit-like
            The unit to convert to.
        equivalencies : list of tuple, optional
            Equivalencies to use if a direct conversion is not possible. If
            `None`, no equivalencies are used, not even enabled ones.
        copy : bool, optional
            If `True` (default), the value is copied. Otherwise, a copy is
            only made if necessary.
        """
        unit = _as_unit(unit)
        value = self._to_value(unit, equivalencies)

        if copy and value is self._value:
            value = array_namespace(value).asarray(value, copy=True)

        return self._new(value, unit)

    def to_value(self, unit=None, equivalencies=[]):
        """The numerical value, possibly in a different unit.

        Parameters
        ----------
        unit : unit-like, optional
            The unit in which the value should be given. If not given or `None`,
            use the current unit.
        equivalencies : list of tuple, optional
            A list of equivalence pairs to try if the units are not directly
            convertible (see :ref:`astropy:unit_equivalencies`). If `None`, no
            equivalencies are used, not even any enabled globally or within a
            context.

        Returns
        -------
        value : array
            The value in the unit given, as an array of the underlying array
            library. It is the array of the quantity itself if no conversion
            was necessary.

        See Also
        --------
        to : Get a new quantity in a different unit.
        """
        if unit is None:
            return self._value

        return self._to_value(_as_unit(unit), equivalencies)

    def _to_value(self, unit, equivalencies):
        if unit is self._unit:
            return self._value

        try:
            scale = self._unit._to(unit)
        except UnitsError:
            conv = self._unit._get_converter(unit, equivalencies=equivalencies)
            return conv(self._value)

        if scale == 1.0:
            return self._value

        return _namespace.apply_converter(self._value, float(scale))

    def __lshift__(self, other):
        try:
            unit = _as_unit(other)
        except (TypeError, ValueError):
            return NotImplemented

        return self.to(unit, copy=False)

    def decompose(self, bases=()):
        """Return a quantity in terms of irreducible units.

        Parameters
        ----------
        bases : sequence of `~astropy.units.UnitBase`, optional
            The bases to decompose into; by default, the irreducible units.
        """
        unit = self._unit.decompose(bases=bases)
        value = self._value

        if unit.scale != 1.0:
            value = _namespace.apply_converter(value, float(unit.scale))

        unit = CompositeUnit(1, unit.bases, unit.powers, _error_check=False)

        return self._new(value, unit)

    def __str__(self):
        unit = self._unit.to_string()

        return f"{self._value}{' ' + unit if unit else ''}"

    def __repr__(self):
        return f"<{type(self).__name__} {self}>"
