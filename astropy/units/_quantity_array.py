# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A quantity class for any array that follows the Array API standard."""

from __future__ import annotations

import array_api_compat
import numpy as np

from astropy.units.core import CompositeUnit, Unit, UnitBase, dimensionless_unscaled
from astropy.units.errors import UnitsError
from astropy.units.quantity import Quantity as _LegacyQuantity
from astropy.units.quantity_helper import namespace as _namespace
from astropy.utils.compat.array_api import array_namespace, default_floating_dtype

__all__ = ["Quantity"]


def _as_unit(unit):
    """Convert ``unit`` to a unit, rejecting units that are not yet supported."""
    unit = Unit(unit)

    if not isinstance(unit, UnitBase):
        raise NotImplementedError(
            f"quantities with {type(unit).__name__} units are not yet supported."
        )

    return unit


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
        The data type. If not given, integer and boolean values are
        converted to the default floating point type of the array library.
    copy : bool, optional
        Whether to copy ``value``; by default it is copied.
    device : device, optional
        The device to place the value on.

    Notes
    -----
    The unit of a quantity never changes.
    """

    __slots__ = ("_unit", "_value")

    # numpy functions should not silently drop the unit.
    __array_ufunc__ = None

    def __init__(self, value, unit=None, dtype=None, copy=True, *, device=None):
        if isinstance(value, Quantity):
            if unit is not None:
                value = value.to(unit)

            value, unit = value._value, value._unit
        elif isinstance(value, _LegacyQuantity):
            unit = value.unit if unit is None else _as_unit(unit)
            value = value.to_value(unit)
        else:
            unit = dimensionless_unscaled if unit is None else _as_unit(unit)

        xp = array_namespace(value)
        value = xp.asarray(value, dtype=dtype, device=device, copy=copy)

        if dtype is None and xp.isdtype(value.dtype, ("integral", "bool")):
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
        value = array_api_compat.to_device(self._value, device, stream=stream)
        return self._new(value, self._unit)

    @property
    def T(self):
        return self._new(self._value.T, self._unit)

    def __getitem__(self, key):
        return self._new(self._value[key], self._unit)

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
        try:
            value = self.to_value(dimensionless_unscaled)
        except UnitsError:
            raise TypeError(
                f"a quantity with unit '{self._unit}' cannot be converted to a "
                "numpy array, since that would drop the unit; use "
                "q.to_value(unit) to get the value in a given unit."
            ) from None

        return np.asarray(value, dtype=dtype, copy=copy)

    def to(self, unit, equivalencies=[]):
        """Return a new quantity converted to ``unit``.

        Parameters
        ----------
        unit : unit-like
            The unit to convert to.
        equivalencies : list of tuple, optional
            Equivalencies to use if a direct conversion is not possible. If
            `None`, no equivalencies are used, not even enabled ones.
        """
        unit = _as_unit(unit)

        return self._new(self._to_value(unit, equivalencies), unit)

    def to_value(self, unit=None, equivalencies=[]):
        """The value in ``unit``, as an array of the underlying array library.

        If ``unit`` is not given, the value is returned in the quantity's own
        unit.
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

        return self.to(unit)

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
