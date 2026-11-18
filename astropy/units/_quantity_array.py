# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A quantity class for any array that follows the Array API standard."""

from __future__ import annotations

import array_api_compat
import numpy as np

from astropy.units.core import Unit, UnitBase, dimensionless_unscaled
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
    value : array-like
        The numerical value. Arrays are used with their own array library; Python
        numbers and sequences become numpy arrays.
    unit : unit-like, optional
        The unit; dimensionless if not given.
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

    def __init__(self, value, unit=None, dtype=None, copy=True, *, device=None):
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

    def __str__(self):
        unit = self._unit.to_string()
        return f"{self._value}{' ' + unit if unit else ''}"

    def __repr__(self):
        return f"<{type(self).__name__} {self}>"
