# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Converters for Quantity."""

import threading

import numpy as np

from astropy.units.core import dimensionless_unscaled
from astropy.units.errors import UnitConversionError, UnitsError, UnitTypeError

__all__ = [
    "UFUNC_HELPERS",
    "UNSUPPORTED_UFUNCS",
    "can_have_arbitrary_unit",
    "check_output",
    "check_partial_output",
    "check_quantity_output",
    "converters_and_unit",
    "writes_partially",
]


class UfuncHelpers(dict):
    """Registry of unit conversion functions to help ufunc evaluation.

    Based on dict for quick access, but with a missing method to load
    helpers for additional modules such as scipy.special and erfa.

    Such modules should be registered using ``register_module``.
    """

    def __init__(self, *args, **kwargs):
        self.modules = {}
        self.UNSUPPORTED = set()  # Upper-case for backwards compatibility
        self._lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def register_module(self, module, names, importer):
        """Register (but do not import) a set of ufunc helpers.

        Parameters
        ----------
        module : str
            Name of the module with the ufuncs (e.g., 'scipy.special').
        names : iterable of str
            Names of the module ufuncs for which helpers are available.
        importer : callable
            Function that imports the ufuncs and returns a dict of helpers
            keyed by those ufuncs.  If the value is `None`, the ufunc is
            explicitly *not* supported.
        """
        with self._lock:
            self.modules[module] = {"names": names, "importer": importer}

    def import_module(self, module):
        """Import the helpers from the given module using its helper function.

        Parameters
        ----------
        module : str
            Name of the module. Has to have been registered beforehand.
        """
        with self._lock:
            module_info = self.modules.pop(module)
            self.update(module_info["importer"]())

    def __missing__(self, ufunc):
        """Called if a ufunc is not found.

        Check if the ufunc is in any of the available modules, and, if so,
        import the helpers for that module.
        """
        with self._lock:
            # Check if it was loaded while we waited for the lock
            if ufunc in self:
                return self[ufunc]

            if ufunc in self.UNSUPPORTED:
                raise TypeError(f"Cannot use ufunc '{ufunc.__name__}' with quantities")

            for module, module_info in list(self.modules.items()):
                if ufunc.__name__ in module_info["names"]:
                    # A ufunc with the same name is supported by this module.
                    # Of course, this doesn't necessarily mean it is the
                    # right module. So, we try let the importer do its work.
                    # If it fails (e.g., for `scipy.special`), then that's
                    # fine, just raise the TypeError.  If it succeeds, but
                    # the ufunc is not found, that is also fine: we will
                    # enter __missing__ again and either find another
                    # module or get the TypeError there.
                    try:
                        self.import_module(module)
                    except ImportError:  # pragma: no cover
                        pass
                    else:
                        return self[ufunc]

        raise TypeError(
            f"unknown ufunc {ufunc.__name__}.  If you believe this ufunc "
            "should be supported, please raise an issue on "
            "https://github.com/astropy/astropy"
        )

    def __setitem__(self, key, value):
        # Implementation note: in principle, we could just let `None`
        # mean that something is not implemented, but this means an
        # extra if clause for the output, slowing down the common
        # path where a ufunc is supported.
        with self._lock:
            if value is None:
                self.UNSUPPORTED |= {key}
                self.pop(key, None)
            else:
                super().__setitem__(key, value)
                self.UNSUPPORTED -= {key}


UFUNC_HELPERS = UfuncHelpers()
UNSUPPORTED_UFUNCS = UFUNC_HELPERS.UNSUPPORTED


def can_have_arbitrary_unit(value):
    """Test whether the items in value can have arbitrary units.

    Numbers whose value does not change upon a unit change, i.e.,
    zero, infinity, or not-a-number

    Parameters
    ----------
    value : number or array

    Returns
    -------
    bool
        `True` if each member is either zero or not finite, `False` otherwise
    """
    return np.all(np.logical_or(np.equal(value, 0.0), ~np.isfinite(value)))


def converters_and_unit(function, method, *args):
    """Determine the required converters and the unit of the ufunc result.

    Converters are functions required to convert to a ufunc's expected unit,
    e.g., radian for np.sin; or to ensure units of two inputs are consistent,
    e.g., for np.add.  In these examples, the unit of the result would be
    dimensionless_unscaled for np.sin, and the same consistent unit for np.add.

    Parameters
    ----------
    function : `~numpy.ufunc`
        Numpy universal function
    method : str
        Method with which the function is evaluated, e.g.,
        '__call__', 'reduce', etc.
    *args :  `~astropy.units.Quantity` or ndarray subclass
        Input arguments to the function

    Raises
    ------
    TypeError : when the specified function cannot be used with Quantities
        (e.g., np.logical_or), or when the routine does not know how to handle
        the specified function (in which case an issue should be raised on
        https://github.com/astropy/astropy).
    UnitTypeError : when the conversion to the required (or consistent) units
        is not possible.
    """
    # Check whether we support this ufunc, by getting the helper function
    # (defined in helpers) which returns a list of function(s) that convert the
    # input(s) to the unit required for the ufunc, as well as the unit the
    # result will have (a tuple of units if there are multiple outputs).
    ufunc_helper = UFUNC_HELPERS[function]

    if method == "__call__" or (method == "outer" and function.nin == 2):
        # Find out the units of the arguments passed to the ufunc; usually,
        # at least one is a quantity, but for two-argument ufuncs, the second
        # could also be a Numpy array, etc.  These are given unit=None.
        units = [getattr(arg, "unit", None) for arg in args]

        # Determine possible conversion functions, and the result unit.
        converters, result_unit = ufunc_helper(function, *units)

        if any(converter is False for converter in converters):
            # for multi-argument ufuncs with a quantity and a non-quantity,
            # the quantity normally needs to be dimensionless, *except*
            # if the non-quantity can have arbitrary unit, i.e., when it
            # is all zero, infinity or NaN.  In that case, the non-quantity
            # can just have the unit of the quantity
            # (this allows, e.g., `q > 0.` independent of unit)
            try:
                # Don't fold this loop in the test above: this rare case
                # should not make the common case slower.
                for i, converter in enumerate(converters):
                    if converter is not False:
                        continue
                    if can_have_arbitrary_unit(args[i]):
                        converters[i] = None
                    else:
                        raise UnitConversionError(
                            f"Can only apply '{function.__name__}' function to "
                            "dimensionless quantities when other argument is not "
                            "a quantity (unless the latter is all zero/infinity/nan)."
                        )
            except TypeError:
                # _can_have_arbitrary_unit failed: arg could not be compared
                # with zero or checked to be finite. Then, ufunc will fail too.
                raise TypeError(
                    "Unsupported operand type(s) for ufunc {}: '{}'".format(
                        function.__name__,
                        ",".join([arg.__class__.__name__ for arg in args]),
                    )
                )

        # In the case of np.power and np.float_power, the unit itself needs to
        # be modified by an amount that depends on one of the input values,
        # so we need to treat this as a special case.
        # TODO: find a better way to deal with this.
        if result_unit is False:
            if units[0] is None or units[0] == dimensionless_unscaled:
                result_unit = dimensionless_unscaled
            else:
                if units[1] is None:
                    p = args[1]
                else:
                    p = args[1].to(dimensionless_unscaled).value

                try:
                    result_unit = units[0] ** p
                except ValueError as exc:
                    # Changing the unit does not work for, e.g., array-shaped
                    # power, but this is OK if we're (scaled) dimensionless.
                    try:
                        converters[0] = units[0].get_converter(dimensionless_unscaled)
                    except UnitConversionError:
                        raise exc
                    else:
                        result_unit = dimensionless_unscaled

    else:  # methods for which the unit should stay the same
        nin = function.nin
        unit = getattr(args[0], "unit", None)

        if method in {"at", "reduceat"}:
            index_unit = getattr(args[1], "unit", None)
            if index_unit is not None and not index_unit.is_unity():
                raise UnitTypeError(
                    f"Cannot use '{method}' method on ufunc {function.__name__} "
                    f"with indices that have a unit (got '{index_unit}')."
                )

        if method == "at" and nin <= 2:
            if nin == 1:
                units = [unit]
            else:
                units = [unit, getattr(args[2], "unit", None)]

            converters, result_unit = ufunc_helper(function, *units)

            # ensure there is no 'converter' for indices (2nd argument)
            converters.insert(1, None)

        elif method in {"reduce", "accumulate", "reduceat"} and nin == 2:
            converters, result_unit = ufunc_helper(function, unit, unit)
            converters = converters[:1]
            if method == "reduceat":
                # add 'scale' for indices (2nd argument)
                converters += [None]

        else:
            if method in {"reduce", "accumulate", "reduceat", "outer"} and nin != 2:
                raise ValueError(f"{method} only supported for binary functions")

            raise TypeError(
                f"Unexpected ufunc method {method}.  If this should work, please "
                "raise an issue on https://github.com/astropy/astropy"
            )

        # for all but __call__ method, scaling is not allowed
        if (
            unit is None
            and result_unit is not None
            and result_unit is not dimensionless_unscaled
            and not result_unit.is_unity()
        ):
            raise UnitTypeError(
                f"Cannot use '{method}' method on ufunc {function.__name__} with "
                "a non-Quantity instance as the result would have unit "
                f"'{result_unit}'."
            )

        if unit is not None and result_unit is None:
            raise TypeError(
                f"Cannot use '{method}' method on ufunc {function.__name__} with a "
                "Quantity instance as the result is not a Quantity."
            )

        if converters[0] is not None or (
            unit is not None
            and unit is not result_unit
            and (not result_unit.is_equivalent(unit) or result_unit.to(unit) != 1.0)
        ):
            # NOTE: this cannot be the more logical UnitTypeError, since
            # then things like np.cumprod will not longer fail (they check
            # for TypeError).
            raise UnitsError(
                f"Cannot use '{method}' method on ufunc {function.__name__} with a "
                "Quantity instance as it would change the unit."
            )

    return converters, result_unit


def writes_partially(method, kwargs):
    """Whether a ufunc call may leave elements of its output unwritten."""
    where = kwargs.get("where", True)

    if method != "__call__" or hasattr(where, "unit"):
        # For reductions, 'where' selects the inputs, so the output is fully
        # written. Masks with a unit are rejected by numpy itself.
        return False

    return not np.all(where)


def check_partial_output(output, unit, function=None):
    """Check that writing only part of ``output`` keeps its unit.

    If a ufunc is given ``where``, elements of ``output`` that are not written
    keep their value, so their unit cannot change.

    Raises
    ------
    UnitsError : If the unit of ``output`` differs from ``unit``.
    """
    output_unit = getattr(output, "unit", None)

    if output_unit is not None and unit is not None and output_unit != unit:
        raise UnitsError(
            "Cannot store output with unit '{}'{} in an output with unit '{}' "
            "when 'where' is given, since the elements not written would change "
            "unit.".format(
                unit,
                f" from {function.__name__} function" if function is not None else "",
                output_unit,
            )
        )


def check_quantity_output(output, unit, function=None):
    """Check that a result with ``unit`` can be stored in quantity ``output``.

    Raises
    ------
    TypeError : If the result is not a quantity (``unit`` is `None`).
    UnitTypeError : If the class of ``output`` cannot hold ``unit`` (e.g., a
        unit that is not an angle for `~astropy.coordinates.Angle`).
    """
    from_function = f" from {function.__name__} function" if function else ""

    if unit is None:
        raise TypeError(
            f"Cannot store non-quantity output{from_function} in "
            f"{type(output)} instance"
        )

    q_cls, subok = output.__quantity_subclass__(unit)

    if not (subok or q_cls is type(output)):
        raise UnitTypeError(
            f"Cannot store output with unit '{unit}'{from_function} in "
            f"{type(output)} instance.  Use {q_cls} instance instead."
        )


def check_output(output, unit, inputs, function=None):
    """Check that function output can be stored in the output array given.

    Parameters
    ----------
    output : array or `~astropy.units.Quantity` or tuple
        Array that should hold the function output (or tuple of such arrays).
    unit : `~astropy.units.Unit` or None, or tuple
        Unit that the output will have, or `None` for pure numbers (should be
        tuple of same if output is a tuple of outputs).
    inputs : tuple
        Any input arguments.  These should be castable to the output.
    function : callable
        The function that will be producing the output.  If given, used to
        give a more informative error message.

    Returns
    -------
    arrays : ndarray view or tuple thereof
        The view(s) is of ``output``.

    Raises
    ------
    UnitTypeError : If ``unit`` is inconsistent with the class of ``output``

    TypeError : If the ``inputs`` cannot be cast safely to ``output``.
    """
    if isinstance(output, tuple):
        return tuple(
            check_output(output_, unit_, inputs, function)
            for output_, unit_ in zip(output, unit)
        )

    # ``None`` indicates no actual array is needed.  This can happen, e.g.,
    # with np.modf(a, out=(None, b)).
    if output is None:
        return None

    if hasattr(output, "__quantity_subclass__"):
        check_quantity_output(output, unit, function)

        # check we can handle the dtype (e.g., that we are not int
        # when float is required).  Note that we only do this for Quantity
        # output; for array output, we defer to numpy's default handling.
        # Also, any structured dtype are ignored (likely erfa ufuncs).
        # TODO: make more logical; is this necessary at all?
        if inputs and not output.dtype.names:
            result_type = np.result_type(*inputs)
            if not (
                result_type.names
                or np.can_cast(result_type, output.dtype, casting="same_kind")
            ):
                raise TypeError(
                    "Arguments cannot be cast safely to inplace "
                    f"output with dtype={output.dtype}"
                )
        # Turn into ndarray, so we do not loop into array_wrap/array_ufunc
        # if the output is used to store results of a function.
        return output.view(np.ndarray)

    else:
        # output is not a Quantity. Some ndarray subclasses (such as astropy
        # ``Column``) carry a ``unit`` attribute; for those, storing is fine
        # if the result unit is identical to the output's unit.  For anything
        # else (like ndarray proper), we require the output to be
        # dimensionless.
        if not (
            unit is None or unit == getattr(output, "unit", dimensionless_unscaled)
        ):
            raise UnitTypeError(
                "Cannot store quantity with dimension "
                "{}in a non-Quantity instance.".format(
                    f"resulting from {function.__name__} function "
                    if function is not None
                    else ""
                )
            )

        return output
