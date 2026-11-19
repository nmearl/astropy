# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Support for the Python Array API standard.

Centralized all lookups of an array's namespace via `array_namespace`.
"""

from array_api_compat import array_namespace as _array_namespace
from array_api_compat import is_array_api_obj
from array_api_compat import numpy as _numpy_namespace

__all__ = ["array_namespace", "default_floating_dtype"]


def array_namespace(*values):
    """Return the Array API namespace of the arrays among ``values``.

    Values that are not arrays (Python scalars, lists, `None`, ...) are
    ignored. If none of the values is an array, the NumPy namespace is
    returned.

    Quantities can be mixed with plain arrays of the namespace of their
    values (e.g. masks or weights), in which case the namespace for
    quantities is returned.

    Parameters
    ----------
    *values
        Arrays and other values.

    Returns
    -------
    namespace : module
        A namespace that follows the Array API standard (as provided by
        ``array-api-compat`` for libraries that do not fully follow it).

    Raises
    ------
    TypeError
        If the arrays belong to different, incompatible namespaces.
    """
    arrays = [value for value in values if is_array_api_obj(value)]

    if not arrays:
        return _numpy_namespace

    try:
        return _array_namespace(*arrays)
    except TypeError:
        # E.g. quantities and plain arrays.
        return _quantity_namespace(arrays)


def _quantity_namespace(arrays):
    """The namespace for quantities, if all arrays can be used with it.

    That is, if the arrays are quantities and plain arrays, and the values
    of the quantities and the plain arrays all belong to the same namespace.
    """
    # Imported here, since astropy.units depends on this module.
    from astropy.units.quantity_helper.namespace import QuantityNamespace

    # Each array is looked up separately, since array_api_compat considers
    # namespaces with the same name to be the same.
    namespaces = [_array_namespace(array) for array in arrays]
    wrapped = [ns._xp if isinstance(ns, QuantityNamespace) else ns for ns in namespaces]

    if any(xp is not wrapped[0] for xp in wrapped):
        names = ", ".join(xp.__name__ for xp in dict.fromkeys(wrapped))

        raise TypeError(f"Multiple namespaces for array inputs: {names}") from None

    return next(ns for ns in namespaces if isinstance(ns, QuantityNamespace))


def default_floating_dtype(xp, device=None):
    """Return the default real floating-point dtype of namespace ``xp``.

    Parameters
    ----------
    xp : module
        A namespace that follows the Array API standard.
    device : device, optional
        The device for which to return the default; if not given, the
        current default device.

    Returns
    -------
    dtype
        The default real floating-point dtype, or ``xp.float64`` if the
        namespace does not provide its defaults.
    """
    try:
        info = xp.__array_namespace_info__()
    except AttributeError:
        return xp.float64

    return info.default_dtypes(device=device)["real floating"]
