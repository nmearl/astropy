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

    return _array_namespace(*arrays)


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
