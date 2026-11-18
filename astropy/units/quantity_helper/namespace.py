# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Array API namespaces for quantities."""

from __future__ import annotations

import array_api_compat
import numpy as np

from astropy.utils.compat.array_api import default_floating_dtype


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
