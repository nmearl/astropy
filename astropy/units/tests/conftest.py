# Licensed under a 3-clause BSD style license - see LICENSE.rst
import importlib.util

import array_api_compat
import pytest

from astropy.utils.compat.optional_deps import HAS_ARRAY_API_STRICT

BACKENDS = [
    "numpy",
    pytest.param(
        "array_api_strict",
        marks=pytest.mark.skipif(
            not HAS_ARRAY_API_STRICT, reason="array-api-strict not installed"
        ),
    ),
    pytest.param(
        "jax",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("jax") is None, reason="jax not installed"
        ),
    ),
]


@pytest.fixture(params=BACKENDS)
def xp(request):
    """The namespace of each array back-end that is installed."""
    if request.param == "numpy":
        yield array_api_compat.numpy
    elif request.param == "array_api_strict":
        import array_api_strict

        yield array_api_strict
    else:
        import jax

        # Double precision, as for the other back-ends. Tests of the default
        # single precision are in test_quantity_array_jax.py.
        enabled = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)

        try:
            yield jax.numpy
        finally:
            jax.config.update("jax_enable_x64", enabled)
