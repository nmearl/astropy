# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from array_api_compat import numpy as numpy_namespace

from astropy import units as u
from astropy.utils.compat.array_api import array_namespace, default_floating_dtype
from astropy.utils.compat.optional_deps import HAS_ARRAY_API_STRICT


@pytest.mark.parametrize(
    "values",
    [
        pytest.param((np.ones(2),), id="array"),
        pytest.param((np.float64(1.0),), id="numpy-scalar"),
        pytest.param((np.ones(2) * u.m,), id="quantity"),
        pytest.param((1.0,), id="python-scalar"),
        pytest.param(([1.0, 2.0],), id="list"),
        pytest.param((None,), id="none"),
        pytest.param((), id="no-values"),
        pytest.param((1.0, np.ones(2), None), id="mixed"),
    ],
)
def test_numpy(values):
    assert array_namespace(*values) is numpy_namespace


@pytest.mark.skipif(not HAS_ARRAY_API_STRICT, reason="tests array_api_strict")
def test_array_api_strict():
    import array_api_strict as xp

    x = xp.asarray([1.0, 2.0])
    assert array_namespace(x) is xp
    assert array_namespace(x, 2.0, None) is xp

    with pytest.raises(TypeError):
        array_namespace(x, np.ones(2))


def test_default_floating_dtype():
    assert default_floating_dtype(numpy_namespace) == np.float64


@pytest.mark.skipif(not HAS_ARRAY_API_STRICT, reason="tests array_api_strict")
def test_default_floating_dtype_array_api_strict():
    import array_api_strict as xp

    assert default_floating_dtype(xp) == xp.float64
    assert default_floating_dtype(xp, device=xp.asarray(1.0).device) == xp.float64


def test_default_floating_dtype_without_info():
    class Namespace:
        float64 = np.float64

    assert default_floating_dtype(Namespace()) is np.float64
