# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests of the Array API quantity with JAX transformations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from astropy import units as u
from astropy.units._quantity_array import Quantity
from astropy.utils.compat.array_api import array_namespace

jax = pytest.importorskip("jax")
jnp = jax.numpy


@pytest.fixture(autouse=True)
def default_precision():
    # JAX uses single precision by default, unlike the other back-ends.
    enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)

    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", enabled)


class MyQuantity(Quantity):
    __slots__ = ()


def q(value, unit):
    return Quantity(jnp.asarray(value), unit)


class TestPytree:
    def test_flatten(self):
        a = q([1.0, 2.0], u.km)
        leaves, treedef = jax.tree_util.tree_flatten(a)

        assert len(leaves) == 1
        assert leaves[0] is a.value

        result = jax.tree_util.tree_unflatten(treedef, leaves)

        assert type(result) is Quantity
        assert result.unit is u.km

    def test_subclass(self):
        a = MyQuantity(jnp.asarray([1.0, 2.0]), u.km)
        result = jax.jit(lambda x: 2 * x)(a)

        assert type(result) is MyQuantity
        assert result.unit is u.km

    def test_namespace_results(self):
        ns = q(1.0, u.m).__array_namespace__()
        result = jax.jit(lambda x: ns.asarray(x, unit=u.s))(jnp.ones(2))

        assert type(result) is Quantity
        assert result.unit is u.s


class TestJit:
    def test_units_are_handled_while_tracing(self):
        calls = []

        @jax.jit
        def kinetic_energy(m, v):
            calls.append(1)  # Only while tracing.
            return (0.5 * m * v**2).to(u.J)

        m = q([1.0, 2.0], u.kg)
        v = q([3.0, 4.0], u.km / u.s)
        result = kinetic_energy(m, v)

        assert result.unit == u.J
        assert_allclose(np.asarray(result.value), [4.5e6, 16e6])

        kinetic_energy(m, v)

        assert len(calls) == 1

        # A different unit is different static data, so it is traced again.
        result = kinetic_energy(m, q([3.0, 4.0], u.m / u.s))

        assert len(calls) == 2
        assert_allclose(np.asarray(result.value), [4.5, 16.0])

    def test_unit_errors_while_tracing(self):
        add = jax.jit(lambda a, b: a + b)

        with pytest.raises(u.UnitConversionError):
            add(q([1.0], u.m), q([1.0], u.s))

    def test_enabled_equivalencies(self):
        calls = []

        @jax.jit
        def add(a, b):
            calls.append(1)
            return a + b

        celsius = q(0.0, u.deg_C)
        kelvin = q(10.0, u.K)

        with u.set_enabled_equivalencies(u.temperature()):
            result = add(celsius, kelvin)

        assert result.unit is u.deg_C
        assert_allclose(np.asarray(result.value), -263.15, rtol=1e-6)

        # Outside the context, the function is traced again rather than
        # reusing the conversion, which is no longer possible.
        with pytest.raises(u.UnitConversionError):
            add(celsius, kelvin)

        assert len(calls) == 2

    def test_generic_code(self):
        @jax.jit
        def centred(x):
            xp = array_namespace(x)
            grid = xp.linspace(xp.min(x), xp.max(x), 3)

            return xp.where(x > xp.mean(x), x - xp.mean(x), xp.zeros_like(x)), grid

        result, grid = centred(q([1.0, 2.0, 3.0], u.km))

        assert result.unit is u.km
        assert_allclose(np.asarray(result.value), [0.0, 0.0, 1.0])
        assert grid.unit is u.km
        assert_allclose(np.asarray(grid.value), [1.0, 2.0, 3.0])


class TestOtherTransformations:
    def test_vmap(self):
        def f(x):
            return x.__array_namespace__().sum(x * x)

        result = jax.vmap(f)(Quantity(jnp.ones((4, 3)), u.m))

        assert result.unit == u.m**2
        assert result.shape == (4,)

    @staticmethod
    def _sum_of_squares(x):
        return x.__array_namespace__().sum(x * x).to_value(u.cm**2)

    def test_grad_values(self):
        result = jax.grad(self._sum_of_squares)(q([1.0, 2.0], u.m))

        # In cm2 per m.
        assert_allclose(np.asarray(result.value), [2e4, 4e4])

    @pytest.mark.xfail(
        strict=True, reason="gradients get the unit of the argument (not yet fixed)"
    )
    def test_grad_unit(self):
        result = jax.grad(self._sum_of_squares)(q([1.0, 2.0], u.m))

        assert result.unit == u.cm**2 / u.m

    def test_grad_of_values(self):
        # The way to get gradients with the correct unit for now.
        def f(value):
            return self._sum_of_squares(Quantity(value, u.m))

        result = Quantity(jax.grad(f)(jnp.asarray([1.0, 2.0])), u.cm**2 / u.m)

        assert_allclose(np.asarray(result.to_value(u.cm)), [200.0, 400.0])


class TestDefaultPrecision:
    def test_integers_become_single_precision(self):
        a = Quantity(jnp.asarray([1, 2]), u.km)

        assert a.dtype == jnp.float32

    def test_single_precision_is_kept(self):
        a = q([1.0, 2.0], u.km)

        assert a.dtype == jnp.float32
        assert a.to(u.m).dtype == jnp.float32
        assert (a + q([1.0, 2.0], u.m)).dtype == jnp.float32
        assert (0.5 * a).dtype == jnp.float32
        assert_allclose(np.asarray(a.to_value(u.m)), [1000.0, 2000.0])

    def test_integer_values_are_scaled_as_single_precision(self):
        a = Quantity(jnp.asarray([1, 2]), u.km, dtype=None)
        result = a.to(u.m)

        assert result.dtype == jnp.float32
        assert_allclose(np.asarray(result.value), [1000.0, 2000.0])
