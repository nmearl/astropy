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


def q(value, unit):
    return Quantity(jnp.asarray(value), unit)


SPRING_CONSTANT = q(3.0, u.J / u.m**2)


def energy(x):
    """The energy in J of springs stretched by ``x``, in m."""
    e = SPRING_CONSTANT * Quantity(x, u.m) ** 2 / 2

    return array_namespace(e).sum(e).to_value(u.J)


def raises_at_boundary():
    # Recent JAX versions raise before calling __jax_array__, older ones call it.
    return pytest.raises((TypeError, ValueError), match="__jax_array__|not a valid JAX")


def double(x):
    return 2 * x


class TestBoundary:
    """Quantities cannot be passed to or returned from transformations."""

    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(jax.jit(double), id="jit"),
            pytest.param(lambda x: jax.jit(double).lower(x), id="lower"),
            pytest.param(lambda x: jax.eval_shape(double, x), id="eval_shape"),
            pytest.param(jax.vmap(double), id="vmap"),
            pytest.param(jax.grad(lambda x: x.value[0]), id="grad"),
            pytest.param(
                jax.value_and_grad(lambda x: x.value[0]),
                id="value_and_grad",
            ),
            pytest.param(jax.jacfwd(double), id="jacfwd"),
            pytest.param(jax.jacrev(double), id="jacrev"),
            pytest.param(jax.hessian(double), id="hessian"),
            pytest.param(lambda x: jax.jvp(double, (x,), (x,)), id="jvp"),
            pytest.param(lambda x: jax.vjp(double, x), id="vjp"),
            pytest.param(jax.checkpoint(double), id="checkpoint"),
            pytest.param(
                lambda x: jax.lax.scan(lambda c, _: (c, None), x, None, length=2),
                id="scan",
            ),
            pytest.param(jax.device_put, id="device_put"),
            pytest.param(jnp.sin, id="jnp.sin"),
        ],
    )
    def test_quantity_inputs(self, transform):
        with raises_at_boundary():
            transform(q([1.0, 2.0], u.m))

    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(jax.jit, id="jit"),
            pytest.param(jax.vmap, id="vmap"),
            pytest.param(lambda f: lambda x: jax.eval_shape(f, x), id="eval_shape"),
        ],
    )
    def test_quantity_outputs(self, transform):
        with raises_at_boundary():
            transform(lambda x: Quantity(x, u.m))(jnp.asarray([1.0, 2.0]))

    def test_conversion(self):
        with pytest.raises(TypeError):
            jnp.asarray(q([1.0, 2.0], u.m))

    def test_operators_of_plain_arrays(self):
        # JAX arrays defer to quantities.
        result = jnp.asarray([2.0, 3.0]) * q([1.0, 2.0], u.m)

        assert result.unit is u.m
        assert_allclose(np.asarray(result.value), [2.0, 6.0])

    def test_tree_leaf(self):
        a = q([1.0, 2.0], u.m)

        assert jax.tree_util.tree_leaves({"a": a})[0] is a


class TestQuantitiesInside:
    """Functions of plain arrays can use quantities inside transformations."""

    def test_units_are_handled_while_tracing(self):
        calls = []

        @jax.jit
        def kinetic_energy(m, v):
            # m in kg and v in km/s.
            calls.append(1)  # Only while tracing.
            m, v = Quantity(m, u.kg), Quantity(v, u.km / u.s)

            return (0.5 * m * v**2).to_value(u.J)

        result = kinetic_energy(jnp.asarray([1.0, 2.0]), jnp.asarray([3.0, 4.0]))

        assert_allclose(np.asarray(result), [4.5e6, 16e6])

        kinetic_energy(jnp.asarray([1.0, 2.0]), jnp.asarray([3.0, 4.0]))

        assert len(calls) == 1

    def test_unit_errors_while_tracing(self):
        @jax.jit
        def add(a, b):
            return (Quantity(a, u.m) + Quantity(b, u.s)).value

        with pytest.raises(u.UnitConversionError):
            add(jnp.asarray([1.0]), jnp.asarray([1.0]))

    def test_generic_code(self):
        @jax.jit
        def centred(x):
            x = Quantity(x, u.km)
            xp = array_namespace(x)
            centred = xp.where(x > xp.mean(x), x - xp.mean(x), xp.zeros_like(x))

            return centred.to_value(u.m)

        result = centred(jnp.asarray([1.0, 2.0, 3.0]))

        assert_allclose(np.asarray(result), [0.0, 0.0, 1000.0])

    def test_namespace_results(self):
        ns = q(1.0, u.m).__array_namespace__()
        result = jax.jit(lambda x: ns.asarray(x, unit=u.s).to_value(u.ms))(jnp.ones(2))

        assert_allclose(np.asarray(result), [1000.0, 1000.0])

    def test_mixed_lookup(self):
        def weighted(x, mask, weights):
            x = Quantity(x, u.m)
            xp = array_namespace(mask, x, weights)

            return xp.sum(xp.where(mask, x, 0) * weights).to_value(u.cm)

        args = (
            jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            jnp.asarray([[True, False], [False, True]]),
            jnp.asarray([[2.0, 3.0], [4.0, 5.0]]),
        )

        assert_allclose(np.asarray(jax.jit(weighted)(*args)), 2200.0)
        assert_allclose(np.asarray(jax.vmap(weighted)(*args)), [200.0, 2000.0])

    def test_vmap(self):
        def f(x):
            x = Quantity(x, u.m)

            return array_namespace(x).sum(x * x).to_value(u.cm**2)

        result = jax.vmap(f)(jnp.ones((4, 3)))

        assert_allclose(np.asarray(result), np.full(4, 3e4))

    def test_derivatives(self):
        # In J/m, for x in m.
        x = jnp.asarray([1.0, 2.0])

        assert_allclose(np.asarray(jax.grad(energy)(x)), [3.0, 6.0])
        assert_allclose(np.asarray(jax.value_and_grad(energy)(x)[1]), [3.0, 6.0])
        assert_allclose(np.asarray(jax.jacfwd(energy)(x)), [3.0, 6.0])
        assert_allclose(np.asarray(jax.jacrev(energy)(x)), [3.0, 6.0])
        assert_allclose(np.asarray(jax.grad(jax.jit(energy))(x)), [3.0, 6.0])
        # In J/m2.
        assert_allclose(np.asarray(jax.hessian(energy)(x)), np.diag([3.0, 3.0]))

    def test_closed_over_quantities_in_other_units(self):
        # The constant in J/cm2 is converted while tracing.
        k = q(3e-4, u.J / u.cm**2)

        def f(x):
            e = k * Quantity(x, u.m) ** 2 / 2

            return array_namespace(e).sum(e).to_value(u.J)

        assert_allclose(np.asarray(jax.grad(f)(jnp.asarray([1.0, 2.0]))), [3.0, 6.0])


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
