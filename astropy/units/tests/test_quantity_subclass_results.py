# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test the class, metadata, and info of results of operations on quantities."""

import copy

import numpy as np
import pytest

from astropy import units as u
from astropy.constants import c
from astropy.coordinates import (
    Angle,
    Distance,
    EarthLocation,
    Latitude,
    Longitude,
    SpectralCoord,
    SpectralQuantity,
)

# Tests of numpy-specific behavior, for the compatibility layer.
pytestmark = pytest.mark.numpy_compat

# Metadata that results of the same class should keep.
METADATA = {
    Longitude: ["wrap_angle"],
    SpectralQuantity: ["doppler_rest", "doppler_convention"],
    SpectralCoord: ["doppler_rest", "doppler_convention", "radial_velocity"],
}


def assert_metadata_kept(result, x):
    """Check that result has the metadata of x listed in METADATA."""
    for attr in METADATA.get(type(x), []):
        assert np.all(getattr(result, attr) == getattr(x, attr)), attr


def longitude():
    return Longitude([1.0, 2.0], u.deg, wrap_angle=180 * u.deg)


def spectral_quantity():
    return SpectralQuantity(
        [1.0, 2.0], u.GHz, doppler_rest=5 * u.GHz, doppler_convention="radio"
    )


def spectral_coord():
    return SpectralCoord(
        [1.0, 2.0],
        u.GHz,
        doppler_rest=5 * u.GHz,
        doppler_convention="radio",
        radial_velocity=10 * u.km / u.s,
    )


# Each subclass, with an equivalent unit to convert to.
SUBCLASSES = [
    pytest.param(u.Quantity([1.0, 2.0], u.m), u.km, id="Quantity"),
    pytest.param(Angle([1.0, 2.0], u.deg), u.rad, id="Angle"),
    pytest.param(Latitude([1.0, 2.0], u.deg), u.rad, id="Latitude"),
    pytest.param(longitude(), u.rad, id="Longitude"),
    pytest.param(Distance([1.0, 2.0], u.pc), u.kpc, id="Distance"),
    pytest.param(spectral_quantity(), u.MHz, id="SpectralQuantity"),
    pytest.param(spectral_coord(), u.MHz, id="SpectralCoord"),
    pytest.param(u.Magnitude([1.0, 2.0]), u.mag(u.one), id="Magnitude"),
]


@pytest.mark.parametrize("original, other_unit", SUBCLASSES)
class TestSameQuantity:
    """Operations giving the same quantity in another form keep the class."""

    def test_class_metadata_and_info(self, original, other_unit):
        x = copy.deepcopy(original)
        x.info.name = "x"
        results = [
            x[:1],
            x.copy(),
            copy.copy(x),
            copy.deepcopy(x),
            x.reshape(2, 1),
            np.reshape(x, (2, 1)),
            x.T,
            x.take([0]),
            x.insert(0, x[0]),
            x.astype(np.float32),
            x.to(other_unit),
            x << other_unit,
        ]
        for result in results:
            assert type(result) is type(x)
            assert_metadata_kept(result, x)
            assert result.info.name == "x"

    def test_scalars_and_np_copy_drop_info(self, original, other_unit):
        x = copy.deepcopy(original)
        x.info.name = "x"
        for result in [x[0], next(iter(x)), np.copy(x)]:
            assert type(result) is type(x)
            assert_metadata_kept(result, x)
            assert "info" not in result.__dict__

    def test_augmented_assignments_keep_info(self, original, other_unit):
        x = copy.deepcopy(original)
        x.info.name = "x"
        x *= 2
        assert x.info.name == "x"
        x <<= other_unit
        assert x.info.name == "x"


class TestQuantity:
    def setup_method(self):
        self.q = u.Quantity([1.0, 2.0], u.m)
        self.q.info.name = "q"

    def test_computations_give_quantity_without_info(self):
        q = self.q
        results = [
            -q,
            q * 2,
            2 * q,
            q / 2,
            q * u.one,
            q * u.s,
            q**2,
            q + q,
            q - q,
            np.minimum(q, q),
            np.add.reduce(q),
            np.min(q),
            q.min(),
            np.sum(q),
            np.mean(q),
            np.concatenate([q, q]),
            np.stack([q, q]),
            np.where([True, False], q, q),
        ]
        for result in results:
            assert type(result) is u.Quantity
            assert "info" not in result.__dict__

    def test_comparisons_give_plain_arrays(self):
        assert type(self.q == self.q) is np.ndarray
        assert type(self.q < self.q) is np.ndarray

    def test_augmented_assignment_changing_unit(self):
        q = self.q.copy()
        q *= u.s
        assert q.unit == u.m * u.s
        assert q.info.name == "q"

    def test_subclass_without_overrides(self):
        class MyQuantity(u.Quantity):
            pass

        x = MyQuantity([1.0, 2.0], u.m)
        for result in [x * 2, x * u.s, x**2, x + x, np.sum(x)]:
            assert type(result) is MyQuantity


@pytest.mark.parametrize("x", [Angle([1.0, 2.0], u.deg), Distance([1.0, 2.0], u.pc)])
class TestAngleAndDistance:
    """Angle and Distance keep their class while the unit allows it."""

    def test_computations_with_same_physical_type(self, x):
        results = [
            -x,
            x * 2,
            2 * x,
            x / 2,
            x * u.one,
            x + x,
            x - x,
            np.minimum(x, x),
            np.add.reduce(x),
            np.min(x),
            x.min(),
            np.sum(x),
            np.mean(x),
            np.concatenate([x, x]),
            np.stack([x, x]),
            np.where([True, False], x, x),
        ]
        for result in results:
            assert type(result) is type(x)

    def test_computations_with_other_physical_type(self, x):
        assert type(x * u.s) is u.Quantity
        assert type(x**2) is u.Quantity

    def test_augmented_assignment_with_other_physical_type(self, x):
        y = x.copy()
        with pytest.raises(u.UnitTypeError):
            y *= u.s


@pytest.mark.parametrize("cls", [Latitude, Longitude])
class TestLatitudeAndLongitude:
    """Latitude and Longitude become Angle in most computations."""

    def test_computations_giving_angle(self, cls):
        x = cls([1.0, 2.0], u.deg)
        results = [
            -x,
            x * 2,
            2 * x,
            x / 2,
            x + x,
            x - x,
            np.minimum(x, x),
            np.add.reduce(x),
            np.min(x),
            x.min(),
            np.sum(x),
        ]
        for result in results:
            assert type(result) is Angle

    def test_computations_keeping_class(self, cls):
        x = cls([1.0, 2.0], u.deg)
        results = [
            x * u.one,
            np.mean(x),
            np.concatenate([x, x]),
            np.stack([x, x]),
            np.where([True, False], x, x),
        ]
        for result in results:
            assert type(result) is cls

    def test_augmented_assignment(self, cls):
        x = cls([1.0, 2.0], u.deg)
        x *= 2
        assert type(x) is Angle
        x = cls([1.0, 2.0], u.deg)
        x += cls([1.0, 2.0], u.deg)
        assert type(x) is Angle
        x = cls([1.0, 2.0], u.deg)
        with pytest.raises(u.UnitTypeError):
            x *= u.s


def test_longitude_keeps_wrap_angle():
    x = longitude()
    for result in [x * u.one, np.mean(x), np.concatenate([x, x])]:
        assert result.wrap_angle == 180 * u.deg


@pytest.mark.parametrize("x", [spectral_quantity(), spectral_coord()])
class TestSpectral:
    """SpectralQuantity and SpectralCoord keep their class for scaling and selection."""

    def test_computations_keeping_class(self, x):
        results = [
            x * 2,
            2 * x,
            x / 2,
            x * u.one,
            np.min(x),
            x.min(),
            np.mean(x),
            np.concatenate([x, x]),
            np.stack([x, x]),
            np.where([True, False], x, x),
        ]
        for result in results:
            assert type(result) is type(x)
            assert_metadata_kept(result, x)

    def test_computations_giving_quantity(self, x):
        results = [
            -x,
            x + x,
            x - x,
            np.minimum(x, x),
            np.add.reduce(x),
            np.sum(x),
            x * u.s,
            x**2,
        ]
        for result in results:
            assert type(result) is u.Quantity

    def test_augmented_assignment(self, x):
        y = x.copy()
        with pytest.raises(u.UnitTypeError):
            y *= u.s
        y = x.copy()
        with pytest.raises(TypeError):
            y += x


class TestMagnitude:
    def setup_method(self):
        self.m = u.Magnitude([1.0, 2.0])
        self.m.info.name = "m"

    def test_arithmetic_keeps_class_and_info(self):
        m = self.m
        for result in [m * 2, 2 * m, m / 2, m * u.one, m + m, m - m]:
            assert type(result) is u.Magnitude
            assert result.info.name == "m"

    def test_selection_and_joining_keep_class(self):
        m = self.m
        results = [
            np.min(m),
            m.min(),
            np.mean(m),
            np.concatenate([m, m]),
            np.stack([m, m]),
            np.where([True, False], m, m),
        ]
        for result in results:
            assert type(result) is u.Magnitude
            assert "info" not in result.__dict__

    def test_computations_giving_quantity(self):
        assert type(self.m * u.s) is u.Quantity
        assert type(np.sum(self.m)) is u.Quantity
        squared = self.m**2
        assert type(squared) is u.Quantity
        assert squared.info.name == "m"

    def test_unsupported_computations(self):
        with pytest.raises(u.UnitTypeError):
            -self.m
        with pytest.raises(u.UnitTypeError):
            np.add.reduce(self.m)
        with pytest.raises(u.UnitTypeError):
            np.minimum(self.m, self.m)

    def test_conversion_to_other_function_unit(self):
        result = self.m.to(u.dex(1))
        assert type(result) is u.Dex
        assert result.info.name == "m"


def test_constant_results_are_quantities():
    results = [
        c.to(u.km / u.s),
        np.copy(c),
        -c,
        c * 2,
        c * u.one,
        c * u.s,
        c**2,
        c + c,
        np.sum(c),
    ]
    for result in results:
        assert type(result) is u.Quantity


class TestEarthLocation:
    def setup_method(self):
        self.el = EarthLocation.from_geodetic(
            [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], ellipsoid="GRS80"
        )
        self.el.info.name = "el"

    def test_same_quantity(self):
        el = self.el
        for result in [el[:1], el.copy(), el.reshape(2, 1), el.take([0]), el.to(u.m)]:
            assert type(result) is EarthLocation
            assert result.ellipsoid == "GRS80"
            assert result.info.name == "el"

    def test_computations_keeping_class(self):
        el = self.el
        results = [
            el * u.one,
            np.concatenate([el, el]),
            np.stack([el, el]),
            np.where([True, False], el, el),
        ]
        for result in results:
            assert type(result) is EarthLocation
            assert result.ellipsoid == "GRS80"
            assert "info" not in result.__dict__

    def test_unsupported_computations(self):
        with pytest.raises(TypeError):
            -self.el
        with pytest.raises(TypeError):
            self.el + self.el
        with pytest.raises(TypeError):
            np.sum(self.el)

    def test_field(self):
        result = self.el["x"]
        assert type(result) is u.Quantity
        assert result.unit == u.m
        assert result.info.name == "el"

    def test_lshift_unsupported(self):
        with pytest.raises(TypeError):
            self.el << u.m


class TestMixedClasses:
    """The most derived class decides, else the leftmost."""

    def test_latitude_and_longitude(self):
        lat, lon = Latitude([1.0, 2.0], u.deg), longitude()
        assert type(lat + lon) is Angle
        assert type(lon + lat) is Angle
        assert type(lat * lon) is u.Quantity
        assert type(np.concatenate([lat, lon])) is Latitude
        assert type(np.where([True, False], lat, lon)) is Latitude
        joined = np.concatenate([lon, lat])
        assert type(joined) is Longitude
        assert joined.wrap_angle == 180 * u.deg

    def test_angle_and_longitude(self):
        angle, lon = Angle([1.0, 2.0], u.deg), longitude()
        assert type(angle + lon) is Angle
        for result in [np.concatenate([angle, lon]), np.stack([lon, angle])]:
            assert type(result) is Longitude
            assert result.wrap_angle == 180 * u.deg

    def test_quantity_and_angle(self):
        q, angle = u.Quantity([1.0, 2.0], u.deg), Angle([1.0, 2.0], u.deg)
        assert type(q + angle) is Angle
        assert type(q * angle) is u.Quantity
        assert type(np.concatenate([q, angle])) is Angle
        assert type(u.Quantity([1.0, 2.0]) * angle) is Angle

    def test_distance_and_angle(self):
        distance, angle = Distance([1.0, 2.0], u.m), Angle([1.0, 2.0], u.deg)
        assert type(distance * angle) is u.Quantity
        with pytest.raises(u.UnitConversionError):
            distance + angle

    def test_distance_and_spectral_coord(self):
        distance, sc = Distance([1.0, 2.0], u.m), SpectralCoord([3.0, 4.0], u.m)
        assert type(distance + sc) is Distance
        assert type(sc + distance) is u.Quantity
        assert type(np.concatenate([distance, sc])) is Distance
        assert type(np.concatenate([sc, distance])) is SpectralCoord

    def test_spectral_coord_and_spectral_quantity(self):
        sc, sq = SpectralCoord([1.0, 2.0], u.m), SpectralQuantity([3.0, 4.0], u.m)
        assert type(sc + sq) is u.Quantity
        assert type(sq + sc) is u.Quantity
        assert type(np.concatenate([sc, sq])) is SpectralCoord
        assert type(np.concatenate([sq, sc])) is SpectralCoord


class TestOutput:
    """With ``out``, the output's class is checked and can decide the result."""

    def test_ufunc_result_is_not_output(self):
        out = Longitude([0.0, 0.0], u.deg)
        result = np.add(Latitude([1.0, 2.0], u.deg), longitude(), out=out)
        assert type(result) is Angle
        assert result is not out

    def test_output_decides_implementation(self):
        result = np.add(1 * u.dex, 2 * u.dex, out=u.Dex(0.0))
        assert type(result) is u.Dex

    def test_output_unit_checked(self):
        with pytest.raises(u.UnitTypeError):
            np.sin(Angle([1.0, 2.0], u.deg), out=Angle([0.0, 0.0], u.deg))
        with pytest.raises(u.UnitTypeError):
            np.multiply(
                Distance([1.0, 2.0], u.m), 2 * u.s, out=Distance([0.0, 0.0], u.m)
            )

    def test_output_of_base_class(self):
        out = u.Quantity([0.0, 0.0], u.km)
        result = np.multiply(Distance([1.0, 2.0], u.pc), 2, out=out)
        assert result is out
        assert result.unit == u.pc

    def test_comparison_output(self):
        distance = Distance([1.0, 2.0], u.m)
        with pytest.raises(TypeError):
            np.less(distance, distance, out=u.Quantity([0.0, 0.0]))

    def test_concatenate_output_unit_set(self):
        out = SpectralQuantity([0.0, 0.0], u.m)
        result = np.concatenate([[1.0] * u.Hz, [2.0] * u.Hz], out=out)
        assert result is out
        assert result.unit == u.Hz


def test_structured_field():
    x = u.Quantity([(1.0, 2.0)], u.Unit("(m, s)"))
    x.info.name = "x"
    result = x["f1"]
    assert type(result) is u.Quantity
    assert result.unit == u.s
    assert "info" not in result.__dict__
