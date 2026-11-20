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


def make(name):
    """A new array of the subclass ``name``, with metadata differing from defaults."""
    match name:
        case "Quantity":
            return u.Quantity([1.0, 2.0], u.m)
        case "Angle":
            return Angle([1.0, 2.0], u.deg)
        case "Latitude":
            return Latitude([1.0, 2.0], u.deg)
        case "Longitude":
            return Longitude([1.0, 2.0], u.deg, wrap_angle=180 * u.deg)
        case "Distance":
            return Distance([1.0, 2.0], u.pc)
        case "SpectralQuantity":
            return SpectralQuantity(
                [1.0, 2.0], u.GHz, doppler_rest=5 * u.GHz, doppler_convention="radio"
            )
        case "SpectralCoord":
            return SpectralCoord(
                [1.0, 2.0],
                u.GHz,
                doppler_rest=5 * u.GHz,
                doppler_convention="radio",
                radial_velocity=10 * u.km / u.s,
            )
        case "Magnitude":
            return u.Magnitude([1.0, 2.0])
        case "EarthLocation":
            return EarthLocation.from_geodetic(
                [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], ellipsoid="GRS80"
            )


SUBCLASSES = [
    "Quantity",
    "Angle",
    "Latitude",
    "Longitude",
    "Distance",
    "SpectralQuantity",
    "SpectralCoord",
    "Magnitude",
    "EarthLocation",
]

# An equivalent unit, for conversions.
OTHER_UNIT = {
    "Quantity": u.km,
    "Angle": u.rad,
    "Latitude": u.rad,
    "Longitude": u.rad,
    "Distance": u.kpc,
    "SpectralQuantity": u.MHz,
    "SpectralCoord": u.MHz,
    "Magnitude": u.mag(u.one),
    "EarthLocation": u.m,
}

METADATA = {
    "Longitude": ("wrap_angle",),
    "SpectralQuantity": ("doppler_rest", "doppler_convention"),
    "SpectralCoord": ("doppler_rest", "doppler_convention", "radial_velocity"),
    "EarthLocation": ("ellipsoid",),
}


def assert_metadata_kept(result, x):
    for attr in METADATA.get(type(x).__name__, ()):
        assert np.all(getattr(result, attr) == getattr(x, attr)), attr


def has_info(result):
    # Info is only stored once accessed or set.
    return "info" in getattr(result, "__dict__", {})


def outcome(f, *args):
    """The class name of the result of ``f(*args)``, or of the exception raised."""
    try:
        return type(f(*args)).__name__
    except Exception as exc:
        return type(exc).__name__


@pytest.fixture(params=SUBCLASSES)
def x(request):
    x = make(request.param)
    x.info.name = "x"

    return x


class TestSameQuantity:
    """Operations giving the same quantity in another form keep the class."""

    OPS = {
        "x[:1]": lambda x: x[:1],
        "x.copy()": lambda x: x.copy(),
        "copy.copy(x)": copy.copy,
        "copy.deepcopy(x)": copy.deepcopy,
        "x.reshape(2, 1)": lambda x: x.reshape(2, 1),
        "np.reshape(x, (2, 1))": lambda x: np.reshape(x, (2, 1)),
        "x.T": lambda x: x.T,
        "x.take([0])": lambda x: x.take([0]),
        "x.insert(0, x[0])": lambda x: x.insert(0, x[0]),
        "x.to(other_unit)": lambda x: x.to(OTHER_UNIT[type(x).__name__]),
    }

    # These keep the class and metadata, but not info.
    OPS_WITHOUT_INFO = {
        "x[0]": lambda x: x[0],
        "next(iter(x))": lambda x: next(iter(x)),
        "np.copy(x)": np.copy,
    }

    @pytest.mark.parametrize("op", OPS)
    def test_class_metadata_and_info(self, x, op):
        result = self.OPS[op](x)

        assert type(result) is type(x)
        assert_metadata_kept(result, x)
        assert result.info.name == "x"

    @pytest.mark.parametrize("op", OPS_WITHOUT_INFO)
    def test_class_and_metadata_without_info(self, x, op):
        result = self.OPS_WITHOUT_INFO[op](x)

        assert type(result) is type(x)
        assert_metadata_kept(result, x)
        assert not has_info(result)

    def test_lshift(self, x):
        if isinstance(x, EarthLocation):
            # EarthLocation does not support <<.
            with pytest.raises(TypeError):
                x << u.m
        else:
            result = x << OTHER_UNIT[type(x).__name__]

            assert type(result) is type(x)
            assert_metadata_kept(result, x)
            assert result.info.name == "x"

    def test_conversion_to_other_function_unit(self):
        x = make("Magnitude")
        x.info.name = "x"
        result = x.to(u.dex(1))

        assert type(result) is u.Dex
        assert result.info.name == "x"

    def test_astype(self, x):
        if isinstance(x, EarthLocation):
            with pytest.raises(TypeError):
                x.astype(np.float32)
        else:
            result = x.astype(np.float32)

            assert type(result) is type(x)
            assert_metadata_kept(result, x)
            assert result.info.name == "x"


OPS = {
    "-x": lambda x: -x,
    "x * 2": lambda x: x * 2,
    "2 * x": lambda x: 2 * x,
    "x / 2": lambda x: x / 2,
    "x * u.one": lambda x: x * u.one,
    "x * u.s": lambda x: x * u.s,
    "x**2": lambda x: x**2,
    "x + x": lambda x: x + x,
    "x - x": lambda x: x - x,
    "x == x": lambda x: x == x,
    "x < x": lambda x: x < x,
    "np.minimum(x, x)": lambda x: np.minimum(x, x),
    "np.add.reduce(x)": np.add.reduce,
    "np.min(x)": np.min,
    "x.min()": lambda x: x.min(),
    "np.sum(x)": np.sum,
    "np.mean(x)": np.mean,
    "np.concatenate([x, x])": lambda x: np.concatenate([x, x]),
    "np.stack([x, x])": lambda x: np.stack([x, x]),
    "np.where([True, False], x, x)": lambda x: np.where([True, False], x, x),
}

# The class of the result of each operation in OPS, or of the exception raised;
# columns are in the order of SUBCLASSES, except EarthLocation, given separately.
Q, A, LAT, LON, D = "Quantity", "Angle", "Latitude", "Longitude", "Distance"
SQ, SC, MAG, ND = "SpectralQuantity", "SpectralCoord", "Magnitude", "ndarray"
UTE = "UnitTypeError"
RESULT_CLASSES = {  # Quantity, Angle, Latitude, Longitude, Distance, SQ, SC, Magnitude
    "-x": (Q, A, A, A, D, Q, Q, UTE),
    "x * 2": (Q, A, A, A, D, SQ, SC, MAG),
    "2 * x": (Q, A, A, A, D, SQ, SC, MAG),
    "x / 2": (Q, A, A, A, D, SQ, SC, MAG),
    "x * u.one": (Q, A, LAT, LON, D, SQ, SC, MAG),
    "x * u.s": (Q, Q, Q, Q, Q, Q, Q, Q),
    "x**2": (Q, Q, Q, Q, Q, Q, Q, Q),
    "x + x": (Q, A, A, A, D, Q, Q, MAG),
    "x - x": (Q, A, A, A, D, Q, Q, MAG),
    "x == x": (ND, ND, ND, ND, ND, ND, ND, ND),
    "x < x": (ND, ND, ND, ND, ND, ND, ND, ND),
    "np.minimum(x, x)": (Q, A, A, A, D, Q, Q, UTE),
    "np.add.reduce(x)": (Q, A, A, A, D, Q, Q, UTE),
    "np.min(x)": (Q, A, A, A, D, SQ, SC, MAG),
    "x.min()": (Q, A, A, A, D, SQ, SC, MAG),
    "np.sum(x)": (Q, A, A, A, D, Q, Q, Q),
    "np.mean(x)": (Q, A, LAT, LON, D, SQ, SC, MAG),
    "np.concatenate([x, x])": (Q, A, LAT, LON, D, SQ, SC, MAG),
    "np.stack([x, x])": (Q, A, LAT, LON, D, SQ, SC, MAG),
    "np.where([True, False], x, x)": (Q, A, LAT, LON, D, SQ, SC, MAG),
}
EL, UFTE, TE = "EarthLocation", "UFuncTypeError", "TypeError"
EARTH_LOCATION_RESULT_CLASSES = {
    "-x": UFTE,
    "x * 2": UFTE,
    "2 * x": UFTE,
    "x / 2": TE,
    "x * u.one": EL,
    "x**2": TE,
    "x + x": UFTE,
    "x - x": UFTE,
    "x == x": ND,
    "x < x": UFTE,
    "np.minimum(x, x)": UFTE,
    "np.add.reduce(x)": UFTE,
    "np.min(x)": UFTE,
    "x.min()": UFTE,
    "np.sum(x)": UFTE,
    "np.mean(x)": UFTE,
    "np.concatenate([x, x])": EL,
    "np.stack([x, x])": EL,
    "np.where([True, False], x, x)": EL,
}
# Function quantities keep info through arithmetic that keeps a function unit.
MAGNITUDE_OPS_WITH_INFO = {"x * 2", "2 * x", "x / 2", "x * u.one", "x + x", "x - x"}


class TestComputations:
    """Computations give a class depending on the result unit and the operation."""

    @pytest.mark.parametrize("op", RESULT_CLASSES)
    @pytest.mark.parametrize("name", SUBCLASSES[:-1])
    def test_result_class(self, name, op):
        expected = RESULT_CLASSES[op][SUBCLASSES.index(name)]
        assert outcome(OPS[op], make(name)) == expected

    @pytest.mark.parametrize("op", EARTH_LOCATION_RESULT_CLASSES)
    def test_result_class_of_earth_location(self, op):
        expected = EARTH_LOCATION_RESULT_CLASSES[op]
        assert outcome(OPS[op], make("EarthLocation")) == expected

    @pytest.mark.parametrize("op", OPS)
    def test_metadata_and_info(self, x, op):
        try:
            result = OPS[op](x)
        except Exception:
            pytest.skip("raises, as checked in test_result_class")

        if type(result) is type(x):
            assert_metadata_kept(result, x)
        if isinstance(x, u.Magnitude) and op in MAGNITUDE_OPS_WITH_INFO:
            assert result.info.name == "x"
        elif not (isinstance(x, u.Magnitude) and op == "x**2"):
            assert not has_info(result)

    def test_magnitude_squared_keeps_info(self):
        x = make("Magnitude")
        x.info.name = "x"
        result = x**2

        assert type(result) is u.Quantity
        assert result.info.name == "x"

    def test_subclass_without_overrides(self):
        class MyQuantity(u.Quantity):
            pass

        x = MyQuantity([1.0, 2.0], u.m)

        for op in ("x * 2", "x * u.s", "x**2", "x + x", "np.sum(x)"):
            assert type(OPS[op](x)) is MyQuantity, op

    def test_constant(self):
        assert type(c.to(u.km / u.s)) is u.Quantity
        assert type(np.copy(c)) is u.Quantity

        for op in ("-x", "x * 2", "x * u.one", "x * u.s", "x**2", "x + x", "np.sum(x)"):
            assert type(OPS[op](c)) is u.Quantity, op


def imul_2(x):
    x *= 2

    return x


def imul_s(x):
    x *= u.s

    return x


def ilshift(x):
    x <<= OTHER_UNIT[type(x).__name__]

    return x


def iadd(x):
    x += x.copy()

    return x


AUGMENTED = {
    "x *= 2": imul_2,
    "x *= u.s": imul_s,
    "x <<= other_unit": ilshift,
    "x += x": iadd,
}
# Columns are in the order of SUBCLASSES, except EarthLocation, given separately.
AUGMENTED_RESULT_CLASSES = (
    {  # Quantity, Angle, Latitude, Longitude, Distance, SQ, SC, Magnitude
        "x *= 2": (Q, A, A, A, D, SQ, SC, MAG),
        "x *= u.s": (Q, UTE, UTE, UTE, UTE, UTE, UTE, UTE),
        "x <<= other_unit": (Q, A, LAT, LON, D, SQ, SC, MAG),
        "x += x": (Q, A, A, A, D, TE, TE, MAG),
    }
)


class TestAugmentedAssignment:
    """Augmented assignments give a class depending on the result unit, and keep info."""

    @pytest.mark.parametrize("op", AUGMENTED_RESULT_CLASSES)
    @pytest.mark.parametrize("name", SUBCLASSES[:-1])
    def test_result_class(self, name, op):
        expected = AUGMENTED_RESULT_CLASSES[op][SUBCLASSES.index(name)]

        assert outcome(AUGMENTED[op], make(name)) == expected

    @pytest.mark.parametrize("op", AUGMENTED)
    def test_metadata_and_info(self, x, op):
        if isinstance(x, EarthLocation):
            pytest.skip("covered by test_earth_location")

        try:
            result = AUGMENTED[op](x)
        except Exception:
            pytest.skip("raises, as checked in test_result_class")

        if type(result) is type(x):
            assert_metadata_kept(result, x)

        assert result.info.name == "x"

    def test_earth_location(self):
        x = make("EarthLocation")
        x.info.name = "x"

        for op in ("x *= 2", "x += x"):
            assert outcome(AUGMENTED[op], make("EarthLocation")) == UFTE


def make_pair(names):
    """New arrays of the subclasses named in ``names``, with matching units."""
    match names:
        case ("Distance", "SpectralCoord"):
            return Distance([1.0, 2.0], u.m), SpectralCoord([3.0, 4.0], u.m)
        case ("SpectralCoord", "Distance"):
            return SpectralCoord([3.0, 4.0], u.m), Distance([1.0, 2.0], u.m)
        case ("Quantity", "Angle"):
            return u.Quantity([1.0, 2.0], u.deg), Angle([3.0, 4.0], u.deg)
        case ("Dimensionless", "Angle"):
            return u.Quantity([1.0, 2.0]), Angle([3.0, 4.0], u.deg)
        case ("Angle", "Dimensionless"):
            return Angle([3.0, 4.0], u.deg), u.Quantity([1.0, 2.0])
        case _:
            return tuple(make(name) for name in names)


PAIR_OPS = {
    "a + b": lambda a, b: a + b,
    "a * b": lambda a, b: a * b,
    "np.concatenate([a, b])": lambda a, b: np.concatenate([a, b]),
    "np.stack([a, b])": lambda a, b: np.stack([a, b]),
    "np.where([True, False], a, b)": lambda a, b: np.where([True, False], a, b),
}
UCE = "UnitConversionError"
# a + b, a * b, concatenate, stack, where
PAIR_RESULT_CLASSES = {
    ("Latitude", "Longitude"): (A, Q, LAT, LAT, LAT),
    ("Longitude", "Latitude"): (A, Q, LON, LON, LON),
    ("Angle", "Longitude"): (A, Q, LON, LON, LON),
    ("Longitude", "Angle"): (A, Q, LON, LON, LON),
    ("Quantity", "Angle"): (A, Q, A, A, A),
    ("Distance", "Angle"): (UCE, Q, UCE, UCE, UCE),
    ("Angle", "Distance"): (UCE, Q, UCE, UCE, UCE),
    ("Dimensionless", "Angle"): (UCE, A, UCE, UCE, UCE),
    ("Angle", "Dimensionless"): (UCE, A, UCE, UCE, UCE),
    ("Distance", "SpectralCoord"): (D, Q, D, D, D),
    ("SpectralCoord", "Distance"): (Q, Q, SC, SC, SC),
    ("SpectralCoord", "SpectralQuantity"): (Q, Q, SC, SC, SC),
    ("SpectralQuantity", "SpectralCoord"): (Q, Q, SC, SC, SC),
}


class TestMixedClasses:
    """The most derived class decides, else the leftmost."""

    @pytest.mark.parametrize("op", PAIR_OPS)
    @pytest.mark.parametrize("names", PAIR_RESULT_CLASSES)
    def test_result_class(self, names, op):
        expected = PAIR_RESULT_CLASSES[names][list(PAIR_OPS).index(op)]

        assert outcome(PAIR_OPS[op], *make_pair(names)) == expected

    @pytest.mark.parametrize(
        "names", [("Latitude", "Longitude"), ("Angle", "Longitude")]
    )
    def test_longitude_metadata(self, names):
        a, b = make_pair(names)
        lon = a if isinstance(a, Longitude) else b

        for op in ("np.concatenate([a, b])", "np.where([True, False], a, b)"):
            result = PAIR_OPS[op](a, b)

            if isinstance(result, Longitude):
                assert result.wrap_angle == lon.wrap_angle


class TestOutput:
    """With ``out``, the output's class is checked and can decide the result."""

    def test_ufunc_result_is_not_output(self):
        lat, lon = make("Latitude"), make("Longitude")
        out = Longitude([0.0, 0.0], u.deg)
        result = np.add(lat, lon, out=out)

        assert type(result) is Angle
        assert result is not out

    def test_output_decides_implementation(self):
        result = np.add(1 * u.dex, 2 * u.dex, out=u.Dex(0.0))

        assert type(result) is u.Dex

    def test_output_unit_checked(self):
        with pytest.raises(u.UnitTypeError):
            np.sin(make("Angle"), out=Angle([0.0, 0.0], u.deg))

        with pytest.raises(u.UnitTypeError):
            np.multiply(make("Distance"), 2 * u.s, out=Distance([0.0, 0.0], u.m))

    def test_output_of_base_class(self):
        out = u.Quantity([0.0, 0.0], u.km)
        result = np.multiply(make("Distance"), 2, out=out)

        assert result is out
        assert result.unit == u.pc

    def test_comparison_output(self):
        with pytest.raises(TypeError):
            np.less(make("Distance"), make("Distance"), out=u.Quantity([0.0, 0.0]))

    def test_concatenate_output_unit_set(self):
        out = SpectralQuantity([0.0, 0.0], u.m)
        result = np.concatenate([[1.0] * u.Hz, [2.0] * u.Hz], out=out)

        assert result is out
        assert result.unit == u.Hz


class TestFields:
    """Fields of structured quantities are base quantities."""

    def test_earth_location(self):
        x = make("EarthLocation")
        x.info.name = "x"
        result = x["x"]

        assert type(result) is u.Quantity
        assert result.unit == u.m
        assert result.info.name == "x"

    def test_structured(self):
        x = u.Quantity([(1.0, 2.0)], u.Unit("(m, s)"))
        x.info.name = "x"
        result = x["f1"]

        assert type(result) is u.Quantity
        assert result.unit == u.s
        assert not has_info(result)
