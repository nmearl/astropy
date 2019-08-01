import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame
import logging
from astropy.constants import c

DOPPLER_CONVENTIONS = {
    'radio': u.doppler_radio,
    'optical': u.doppler_optical,
    'relativistic': u.doppler_relativistic
}

__all__ = ['SpectralCoord']


class SpectralCoord(u.Quantity):
    """
    Coordinate object representing a spectral axis.

    Parameters
    ----------
    data : ndarray or `Quantity`
        Spectral axis data values.
    unit : str or `Unit`
        Unit for the given data.
    rest : `Quantity`
        The rest value to use for velocity space transformations.
    observer : `BaseCoordinateFrame`
        The coordinate frame of the observer.
    target : `BaseCoordinateFrame`
        The coordinate frame of the target.
    """
    def __init__(self, data, unit=None, rest_value=None,
                 velocity_convention=None, observer=None, target=None):
        super().__init__(data, unit=unit)

        self.rest_value = rest_value or u.Quantity(0, unit)
        self.velocity_convention = DOPPLER_CONVENTIONS.get(
            velocity_convention)

        if not issubclass(observer.__class__, BaseCoordinateFrame) or \
            not issubclass(target.__class__, BaseCoordinateFrame):

            raise ValueError("`{}` is not a subclass of `BaseCoordinateFrame`"
                "".format(observer))

        self._observer = observer
        self._target = target

    def __quantity_subclass__(self, unit):
        """
        Overridden by subclasses to change what kind of view is
        created based on the output unit of an operation.
        """
        return SpectralCoord, True

    @property
    def observer(self):
        return self._observer

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        if not issubclass(value.__class__, BaseCoordinateFrame):
            raise ValueError("`{}` is not a subclass of `BaseCoordinateFrame`"
                "".format(value))

        self._target = value

    @property
    def rest_value(self):
        return self._rest_value

    @property
    @u.quantity_input(value=['length', 'frequency', 'energy', 'speed'])
    def rest_value(self, value):
        self._rest_value = value

    @property
    def velocity_convention(self):
        return self._velocity_convention

    @velocity_convention.setter
    def velocity_convention(self, value):
        if value not in DOPPLER_CONVENTIONS:
            logging.warning("Unrecognized velocity convention: {}".format(
                value))

        self._velocity_convention = value

    def transform_frame(self, frame=None, target=None):
        """
        Transform the frame of the observer.
        """
        if target is not None:
            self.target = target

        # Get the velocity differentials for each frame
        init_obs_vel = self.target.transform_to(self.observer).velocity
        fin_obs_vel = self.target.transform_to(self.observer).velocity

        # Calculate the velocity shift between the two vectors
        obs_vel_shift = init_obs_vel - fin_obs_vel

        # Project the velocity shift vector onto the the line-on-sight vector
        # between the target and the new observation frame.
        delta_vel = np.dot(obs_vel_shift, fin_obs_vel) / np.linalg.norm(
            fin_obs_vel)

        self._observer = frame

        # Apply the velocity shift to the stored spectral data
        new_data = (self.to('hz') * (1 + delta_vel / c.cgs)).to(self.unit)
        new_coord = SpectralCoord(new_data.value,
                                  unit=new_data.unit,
                                  rest_value=self.rest_value,
                                  velocity_convention=self.velocity_convention,
                                  observer=self.observer,
                                  target=self.target)

        return new_coord

    def to(self, *args, rest=None, convention=None, **kwargs):
        """
        Overload the parent ``to`` method to provide parameters for defining
        rest value and pre-defined conventions for unit transformations.
        """
        if rest is not None:
            self.rest = rest

        if convention is not None:
            if isinstance(convention, str):
                convention = DOPPLER_CONVENTIONS.get(convention)

            self.velocity_convention = convention

        kwargs.get('equivalencies', []).append(u.spectral() +
            self.velocity_convention(self.rest_value))

        return super().to(*args, **kwargs)

    def __repr__(self):
        pass
