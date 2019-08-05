import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.coordinates import CartesianDifferential, SkyCoord
from astropy.coordinates.baseframe import BaseCoordinateFrame
from astropy.utils.compat import NUMPY_LT_1_14

DOPPLER_CONVENTIONS = {
    'radio': u.doppler_radio,
    'optical': u.doppler_optical,
    'relativistic': u.doppler_relativistic
}

__all__ = ['SpectralCoord']


class SpectralCoord(u.Quantity):
    """
    Coordinate object representing spectral values.

    Attributes
    ----------
    value : ndarray or `Quantity` or `SpectralCoord`
        Spectral axis data values.
    unit : str or `Unit`
        Unit for the given data.
    rest : `Quantity`, optional
        The rest value to use for velocity space transformations.
    velocity_convention : str
        The convention to use when converting the spectral data to/from
        velocity space.
    observer : `BaseCoordinateFrame` or `SkyCoord`, optional
        The coordinate (position and velocity) of observer.
    target : `BaseCoordinateFrame` or `SkyCoord`, optional
        The coordinate (position and velocity) of observer.
    """
    def __new__(cls, value, unit=None, rest=None,
                velocity_convention=None, observer=None, target=None,
                **kwargs):
        obj = super().__new__(cls, value, unit=unit, **kwargs)

        obj.rest = rest
        obj.velocity_convention = velocity_convention or 'relativistic'

        obj.observer = observer
        obj.target = target

        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)

        # If we're a new object or viewing an ndarray, nothing has to be done.
        if obj is None or obj.__class__ is np.ndarray:
            return

        self.rest = getattr(obj, '_rest_value', None)
        self.velocity_convention = getattr(obj, '_velocity_convention',
                                           'relativistic')
        self.observer = getattr(obj, '_observer', None)
        self.target = getattr(obj, '_target', None)

    def __quantity_subclass__(self, unit):
        """
        Overridden by subclasses to change what kind of view is
        created based on the output unit of an operation.
        """
        return SpectralCoord, True

    @staticmethod
    def _validate_frame(value):
        """
        Checks the type of the frame and whether a velocity differential has
        been defined on the frame object.

        Parameters
        ----------
        value : `BaseCoordinateFrame`
            The new frame to be used for target or observer.
        """
        if value is not None:
            if not isinstance(value, BaseCoordinateFrame):
                if isinstance(value, SkyCoord):
                    value = value.frame
                else:
                    raise ValueError("`{}` is not a subclass of "
                                     "`BaseCoordinateFrame` or "
                                     "`SkyCoord`.".format(value))

            # If the observer frame does not contain information about the
            # velocity of the system, assume that the velocity is zero in the
            # system.
            if 's' not in value.data.differentials:
                value = SkyCoord(value, radial_velocity=0 * u.km/u.s).frame

        return value

    @property
    def observer(self):
        """
        The coordinate frame from which the observation was taken.

        Returns
        -------
        `BaseCoordinateFrame`
            The astropy coordinate frame representing the observation.
        """
        return self._observer

    @observer.setter
    def observer(self, value):
        value = self._validate_frame(value)
        self._observer = value

    @property
    def target(self):
        """
        The coordinate frame of the object being observed.

        Returns
        -------
        `BaseCoordinateFrame`
            The astropy coordinate frame representing the target.
        """
        return self._target

    @target.setter
    def target(self, value):
        value = self._validate_frame(value)
        self._target = value

    @property
    def rest(self):
        """
        The rest value of the spectrum used for transformations to/from
        velocity space.

        Returns
        -------
        `Quantity`
            Rest value as an astropy `Quantity` object.
        """
        return self._rest

    @rest.setter
    @u.quantity_input(value=['length', 'frequency', 'energy', 'speed', None])
    def rest(self, value):
        self._rest = value

    @property
    def velocity_convention(self):
        """
        The defined convention for conversions to/from velocity space.

        Returns
        -------
        str
            One of 'optical', 'radio', or 'relativistic' representing the
            equivalency used in the unit conversions.
        """
        return self._velocity_convention

    @velocity_convention.setter
    def velocity_convention(self, value):
        if value not in DOPPLER_CONVENTIONS:
            raise ValueError("Unrecognized velocity convention: {}".format(
                value))

        self._velocity_convention = value

    def transform_frame(self, frame=None, target=None):
        """
        Transform the frame of the observation.

        Parameters
        ----------
        frame : `BaseCoordinateFrame` or `SkyCoord`
            The new observation frame or coordinate.
        target : `SkyCoord`
            The `SkyCoord` object representing
            the target of the observation.

        Returns
        -------
        `SpectralCoord`
            The new coordinate object representing the spectral data
            transformed to the new observer frame.
        """
        if target is not None:
            self.target = target

        # If no velocities are defined in the new observer frame,
        # assume it has zero velocity in their frame. This is handled in the
        # frame validation.
        frame = self._validate_frame(frame)

        # Get the velocity differentials for each frame
        init_obs_vel = self.target.transform_to(self.observer).velocity.get_d_xyz()
        fin_obs_vel = self.target.transform_to(frame).velocity.get_d_xyz()

        # Store the new frame as the current observer frame
        self.observer = frame

        # Calculate the velocity shift between the two vectors
        obs_vel_shift = init_obs_vel - fin_obs_vel

        # Project the velocity shift vector onto the the line-on-sight vector
        # between the target and the new observation frame.
        delta_vel = np.dot(obs_vel_shift, fin_obs_vel) / np.linalg.norm(
            fin_obs_vel)

        # Apply the velocity shift to the stored spectral data
        new_data = (self.to('hz') * (1 + delta_vel / c.cgs)).to(self.unit)
        new_coord = SpectralCoord(new_data.value,
                                  unit=new_data.unit,
                                  rest_value=self.rest,
                                  velocity_convention=self.velocity_convention,
                                  observer=self.observer,
                                  target=self.target)

        return new_coord

    def to(self, *args, rest=None, convention=None, **kwargs):
        """
        Overloaded parent ``to`` method to provide parameters for defining
        rest value and pre-defined conventions for unit transformations.

        Returns
        -------
        `SpectralCoord`
            New spectral coordinate object with data converted to the new unit.
        """
        if rest is not None:
            self.rest = rest

        if convention is not None:
            self.velocity_convention = convention

        equivs = u.spectral()

        if self.rest is not None and self.velocity_convention is not None:
            vel_equiv = DOPPLER_CONVENTIONS[self.velocity_convention](self.rest)
            equivs += vel_equiv

        # Compose the equivalencies for spectral conversions including the
        # appropriate velocity handling.
        kwargs.get('equivalencies', []).append(equivs)

        return super().to(*args, **kwargs)

    def __repr__(self):
        prefixstr = '<' + self.__class__.__name__ + ' '
        sep = ',' if NUMPY_LT_1_14 else ', '
        arrstr = np.array2string(self.view(np.ndarray), separator=sep,
                                 prefix=prefixstr)
        obs_frame = self.observer.__class__.__name__ \
            if self.observer is not None else 'None'
        tar_frame = self.target.__class__.__name__ \
            if self.target is not None else 'None'
        return f'{prefixstr}{arrstr}{self._unitstr:s}, ' \
            f'rest_value={self.rest}, ' \
            f'velocity_convention={self.velocity_convention}, ' \
            f'observer={obs_frame}, target={tar_frame}>'

