"""Collision frequency models."""

import numpy as np
from numpy.typing import ArrayLike
import ctypes
from scipy import LowLevelCallable, integrate

from src.inference import collision_models_cy
from src.utilities import kramerskronig_fullintegrand, kramerskronig


def logistic_peak(x, activate=0, growth_rate=1, decay_power=0):
    """
    Logistic function with a controllable decay term.

    Parameters:
    ___________
    x: array_like
        Argument of the function.
    activate: float
        The point at which the logistic function is 1/2.
    growth_rate: float
        Roughly the rate of increase of the logistic function before
        `x` = `activate`. The larger this quantity, the slower the rise of this
        function.
    decay_power: float
        The power of the decay term, which governs the power law of the
        function as x -> +infinity
    """
    return 1 / (
        1
        + np.exp(-(x - activate) / growth_rate)
        + np.abs(x / activate) ** decay_power
    )


def screened_born_approx(x, height, width):
    r"""
    Function that has similar low- and high-frequency limits as the Born
    collision frequency in the presence of electron-ion screening. Units are
    such that the output is in atomic units of inverse seconds.

    Parameters:
    ___________
    x: array_like
        Argument of the function.
    height: float
        Height of the peak.
    width: float
        Controls the width of the peak.
    """
    return height / (1.0 + (x / width) ** 1.5)


def born_logpeak_model(
    x,
    born_height,
    born_width,
    logpeak_height,
    logpeak_activate,
    logpeak_growrate,
    logpeak_decay,
):
    """
    Collision frequency model that represents a generalized free electron
    response and a possible inelastic interaction.

    The free electron collision frequency is modeled as a modified using an
    approximation to the Born collision frequency (see `screened_born_approx`),
    while the inelasatic collision frequency is modeled as a logistic-peak
    function (see `logistic_peak`).

    Units are such that the output is in atomic units of inverse seconds.

    Parameters:
    ___________
    x: array_like
        argument of the function
    born_height: scalar
        height of the Born function peak
    born_width: scalar
        Controls the width of the Born peak
    logpeak_height: scalar
        Height of the logistic function.
    logpeak_activate: scalar
        Point at which the logistic function turns on.
    logpeak_growrate: scalar
        The rate at which the logistic function turns on.
    logpeak_decay: float
        The power of the decay term, which governs the power law of the
        function as x -> +infinity
    """

    borncollisions = screened_born_approx(x, born_height, born_width)
    inelasticcollisions = logistic_peak(
        x,
        activate=logpeak_activate,
        gradient=logpeak_growrate,
        decay_power=logpeak_decay,
    )

    return borncollisions + logpeak_height * inelasticcollisions


def born_logpeak_model_imag(
    x,
    born_height,
    born_width,
    logpeak_height,
    logpeak_activate,
    logpeak_growrate,
    logpeak_decay,
):
    """
    Imaginary part of the `born_logpeak_model` function, calculated using
    the Kramers-Kronig transformation.

    Units are such that the output is in atomic units of inverse seconds.
    """
    if np.any(x <= 0):
        raise ValueError("Only accepts positive values for the argument x")

    params = (ctypes.c_double * 6)(
        born_height,
        born_width,
        logpeak_height,
        logpeak_activate,
        logpeak_growrate,
        logpeak_decay,
    )
    user_data = ctypes.cast(params, ctypes.c_void_p)

    kramkronint = LowLevelCallable.from_cython(
        collision_models_cy, "scipy_kramerskronig_integrand", user_data
    )
    cauchyint = LowLevelCallable.from_cython(
        collision_models_cy, "scipy_cauchy_integrand", user_data
    )

    return kramerskronig_fullintegrand(x, cauchyint, kramkronint)


def inverse_screening_length(temperature, density):
    fermi_energy = 0.5 * (3 * np.pi**2 * density) ** (2 / 3)
    effective_temp = np.maximum(temperature, fermi_energy)
    return np.sqrt(4 * np.pi * density / effective_temp)


class BornLogPeak:
    r"""
    Approximate-Born inelastic collision frequency model.

    The real part of the model is given by
    .. math ::

        \mathrm{Re}\{\nu(\omega)\} = \frac{\nu_0}{1 + (\omega / b)^{3/2}}
            + \frac{\nu_1}{1 + \exp(-\alpha (\omega - \omega_a))
                + (\omega / \omega_a)^{p_1}}

    where $b$ is chosen such that the first term (the approximate "Born" term)
    integrates to the same value that the true Born collision frequency
    integrates to, and ..math::`\nu_0, \nu_1, \alpha, \omega_a, p_1` are
    parameters of the model. The second term represents inelastic collisions
    left out of the Born model.

    The imaginary part is determined from the Kramers-Kronig relations.

    Parameters
    ----------
    temperature: float
        Electron temperature, atomic units.
    density: float
        Electron density, atomic units.
    chemicapot: float
        Electron chemical potential, atomic units.
    chargestate: float
        Charge state/average ionization/Z star of the material at the current
        electronic conditions.
    """

    def __init__(
        self, temperature, density, chemicalpot, chargestate: float
    ) -> None:
        self.temperature = temperature
        self.density = density
        self.chemicalpot = chemicalpot
        self.chargestate = chargestate

    def pintegral_screening(self):
        inv_screen_len = inverse_screening_length(
            self.temperature, self.density
        )

        def integrand(p):
            fermi_term = 1 / (
                1 + np.exp((p**2 / 2 - self.chemicalpot) / self.temperature)
            )
            screeningterm = np.pi * np.arctan(
                2 * p / inv_screen_len
            ) - np.pi * 2 * p * inv_screen_len / (
                2 * (4 * p**2 + inv_screen_len**2)
            )
            return p * fermi_term * screeningterm

        # integrate
        p1 = np.geomspace(1e-4, self.chemicalpot, 1000, endpoint=False)
        p2 = np.geomspace(
            self.chemicalpot,
            np.sqrt(20 * self.temperature + 2 * np.abs(self.chemicalpot)),
            1000,
        )
        p = np.concatenate((p1, p2))
        return integrate.trapezoid(integrand(p), p)

    def born_integral_screening(self):
        return 4 * self.chargestate / (3 * np.pi) * self.pintegral_screening()

    def approx_born_width_integral_preserving(self, bornheight):
        """
        Returns the correct width parameter in `screened_born_approx` so that
        the integral agrees with the integral of  the exact screened Born
        collision frequency theory.
        """
        return (
            3
            * np.sqrt(3)
            * self.born_integral_screening()
            / bornheight
            / (4 * np.pi)
        )

    def real(self, x: ArrayLike, params: ArrayLike) -> ArrayLike:
        """
        Real part of the collision frequency model.

        Units are such that the output is in atomic units of inverse seconds.
        """
        return born_logpeak_model(
            x,
            born_height=params[0],
            born_width=self.approx_born_width_integral_preserving(params[0]),
            logpeak_height=params[1],
            logpeak_activate=params[2],
            logpeak_growrate=params[3],
            logpeak_decay=params[4],
        )

    def imag(self, x: ArrayLike, params: ArrayLike) -> ArrayLike:
        """
        Imaginary part of the collision frequency model.

        Units are such that the output is in atomic units of inverse seconds.
        """
        return born_logpeak_model_imag(
            x,
            born_height=params[0],
            born_width=self.approx_born_width_integral_preserving(params[0]),
            logpeak_height=params[1],
            logpeak_activate=params[2],
            logpeak_growrate=params[3],
            logpeak_decay=params[4],
        )

    def __call__(self, x: ArrayLike, params: ArrayLike) -> ArrayLike:
        """
        Complex collision frequency.

        Units are such that the output is in atomic units of inverse seconds.
        """
        return self.real(x, params) + 1j * self.imag(x, params)


class ScreenedBorn:
    """
    Born collision frequency theory in the presence of electron-ion screening.

    Uses the Born collision frequency with the Born-Yukawa scattering cross
    section with an effective screening length.

    Parameters
    ----------
    temperature: float
        Electron temperature, atomic units.
    density: float
        Electron density, atomic units.
    chemicapot: float
        Electron chemical potential, atomic units.
    chargestate: float
        Charge state/average ionization/Z star of the material at the current
        electronic conditions.
    """

    def __init__(self, temperature, density, chemicalpot, chargestate) -> None:
        self.temperature = temperature
        self.density = density
        self.chemicalpot = chemicalpot
        self.chargestate = chargestate
        self.inv_screen_len = inverse_screening_length(temperature, density)

    def _RPAimag(self, wavenum, freq):
        scalarfreq = False
        if np.ndim(freq) == 0:
            scalarfreq = True
            freq = np.array(freq, ndmin=1)
        if np.ndim(wavenum) == 0:
            scalarfreq = True
            wavenum = np.array(wavenum, ndmin=1)

        wavenum, freq = np.meshgrid(wavenum, freq, copy=False, sparse=True)
        a2 = (2 * freq - wavenum**2) ** 2 / (2 * wavenum) ** 2
        b2 = (2 * freq + wavenum**2) ** 2 / (2 * wavenum) ** 2

        numer = 1 + np.exp((self.chemicalpot - a2 / 2) / self.temperature)
        denom = 1 + np.exp((self.chemicalpot - b2 / 2) / self.temperature)

        result = 2 * self.temperature / wavenum**3 * np.log(numer / denom)

        if scalarfreq:
            return np.squeeze(result)
        return result

    def real(self, freq):
        """
        Real part of the collision frequency.

        Units are such that the output is in atomic units of inverse seconds.
        """

        def integrand(wavenum):
            return (
                wavenum**6
                / (wavenum**2 + self.inv_screen_len**2) ** 2
                * self._RPAimag(wavenum, freq)
            )

        q = np.geomspace(1e-6, 1e3, 2000)
        integral = integrate.trapezoid(integrand(q), q)

        return np.where(
            freq == 0, 0, 2 * self.chargestate * integral / (3 * np.pi * freq)
        )

    def imag(self, freq):
        """
        Imaginary part of the collision frequency.

        Because the real part is expensive to call, it turns out to be cheaper
        if we amortize the real part by interpolating it on a dense grid. The
        functional form is pretty simple, and most of the action happens close
        to zero, so we do this on a log-grid. However, this is not meant to be
        a "finished-product", so make sure it works for your use case.

        Units are such that the output is in atomic units of inverse seconds.
        """
        xx = np.geomspace(1e-4, 1e4, 3000)
        r = self.real(xx)
        i = kramerskronig(freq, lambda x: np.interp(x, xx, r))
        return i

    def __call__(self, freq):
        """
        Complex collision frequency.

        Units are such that the output is in atomic units of inverse seconds.
        """
        return self.real(freq) + 1j * self.imag(freq)
