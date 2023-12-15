"""Miscellaneous helper functions"""

import numpy as np
import scipy.fftpack as ft

from numpy.typing import ArrayLike
from typing import Callable, NamedTuple

from uegdielectric.dielectric import Mermin


class ConstantsTuple(NamedTuple):
    """Values of physical constants used in this work.

    Values from <https://en.wikipedia.org/wiki/Boltzmann_constant>
    """

    boltzmann: float = 8.617333262 * 10**-5  # [eV/kelvin]


Constants = ConstantsTuple()


class AtomicUnitsTuple(NamedTuple):
    """Values to convert to (and from) atomic units.

    Atomic units are typically written as some familiar physical constant.
    For example, 1 Hartree = 27.2114... eV = 1 atomic unit of energy.

    Values from <https://en.wikipedia.org/wiki/Hartree_atomic_units>
    """

    energy: float = 27.21138624598  # Hartree energy [eV]
    length: float = 0.529177210903  # Bohr radius [angstrom]


AtomicUnits = AtomicUnitsTuple()


def collrateimag(collratereal: ArrayLike) -> ArrayLike:
    r"""
    Use the Kramers-Kronig transformation to compute the imaginary part of the
    collision rate `collrateimag` from the real part. Here, since the real
    part is symmetric about $\omega = 0$, the input is assumed to start at
    $\omega = 0$. The true Kramers-Kronig transformation is an integral over
    the whole real line, but if `collratereal` goes to zero before that, we can
    get away with doing an integral that goes until `collratereal` is close
    enough to zero. Thus, its important that the inputs `collratereal` are
    close to zero near the end of the array.

    Notes:
    This relies on the inverse Hilbert transform to transform the real values
    to the imaginary values. The implementation of the Hilbert transform uses
    the fast Fourier transform (FFT) algorithm, which essentially assumes a
    periodic function. In other words, our input values are periodically
    extended over the whole real line (so that
    collratereal(x) = collratereal(x + L) where L range of $\omega$ we are
    considering). For the best results, the start and ending values of
    `collratereal` should be near each other, otherwise a periodic extension
    of this function would result in sharp jumps near the boundaries and yield
    high freqeuency components in the Fourier transform. For the inputs assumed
    here, we artificially extend `collratereal` about $\omega = 0$ before
    feeding it to `scipy.fftpack.ihilbert`.

    collratereal: array
        Values of the real part of the function f.

    returns:
    collrateimag: array
        Values of the imaginary part of the function f.
    """
    real_ext = np.concatenate((collratereal[::-1], collratereal))
    imag_ext = ft.ihilbert(real_ext)
    collrateimag = imag_ext[len(collratereal) :]  # noqa 203
    return collrateimag


def elec_loss_fn(
    dielectric: ArrayLike | Mermin,
    wavenum: ArrayLike = None,
    frequency: ArrayLike = None,
    collisonrate: Callable = None,
) -> ArrayLike:
    r"""
    For given values of the dielectric function :math:`\epsilon`, return the
    Electron Loss Function (ELF):

    .. math:

            \mathrm{Im}(-1 / \epsilon) =
                \frac{\epsilon_i}{\epsilon_r**2 + \epsilon_i**2}

    where the subscripts :math:`i, r` denote the real and imaginary parts,
    respectively.

    If dielectric is `ArrayLike`, then no other arguments need to be used.
    However, if dielectric is of type `AbstractDielectric`, at least `wavenum`
    and `frequency` need to be present.
    """
    if isinstance(dielectric, Mermin):
        if wavenum is None or frequency is None:
            errmssg = (
                "Both wavenum and frequency need to be supplied if\n"
                + "dielectric is of type AbstractDielectric. Received\n"
                + f"type(wavenum) = {type(wavenum)}, "
                + f"type(frequency) = {type(frequency)}."
            )
            raise RuntimeError(errmssg)

        vals = dielectric(wavenum, frequency, collisonrate)
        return vals.imag / (vals.real**2 + vals.imag**2)

    return dielectric.imag / (dielectric.real**2 + dielectric.imag**2)
