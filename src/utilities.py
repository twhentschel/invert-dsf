"""Miscellaneous helper functions"""

import numpy as np

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


def kramerskronig(x: ArrayLike, funcreal: ArrayLike) -> ArrayLike:
    """
    Calculates the imaginary part of a function using Kramers-Kronig relations.
    We take advantage of symmetry so the range of the integral is from 0 to
    infinity, where we approximate this integral by assuming the last value of
    `funcreal` is close enough to zero so we don't have to actually go out to
    infinity.

    Parameters:
    - x: Numpy array, grid.
    - funcreal: Numpy array, real function values on the grid.

    Returns:
    - funcimag: Numpy array, imaginary part of the function.
    """
    funcimag = np.zeros_like(x)

    dx = np.zeros(len(x) - 1)
    dx[0] = x[0]
    dx[1:] = (x[2:] - x[:-2]) / 2

    for i in range(1, len(x)):
        # mask to avoid evaluation when i == j
        mask_j = np.ones(len(x), dtype=bool)
        mask_j[i] = False

        funcimag[i] = (
            -2
            / np.pi
            * x[i - 1]
            * np.sum(
                (funcreal[mask_j] - funcreal[i])
                / (x[mask_j] ** 2 - x[i] ** 2)
                * dx
            )
        )

    return funcimag


def elec_loss_fn(
    dielectric: ArrayLike | Mermin,
    wavenum: ArrayLike = None,
    frequency: ArrayLike = None,
    collisionrate: Callable = None,
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

        vals = dielectric(wavenum, frequency, collisionrate)
        return vals.imag / (vals.real**2 + vals.imag**2)

    # convert to array if not already
    dielectric = np.asanyarray(dielectric)

    return dielectric.imag / (dielectric.real**2 + dielectric.imag**2)
