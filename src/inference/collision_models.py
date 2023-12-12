"""Collision frequency models."""

import numpy as np


def logistic(x, activate=0, gradient=1):
    """
    Logistic function

    Parameters:
    ___________
    x: array_like
        Argument of the function.
    activate: float
        The point at which the logistic function is 1/2.
    gradient: float
        The slope of the logistic function at `x` = `activate`. The larger this
        quantity, the faster the rise of this function.
    """
    return 1 / (1 + np.exp(-gradient * (x - activate)))


def gendrude(x, center=0, height=1, decay=1.5):
    r"""
    Generalized Drude collision frequency function where the power-law decay is
    adjustable.

    Parameters:
    ___________
    x: array_like
        Argument of the function.
    center: float
        The location of the peak.
    height: float
        Height of the peak.
    decay: float
        The power at which the peak decays as :math: `|x| \rightarrow \infty`.
    """
    return height / (1 + ((x - center) / (np.pi * height)) ** decay)


def drude(x, center=0, height=1):
    """Typical Drude collision frequency function. See `gendrude` for
    arguments decscriptions."""
    return gendrude(x, center, height)


def collision_activate_decay(x, drudeargs, logisticargs, gendrudeargs):
    r"""
    A composite model for the collision frequency made up of multiple
    components that might represent different physical processes:

    .. math::

        \nu(\omega) = \nu_\mathrm{D'}(\omega; \nu_0, 3/2) +
            \frac{h}{1 + e^{-\alpha(\omega - \omega_0)}} \times
            \nu_\mathrm{D'}(\omega; \nu_1, \alpha)

    where :math:`\nu_\mathrm{D'}(\omega; h, \alpha) = \frac{h}{1 + (\omega /
    \pi h)^{\alpha}}` is the generalized Drude function centered at 0.

    Parameters:
    ___________
    x : array_like
        argument of function
    drudeargs : dict
        Keyword arguments that go into the `Drude` function.
    logisitcargs : dict
        Keyword arguments that go into the `logistic` function.
    gendrudeargs: dict
        Keyword arguments that go into the `gendrude` function.

    """
    drudebasic = drude(x, **drudeargs)
    sigmoid = logistic(x, **logisticargs)
    drudedecay = gendrude(x, **gendrudeargs)
    return drudebasic + sigmoid * drudedecay
