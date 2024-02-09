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


def gendrude(x, center=0, height=1, power=1.5):
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
    return height / (1 + np.abs((x - center) / (np.pi * height)) ** power)


def drude(x, center=0, height=1):
    """Typical Drude collision frequency function. See `gendrude` for
    arguments decscriptions."""
    return gendrude(x, center, height)


def collision_drude_activate_decay(
    x,
    drude_height=1,
    gendrude_height=1,
    gendrude_power=1.5,
    logistic_activate=0,
    logistic_gradient=1,
):
    r"""
    A composite model for the collision frequency made up of multiple
    components that might represent different collisional processes:

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
    drude_height : float
        height argument that goes into the `Drude` function.
    gendrude_height, gendrude_decay: float
        arguments that go into the `gendrude` function.
    logistic_activate, logistic_gradient : float
        arguments that go into the `logistic` function.
    """
    drudebasic = drude(x, center=0, height=drude_height)
    drudedecay = gendrude(
        x, center=0, height=gendrude_height, power=gendrude_power
    )
    sigmoid = logistic(
        x, activate=logistic_activate, gradient=logistic_gradient
    )
    return drudebasic + sigmoid * drudedecay


def collision_activate_decay(
    x,
    lorentzian_height=1,
    lorentzian_powerlaw=1.5,
    logistic_activate=0,
    logistic_gradient=1,
):
    """
    Collision frequency model that represents a generalized free electron
    response and a possible bound electron - free electron interaction.
    Similar to `collision_activate_decay` but neglects the first Drude
    function, so there are only 4 parameters.

    The free electron collision frequency is modeled as a modified Lorentzian/
    Drude function (see `gendrude`), while the bound-free collision frequency
    is modeled as a logistic function (see `logistic`).

    Parameters:
    ___________
    x: array_like
        argument of the function
    lorentzian_height: scalar
        height of the Lorentzian peak
    lorentzian_powerlaw: scalar
        Power that the function follows as x -> infinity
    logistic_activate: scalar
        Point at which the logistic function turns on.
    logistic_gradient: scalar
        How "fast" the logistic function turns on.
    """
    lorentziancollisions = gendrude(
        x, center=0, height=lorentzian_height, power=lorentzian_powerlaw
    )
    logisticcollisions = logistic(
        x, activate=logistic_activate, gradient=logistic_gradient
    )

    # ensures value of at x = 0 is `height`
    factor = 1 + np.exp(logistic_activate * logistic_gradient)

    return lorentziancollisions * logisticcollisions * factor
