"""Collision frequency models."""

import numpy as np
import ctypes
from scipy import LowLevelCallable

from src.inference import collision_models_cy
from src.utilities import kramerskronig_fullintegrand


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

    return lorentziancollisions * logisticcollisions


# def collision_activate_decay_imag(
#     x,
#     lorentzian_height,
#     lorentzian_powerlaw,
#     logistic_activate,
#     logistic_gradient,
# ):
#     """
#     Imaginary part of the `collision_activate_decay` model, calculated using
#     the Kramers-Kronig transformation.
#     """
#     if (
#         np.any(x <= 0)
#         or lorentzian_height < 0
#         or lorentzian_powerlaw <= 0
#         or logistic_activate < 0
#         or logistic_gradient <= 0
#     ):
#         raise ValueError("Only accepts positive values for parameters")

#     params = (ctypes.c_double * 4)(
#         lorentzian_height,
#         lorentzian_powerlaw,
#         logistic_activate,
#         logistic_gradient,
#     )
#     user_data = ctypes.cast(params, ctypes.c_void_p)

#     funcreal_kramkron = LowLevelCallable.from_cython(
#         collision_models_cy, "scipy_kramerskronig_integrand", user_data
#     )
#     funcreal_cauchy = LowLevelCallable.from_cython(
#         collision_models_cy, "scipy_cauchy_integrand", user_data
#     )

#     funcimag = np.zeros_like(x)

#     for i in range(len(funcimag)):
#         # difficult region around the function 1/(y^2 - x[i]^2)
#         invsq_diffregion = 50
#         effectiveinfty = x[i] + invsq_diffregion
#         breakpoint1 = invsq_diffregion
#         breakpoint2 = x[i] - invsq_diffregion
#         splitintegral = breakpoint1 < breakpoint2

#         if splitintegral:
#             # nothing exciting in the integrand in these regions
#             funcimag[i] += integrate.quad(
#                 funcreal_kramkron,
#                 0,
#                 breakpoint2,
#                 points=[breakpoint1],
#                 args=(x[i]),
#             )[0]
#         # integrate cauchy singularity
#         funcimag[i] += integrate.quad(
#             funcreal_cauchy,
#             breakpoint2 if splitintegral else 0,
#             effectiveinfty,
#             weight="cauchy",
#             wvar=x[i],
#             args=(x[i]),
#         )[0]
#         # integrate out to infinity
#         funcimag[i] += integrate.quad(
#             funcreal_kramkron, effectiveinfty, np.inf, args=(x[i])
#         )[0]

#     funcimag = -2 / np.pi * funcimag * x
#     return funcimag


def collision_activate_decay_imag(
    x,
    lorentzian_height,
    lorentzian_powerlaw,
    logistic_activate,
    logistic_gradient,
):
    """
    Imaginary part of the `collision_activate_decay` model, calculated using
    the Kramers-Kronig transformation.
    """
    if (
        np.any(x <= 0)
        or lorentzian_height < 0
        or lorentzian_powerlaw <= 0
        or logistic_activate < 0
        or logistic_gradient <= 0
    ):
        raise ValueError("Only accepts positive values for parameters")

    params = (ctypes.c_double * 4)(
        lorentzian_height,
        lorentzian_powerlaw,
        logistic_activate,
        logistic_gradient,
    )
    user_data = ctypes.cast(params, ctypes.c_void_p)

    kramkronint = LowLevelCallable.from_cython(
        collision_models_cy, "scipy_kramerskronig_integrand", user_data
    )
    cauchyint = LowLevelCallable.from_cython(
        collision_models_cy, "scipy_cauchy_integrand", user_data
    )

    return kramerskronig_fullintegrand(x, cauchyint, kramkronint)


# import time
# from src.utilities import kramerskronigfn, kramerskronig

# x = np.linspace(1e-6, 9, 100)
# x2 = np.linspace(1e-6, 9, 1000)

# a = time.time()
# kramerskronigfn(x, lambda y: collision_activate_decay(y, 0.1, 0.1, 10, 1e1))
# print(f"function Kramers-Kronig: {time.time() - a} s")

# a = time.time()
# kramerskronig(x2, collision_activate_decay(x2, 0.1, 0.1, 10, 1e1))
# print(f"array Kramers-Kronig: {time.time() - a} s")

# a = time.time()
# collision_activate_decay_imag(x, 0.1, 0.1, 10, 1e1)
# print("function Kramers-Kronig, Cython LowLevelCallable: {} s",
#   time.time() - a)

# a = time.time()
# kramerskronigfn(
#     x,
#     lambda y: collision_models_cy.collision_activate_decay(
#         y, 0.1, 0.1, 10, 1e1
#     ),
# )
# print(f"function Kramers-Kronig, Cython: {time.time() - a} s")
