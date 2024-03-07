"""Collision frequency models."""

from libc.math cimport exp, fabs, pi


cdef double logistic(
    double x, double activate, double gradient
 ) except *:
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
    return 1 / (1 + exp(-gradient * (x - activate)))


cdef double gendrude(
    double x, double center, double height, double power
) except *:
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
    return height / (1.0 + fabs((x - center) / (pi * height)) ** power)

cpdef double collision_activate_decay(
    double x,
    double lorentzian_height,
    double lorentzian_powerlaw,
    double logistic_activate,
    double logistic_gradient
) except *:
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
    # forcing symmetry
    # x = fabs(x)
    
    lorentziancollisions = gendrude(
        x, 0, lorentzian_height, lorentzian_powerlaw
    )
    logisticcollisions = logistic(
        x, activate=logistic_activate, gradient=logistic_gradient
    )

    return lorentziancollisions * logisticcollisions

cdef double coll_act_decay_scipy(double x, void *user_data):
    """ Lowlevel callback interface for `collision_activate_decay` for
    scipy.integrate.quad

    """
    cdef double height = (<double *>user_data)[0]
    cdef double powerlaw = (<double *>user_data)[1]
    cdef double activate = (<double *>user_data)[2]
    cdef double gradient = (<double *>user_data)[3]

    return collision_activate_decay(x, height, powerlaw, activate, gradient)

cdef double scipy_cauchy_integrand(int n, double *xx, void *user_data):
    """ Alternate Lowlevel callback interface for scipy.integrate.quad, with
    the cauchy weight function to perform Kramers-Kronig integration.
    """
    cdef double height = (<double *>user_data)[0]
    cdef double powerlaw = (<double *>user_data)[1]
    cdef double activate = (<double *>user_data)[2]
    cdef double gradient = (<double *>user_data)[3]
    cdef double cauchyprinciplepoint = xx[1]

    return collision_activate_decay(xx[0], height, powerlaw, activate, gradient) / (xx[0] + xx[1])

cdef double scipy_kramerskronig_integrand(int n, double *xx, void *user_data):
    """ Alternate Lowlevel callback interface for scipy.integrate.quad
    to perform Kramers-Kronig integration.
    """
    cdef double height = (<double *>user_data)[0]
    cdef double powerlaw = (<double *>user_data)[1]
    cdef double activate = (<double *>user_data)[2]
    cdef double gradient = (<double *>user_data)[3]
    cdef double cauchyprinciplepoint = xx[1]

    return collision_activate_decay(xx[0], height, powerlaw, activate, gradient) / (xx[0]**2 - xx[1]**2)
