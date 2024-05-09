"""Collision frequency models."""

from libc.math cimport exp


cdef double logistic_peak(
    double x, double activate, double gradient, double decay_power
 ) except *:
    """
    Logistic function with a controllable decay term.

    Parameters:
    ___________
    x: array_like
        Argument of the function.
    activate: float
        The point at which the logistic function is 1/2.
    gradient: float
        The slope of the logistic function at `x` = `activate`. The larger this
        quantity, the faster the rise of this function.
    decay_power: float
        The power of the decay term, which governs the power law of the
        function as x -> +infinity
    """
    return 1 / (
        1 + exp(-gradient * (x - activate)) + (x / activate) ** decay_power
    )


cdef double screened_born_approx(
    double x, double height, double width
) except *:
    r"""
    Function that has similar low- and high-frequency limits as the Born
    collision frequency in the presence of electron-ion screening.

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


cdef double born_logpeak_model(
    double x,
    double born_height,
    double born_width,
    double logpeak_height,
    double logpeak_activate,
    double logpeak_gradient,
    double logpeak_decay
) except *:
    """
    Collision frequency model that represents a generalized free electron
    response and a possible inelastic interaction.

    The free electron collision frequency is modeled as a modified using an
    approximation to the Born collision frequency (see `screened_born_approx`),
    while the inelasatic collision frequency is modeled as a logistic-peak
    function (see `logistic_peak`).

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
    logpeak_gradient: scalar
        How "fast" the logistic function turns on.
    logpeak_decay: float
        The power of the decay term, which governs the power law of the
        function as x -> +infinity
    """
    
    borncollisions = screened_born_approx(x, born_height, born_width)
    inelasticcollisions = logistic_peak(
        x,
        activate=logpeak_activate,
        gradient=logpeak_gradient,
        decay_power=logpeak_decay
    )

    return borncollisions + logpeak_height * inelasticcollisions

cdef double scipy_cauchy_integrand(int n, double *xx, void *user_data):
    """ Alternate Lowlevel callback interface for scipy.integrate.quad, with
    the cauchy weight function to perform Kramers-Kronig integration.
    """
    cdef double height1 = (<double *>user_data)[0]
    cdef double width = (<double *>user_data)[1]
    cdef double height2 = (<double *>user_data)[2]
    cdef double activate = (<double *>user_data)[3]
    cdef double gradient = (<double *>user_data)[4]
    cdef double decay = (<double *>user_data)[5]
    cdef double cauchyprinciplepoint = xx[1]

    return born_logpeak_model(
        xx[0], height1, width, height2, activate, gradient, decay
    ) / (xx[0] + xx[1])

cdef double scipy_kramerskronig_integrand(int n, double *xx, void *user_data):
    """ Alternate Lowlevel callback interface for scipy.integrate.quad
    to perform Kramers-Kronig integration.
    """
    cdef double height1 = (<double *>user_data)[0]
    cdef double width = (<double *>user_data)[1]
    cdef double height2 = (<double *>user_data)[2]
    cdef double activate = (<double *>user_data)[3]
    cdef double gradient = (<double *>user_data)[4]
    cdef double decay = (<double *>user_data)[5]
    cdef double cauchyprinciplepoint = xx[1]

    return born_logpeak_model(
        xx[0], height1, width, height2, activate, gradient, decay
    ) / (xx[0]**2 - xx[1]**2)
