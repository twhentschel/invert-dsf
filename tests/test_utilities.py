"""Tests for utility functions."""

import pytest
import numpy as np
import src.utilities as utils


def test_collfreqimag_knownfunc():
    r"""
    Test the collfreqimag Kramers-Kronig transformation against a known
    function.

    In this test, the known function is the response function of a particle
    being forced in a viscous fluid. The equation of motion is

    .. math::

        \tau \frac{\mathrm{d} v}{\mathrm{d} t} + v = F(t)

    where :math:`\tau = m / b` is the mass of the object divided by the damping
    constant, and :math:`F(t)` is the driving force. The response function in
    this case is

    .. math::

        \chi(\omega) = \frac{\nu(\omega)}{F(\omega)} =
            \frac{1}{1 + i \omega \tau}.

    The real and imaginary parts of :math:`\chi(\omega)` are

    .. math::

        \mathrm{Re}(\chi(\omega) &= \frac{1}{1 + \omega^2 \tau^2} \\

        \mathrm{Im}(\chi(\omega) &= \frac{\omega \tau}{1 + \omega^2 \tau^2}

    """

    # real part of known function that obeys Kramers-Kronig
    def freal(x):
        return 1 / (1 + x**2)

    # imaginary part of known function that obeys Kramers-Kronig
    def fimag(x):
        return x / (1 + x**2)

    # domain of f
    omega = np.linspace(0, 15.6, 500)
    # get real part of function at values of omega
    freal_val = freal(omega)

    # Perform Kramers-Kronig transform
    kramkron_imag = utils.collfreqimag(freal_val)

    # perform test, examining the difference between the true imaginary part
    # and the calculated imaginary part
    diff = np.linalg.norm(fimag(omega) - kramkron_imag)
    testres = diff == pytest.approx(0.0, abs=1e-4)
    errmsg = (
        "`collfreqimag` Hilbert based Kramers-Kronig transformation failing."
    )
    assert testres, errmsg
