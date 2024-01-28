"""Tests for utility functions."""

import pytest
import numpy as np

from uegdielectric.dielectric import Mermin
from uegdielectric import ElectronGas

import src.utilities as utils


def test_kramerskroning_knownfunc():
    r"""
    Test the `kramerkronig` Kramers-Kronig transformation against a known
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
    x = np.linspace(0, 15.6, 500)
    # get real part of function at values of x
    freal_val = freal(x)

    # Perform Kramers-Kronig transform
    kramkron_imag = utils.kramerskronig(x, freal_val)

    # perform test, examining the difference between the true imaginary part
    # and the calculated imaginary part
    diff = np.max(np.abs((fimag(x) - kramkron_imag)))
    testres = diff == pytest.approx(0.0, abs=1e-1)
    errmsg = "Kramers-Kronig transformation (array input) failing."
    assert testres, errmsg


def test_kramerskroningfn_knownfunc():
    r"""
    Test the `kramerkronigfn` Kramers-Kronig transformation against a known
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
    x = np.linspace(0, 15.6, 100)

    # Perform Kramers-Kronig transform
    kramkron_imag = utils.kramerskronigfn(x, freal)

    # perform test, examining the difference between the true imaginary part
    # and the calculated imaginary part
    diff = np.max(np.abs((fimag(x) - kramkron_imag)))
    testres = diff == pytest.approx(0.0, rel=1e-08, abs=1e-06)
    errmsg = "Kramers-Kronig transformation (function input) failing."
    assert testres, errmsg


class TestElectronLossfn:
    """Testing the elec_loss_fn function."""

    testdata = [
        ([[1, 2]], {}, (2,)),
        ([2j], {}, ()),
        (
            [Mermin(ElectronGas(1, 1))],
            {
                "wavenum": [1, 3],
                "frequency": [0.5, 1.0],
                "collisionrate": lambda x: x + 1j * x,
            },
            (2, 2),
        ),
    ]

    @pytest.mark.parametrize("args, kwargs, output_shape", testdata)
    def test_elec_loss_fn_inputs(self, args, kwargs, output_shape):
        """
        Testing the elec_loss_fn function against different inputs.
        """
        assert utils.elec_loss_fn(*args, **kwargs).shape == output_shape

    def test_elec_loss_fn_error(self):
        """
        Test that an error is raised when calling `elec_loss_fn` with a Mermin
        dielectric object without other arguments.
        """
        with pytest.raises(Exception):
            utils.elec_loss_fn(Mermin(ElectronGas(1, 1)))
