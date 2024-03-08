"""Tests for utility functions."""

import pytest
import numpy as np

from uegdielectric.dielectric import Mermin
from uegdielectric import ElectronGas

import src.utilities as utils


class TestKramersKronig:
    """Tests for Kramers-Kronig transformation"""

    examples_kramerskronig_funcs = [
        # function 0
        {"real": lambda x: 1 / (1 + x**2), "imag": lambda x: x / (1 + x**2)},
        # function 1
        {
            "real": lambda x: (1 - x**2) / (x**4 - x**2 + 1),
            "imag": lambda x: x / (x**4 - x**2 + 1),
        },
    ]

    @pytest.mark.parametrize("kkfunc", examples_kramerskronig_funcs)
    def test_kramerskronig_array(self, kkfunc):
        x = np.linspace(0, 50, 1_000)
        freal = kkfunc["real"]
        fimag = kkfunc["imag"]

        kramkron_imag = utils.kramerskronig_arr(x, freal(x))

        # perform test, examining the difference between the true imaginary
        # part and the calculated imaginary part
        diff = np.max(np.abs((fimag(x) - kramkron_imag)))
        testres = diff == pytest.approx(0.0, abs=5e-1)
        errmsg = "Kramers-Kronig transformation (array input) failing."
        assert testres, errmsg

    @pytest.mark.parametrize("kkfunc", examples_kramerskronig_funcs)
    def test_adaptive_kramerskronig(self, kkfunc):
        x = np.linspace(1e-6, 10, 10)
        freal = kkfunc["real"]
        fimag = kkfunc["imag"]

        kramkron_imag = utils.kramerskronig(x, freal)

        # perform test, examining the difference between the true imaginary
        # part and the calculated imaginary part
        diff = np.max(np.abs((fimag(x) - kramkron_imag)))
        testres = diff == pytest.approx(0.0, abs=1e-8)
        errmsg = (
            "Kramers-Kronig transformation (adaptive integration) failing."
        )
        assert testres, errmsg

    @pytest.mark.parametrize("kkfunc", examples_kramerskronig_funcs)
    def test_adaptive_kramerskronig_fullintegrand(self, kkfunc):
        x = np.linspace(1e-6, 10, 10)

        kramkron_imag = utils.kramerskronig_fullintegrand(
            x,
            lambda x, p: kkfunc["real"](x) / (x + p),
            lambda x, p: kkfunc["real"](x) / (x**2 - p**2),
        )

        # perform test, examining the difference between the true imaginary
        # part and the calculated imaginary part
        diff = np.max(np.abs((kkfunc["imag"](x) - kramkron_imag)))
        testres = diff == pytest.approx(0.0, abs=1e-8)
        errmsg = (
            "Kramers-Kronig transformation (adaptive integration, full "
            + "integrand specified) failing."
        )
        assert testres, errmsg

    @pytest.mark.parametrize("kkfunc", examples_kramerskronig_funcs)
    def test_adaptive_kramerskronig_extendedrange(self, kkfunc):
        x = np.geomspace(1e-3, 1e3, 100)
        freal = kkfunc["real"]
        fimag = kkfunc["imag"]

        kramkron_imag = utils.kramerskronig(x, freal)

        # perform test, examining the difference between the true imaginary
        # part and the calculated imaginary part
        diff = np.max(np.abs((fimag(x) - kramkron_imag)))
        testres = diff == pytest.approx(0.0, abs=1e-7)
        errmsg = (
            "Kramers-Kronig transformation (adaptive integration, extended "
            + "range) failing."
        )
        assert testres, errmsg

    def test_adaptive_at_zero(self):
        """Evaluating at zero raises an error for now."""
        with pytest.raises(ValueError):
            utils.kramerskronig(0, lambda x: 1)


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
