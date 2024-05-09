"""Tests for probability models"""

import pytest
import numpy as np
import src.inference.probability_models as pmods


class TestResidual:
    @classmethod
    def setup_class(cls):
        cls.N = 10
        cls.arraydata = np.linspace(0, 10, cls.N)
        cls.arrayx = np.linspace(1, 11, cls.N)
        cls.model = lambda self, x, params: params[0] + params[1] * x
        cls.params = [0, 1]

    def test_residual_outshape(self):
        M, N = 2, 3
        ydata = np.arange(0, 6).reshape((M, N))
        x = np.linspace(0, 1, N)
        res = pmods.residual(self.model, x, ydata, self.params, weight="abs")
        assert res.shape == (6,)

    def test_type_error(self):
        with pytest.raises(NotImplementedError):
            pmods.residual(
                self.model, self.arrayx, self.arraydata, [0, 1], "other"
            )

    def test_point_data_abs(self):
        res = pmods.residual(self.model, 1, 2, [0, 1], weight="abs")
        assert res == 1

    def test_point_data_rel(self):
        res = pmods.residual(self.model, 1, 2, [0, 1], weight="rel")
        assert res == 0.5


class TestUniformLogPrior:
    def test_bad_params_1(self):
        """Test for 2D limits shape"""
        with pytest.raises(ValueError):
            pmods.UniformLogPrior([1, 2, 3])

    def test_bad_params_2(self):
        """Tests for (n, 2) limits shape"""
        with pytest.raises(ValueError):
            pmods.UniformLogPrior([[1, 2, 3], [4, 5, 6]])

    @pytest.mark.parametrize(
        "x, expected", [(0, -np.inf), (1, -np.inf), (-1, -np.inf), (0.5, 0)]
    )
    def test_1D_uniform(self, x, expected):
        """1D uniform distribution"""
        unif1D = pmods.UniformLogPrior([[0, 1]])
        assert unif1D(x) == expected

    @pytest.mark.parametrize(
        "x, expected",
        [
            ([0, 0], -np.inf),
            ([0.5, 1], -np.inf),
            ([-1, 0], -np.inf),
            ([0.5, 0.5], 0),
        ],
    )
    def test_2D_uniform(self, x, expected):
        """1D uniform distribution"""
        unif1D = pmods.UniformLogPrior([[0, 1], [0, 1]])
        assert unif1D(x) == expected


class TestSoftCutoffLogLikelihood:
    @pytest.mark.parametrize("cutoff", (0, -1))
    def test_bad_cutoff(self, cutoff):
        with pytest.raises(ValueError):
            pmods.SoftCutoffLogLikelihood(
                1, 1, lambda x, p: p * x, cutoff=cutoff
            )


class TestLogPosterior:
    def test_posterior_inf_prior(self):
        prior = pmods.UniformLogPrior([[0, 1]])
        likelihood = pmods.SoftCutoffLogLikelihood(
            [1, 2], [3, 4], lambda x, p: p * np.asarray(x)
        )
        posterior = pmods.LogPosterior(prior, likelihood)

        assert posterior(0) == -np.inf
