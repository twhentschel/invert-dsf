""" Functions used to define probability models """

import numpy as np
from numpy.typing import ArrayLike
from abc import abstractmethod, ABC
from typing import Callable


class LogProbabilityBase(ABC):
    """Base log-probability function class"""

    @abstractmethod
    def __call__(self, params: ArrayLike) -> float:
        pass


class LogPosterior(LogProbabilityBase):
    """Class for the log posterior function"""

    def __init__(
        self, logprior: LogProbabilityBase, loglikelihood: LogProbabilityBase
    ) -> None:
        self.logprior = logprior
        self.loglikelihood = loglikelihood

    def __call__(self, params: ArrayLike) -> float:
        logprior_eval = self.logprior(params)
        loglikelihood_eval = self.loglikelihood(params)

        if not (
            np.isfinite(logprior_eval) and np.isfinite(loglikelihood_eval)
        ):
            return -np.inf

        return logprior_eval + loglikelihood_eval


class UniformLogPrior(LogProbabilityBase):
    """Uniform prior distribution over parameters

    params: ArrayLike
        Parameters that go into `model`
    limits: ArrayLike shape (n, 2)
        Lower and upper bounds of the uniform distribution for each parameter.
        For only one parameter, this argument must still be 2-dimesional:
        [[lower, upper]]

    """

    def __init__(self, limits: ArrayLike):
        limits = np.asanyarray(limits)
        if limits.ndim != 2:
            raise ValueError(
                "hyperparams must be 2 dimensional; it has shape"
                + f"{limits.shape}"
            )
        if limits.shape[1] != 2:
            raise ValueError(
                "hyperparams must be of shape (n, 2); it has shape"
                + f"{limits.shape}"
            )
        self.limits = limits

    def __call__(self, params: ArrayLike) -> float:
        params = np.asanyarray(params)
        if np.all(self.limits[:, 0] < params) and np.all(
            params < self.limits[:, 1]
        ):
            return 0.0
        return -np.inf


class SoftCutoffLogLikelihood(LogProbabilityBase):
    """Log likelihood functions that is a soft (exponential) cutoff function

    Parameters:
    ydata: ArrayLike
        Observed data
    x: ArrayLike
        Indepedent variable, points at which we observe `ydata`
    model: Callable
        Model that describe the data: `model(x, params) = ydata`
    params: ArrayLike
        Parameters that go into `model`
    cutoff: scalar, default 0.01
        Cutoff point
    residualweight: str or array_like, default "abs"
        Weighting of the residual function. Specific to the `residual` function
        defined in this file.
    """

    def __init__(
        self,
        ydata: ArrayLike,
        x: ArrayLike,
        model: Callable,
        cutoff: float = 0.01,
        residualweight: str | ArrayLike = "abs",
    ) -> None:
        self.ydata = ydata
        self.x = x
        self.model = model

        if cutoff <= 0:
            raise ValueError(
                f"hyperparams {cutoff} must be a positive and nonzero"
                + " scalar."
            )
        self.cutoff = cutoff
        self.residualweight = residualweight

    def __call__(self, params: ArrayLike) -> float:
        residualeval = residual(
            self.model, self.x, self.ydata, params, self.residualweight
        )

        return -np.max((residualeval / (np.sqrt(2) * self.cutoff)) ** 2)


class SquaredExpLogLikelihood(LogProbabilityBase):
    """Squared exponential log likelihood function"""

    def __init__(
        self,
        ydata: ArrayLike,
        x: ArrayLike,
        model: Callable,
        stddev: float = 0.01,
        residualweight: str | ArrayLike = "abs",
    ) -> None:
        self.ydata = ydata
        self.x = x
        self.model = model

        if stddev <= 0:
            raise ValueError(
                f"hyperparams {stddev} must be a positive and nonzero"
                + " scalar."
            )
        self.stddev = stddev
        self.residualweight = residualweight

    def __call__(self, params: ArrayLike) -> float:
        residualeval = residual(
            self.model, self.x, self.ydata, params, self.residualweight
        )

        return (
            -np.linalg.norm(residualeval / (np.sqrt(2) * self.stddev)) ** 2
        )


def residual(
    model: Callable,
    x: ArrayLike,
    ydata: ArrayLike,
    params: ArrayLike,
    weight: str | ArrayLike = "abs",
) -> ArrayLike:
    """
    Returns a residual function between the data and `model` function.
    Can be used with multi-dimensional array inputs as long as `ydata`
    and `model(x, params)` are broadcastable.

    Parameters
    ----------
    model: Callable
        Function that models the data with the signature `model(x, params)`
        and returns an array or scalar with the same shape as `ydata`.
    x : array_like
        Input/indepedent data points
    ydata: array_like
        Depedent data
    params: array_like
        parameters of our model
    weight: str or array_like
        string or array to weight the residual, with strings referring to
        specific weightings for backwards compatibility. Supported string
        options:
        "abs" : absolute residual -> weight = 1
        "rel" : relative residual -> weight = 1 / ydata

        Default is "abs".

    returns
    -------
        residual of the model with respect to some weighting:
        residual = (ydata - model(x, params)) * weight. 1D array_like object.
    """
    if isinstance(weight, str):
        match weight:
            case "abs":
                weight = 1
            case "rel":
                weight = 1 / ydata
            case _:
                m = f"Weight type {weight} not implemented."
                raise NotImplementedError(m)

    ydata = np.asanyarray(ydata)

    return ((ydata - model(x, params)) * weight).flatten()
