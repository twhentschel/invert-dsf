"""Main inference loop and helper funtions for sampling the posterior using
MCMC.
"""

from typing import Callable
from numpy.typing import ArrayLike

import emcee
from multiprocessing import Pool
import h5py


def inference_loop(
    initial_state: ArrayLike,
    logposterior: Callable,
    numsamples: int,
    file: str,
    dataset: str = "mcmc_posterior_samples",
    params: dict = None,
    overwrite: bool = False,
) -> emcee.EnsembleSampler:
    """Perform Markov Chain Monte Carlo to sample from the (log) posterior.

    Parameters:
    __________
    initial_state: ArrayLike
        Initial state or starting point of the MCMC chain(s). For m chains and
        n parameters to infer, this has a shape of (m, n)
    logposterior: Callable
        The (log) posterior function. For n parameters, log_posterior maps an
        an array of length n to a single value.
    numsamples: int
        The number of samples we wish to draw from our MCMC run.
    file: str
        Name of HDF5 we are storing data too, including path.
    dataset: str, optional
        Name of the group where we are storing our data (in the HDF5
        file `file`). Default is "mcmc_posterior_samples"
    params: dict, optional
        Dictionary of parameters used in the inference that we want
        to store with our data.
    overwrite: bool, optional
        If True, will overwrite a data with same name that we are proposing for
        current inference data in our HDF5 output file. Default is False.
    """
    nwalkers, ndim = initial_state.shape

    # avoid overwriting previous data
    if not overwrite:
        dataset = unique_hdf5_group(file, dataset)
    # Set up the backend
    backend = emcee.backends.HDFBackend(file, name=dataset)
    # Don't forget to clear it in case the file already exists
    backend.reset(nwalkers, ndim)
    # set inference parameter info in our HDF5 dataset
    if params is not None:
        with h5py.file(file, "w") as f:
            dset = f[dataset]
            for k, v in params.items():
                dset.attrs[k] = v

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            logposterior,
            pool=pool,
            backend=backend,
        )
        sampler.run_mcmc(initial_state, numsamples, progress=True)

    return sampler


def unique_hdf5_group(file: str, group: str, sep: str = "-") -> str:
    """Returns a unique group in the HDF5 file `file`.

    Ex:
    if group := "g" is a group in `file`, then returns "g-1" as long as
    "g-1" is not a group (if it is, then it returns "g-2" and so on).
    """
    with h5py.file(file, "w") as f:
        counter = 1
        uniquegroup = group
        while uniquegroup in f:
            uniquegroup = group + sep + str(counter)

    return uniquegroup
