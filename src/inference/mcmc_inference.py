"""Main inference loop and helper funtions for sampling the posterior using
MCMC.
"""

from typing import Callable
from numpy.typing import ArrayLike

import emcee
import numpy as np
from multiprocessing import Pool
import h5py


def inference_loop(
    initial_state: ArrayLike,
    logposterior: Callable,
    numsamples: int,
    samplesfile: str,
    dataset: str = None,
    runinfo: dict = None,
    overwrite: bool = False,
    **emceekwargs: dict,
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
    samplesfile: str
        Name of HDF5 we are storing sampling data too, including path.
    dataset: str, optional
        Name of the group where we are storing our data (in the HDF5
        file `file`). Default is None, which becomes "mcmc" due to `emcee`.
    runinfo: dict, optional
        Dictionary of information and parameters used in the inference that we
        want to store with our data.
    overwrite: bool, optional
        If True, will overwrite `dataset` with new inference data in our HDF5
        output file. If False, attempts to add new samples to `dataset`.
        Default is False.
    emceekwargs: dict
        Keyword aruments for `emcee.EnsembleSampler`.
    """
    numchains, ndim = initial_state.shape

    # Set up the backend
    if dataset is None:
        backend_kwargs = {}
    else:
        backend_kwargs = {"name": dataset}
    backend = emcee.backends.HDFBackend(samplesfile, **backend_kwargs)

    if overwrite:
        # Don't forget to clear it in case the file already exists
        backend.reset(numchains, ndim)
    else:
        # check if dataset currently exists
        if dataset_exists(samplesfile, dataset):
            # Don't need initial state if adding to existing data
            initial_state = None

    # perform ensemble MCMC sampling of logposterior
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            numchains,
            ndim,
            logposterior,
            pool=pool,
            backend=backend,
            **emceekwargs,
        )
        sampler.run_mcmc(initial_state, numsamples, progress=True)

    # set inference parameter info in our HDF5 dataset
    if runinfo is not None:
        # add information to h5py file
        with h5py.File(samplesfile, "a") as f:
            dset = f[dataset]
            for k, v in runinfo.items():
                dset.attrs[k] = v

    return sampler

def dataset_exists(filename, dataset_name):
    with h5py.File(filename, 'r') as f:
        return dataset_name in f

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


def flat_mcmc_samples(
    backend: emcee.backends.Backend, printmcmcinfo: bool = True
) -> ArrayLike:
    """Combine the ensemble of chains stored in `backend` into a set of
    approximately indepedent samples that are burned-in and thinned.
    """
    try:
        tau = backend.get_autocorr_time()
    except emcee.autocorr.AutocorrError as e:
        print(e)
        tau = backend.get_autocorr_time(tol=0)

    burnin = int(2 * np.max(tau))
    thin = max(1, int(0.5 * np.min(tau)))
    flat_samples = backend.get_chain(discard=burnin, flat=True, thin=thin)

    if printmcmcinfo:
        print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(tau)))
        print("burn-in: {0}".format(burnin))
        print("thin: {0}".format(thin))
        print("flat chain shape: {0}".format(flat_samples.shape))
        print(
            "Mean acceptance fraction: {0:.3f}".format(
                np.mean(backend.accepted / backend.iteration)
            )
        )

    return flat_samples
