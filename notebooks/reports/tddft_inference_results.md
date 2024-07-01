---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
plt.style.use(["seaborn-v0_8-paper", "paper-style.mplstyle"])
import seaborn as sns
from scipy import optimize
import h5py

import emcee 
from uegdielectric import ElectronGas
from uegdielectric.dielectric import Mermin

from src.inference.collision_models import BornLogPeak
from src.inference.mcmc_inference import inference_loop, flat_mcmc_samples
from src.utilities import AtomicUnits, elec_loss_fn
import src.inference.probability_models as prob

import warnings
warnings.filterwarnings('ignore')
```

# Aluminum

+++

# Get averge-atom collision frequencies and define the electronic state

```{code-cell} ipython3
AAdatafile = "../../data/external/Al-1 _vwf.txt"
# temperature
teV = 1
t = teV / AtomicUnits.energy

# density
d_ang = 0.18071 # 1/[angstroms]**3
d = d_ang * AtomicUnits.length**3

# ionization state (at T = 1 eV)
Z = 3

# read in data
AAdata = np.loadtxt(AAdatafile, skiprows=9, usecols=[0, 5, 6, 7, 8], unpack=True)

# T+ collision frequencies
Tpcollfreq = lambda x : np.interp(x, AAdata[0] / AtomicUnits.energy, AAdata[1] + 1j * AAdata[2])

# Kubo-Greenwood collision frequencies
KGcollfreq = lambda x : np.interp(x, AAdata[0] / AtomicUnits.energy, AAdata[3] + 1j * AAdata[4])

# electron data
electrons = ElectronGas(t, d, DOSratio=None, chemicalpot=None)

# dielectric function
dielectric = Mermin(electrons)

# Wavenumbers used in analysis
wavenum = [1.55, 0.78] # 1/[angstrom]
```

# Get TDDFT ELF data

```{code-cell} ipython3
tddftELFdata = np.loadtxt("../../data/processed/tddft_elf.txt", unpack=True)
freq_grid = tddftELFdata[0]
tddft_ELF = tddftELFdata[[3, 1]]
normalized_tddft_ELF = tddftELFdata[[4, 2]]
```

# Define collision frequency and ELF models

```{code-cell} ipython3
# define our collision frequency function
collisionfreq = BornLogPeak(electrons.temperature, electrons.density, electrons.chemicalpot, Z)

def elfmodel(wavenum, freq, params):
    return elec_loss_fn(
        dielectric,
        wavenum,
        freq,
        lambda x: collisionfreq(x, params)
    )
```

```{code-cell} ipython3
def plotmaskeddata(
    ax,
    x,
    y,
    mask=None,
    plotmaskrange=False,
    bglines=False,
    unmasked_kwargs={
        "color": "tab:orange",
        "lw": 1,
        "alpha": 0.1
    },
    masked_kwargs={
        "color": "tab:grey",
        "lw": 1,
        "alpha": 0.1
    },
    maskrangeline_kwargs={},
):
    """
    A way to plot data that has a mask that includes the masked (hidden) portion of the data.
    
    For a given set of data (x, y) and a mask over the data (len(mask) == len(x) == len(y)), plot the
    masked and unmasked portions of the data without any overlap between these sets.
    """
    if mask is None:
        mask = [True] * len(x)
    # Find edges of mask
    mask_ledge = (tmp := np.where(mask))[0][0] + 1 
    mask_redge = tmp[0][-1]

    # plot left portion of masked data
    ax.plot(x[:mask_ledge], y[:mask_ledge], **masked_kwargs)
    # plot unmasked portion of data
    ax.plot(x[mask], y[mask], **unmasked_kwargs, zorder=2)
    # plot white background lines
    if bglines:
        unmasked_kwargs["color"] = "white"
        unmasked_kwargs["alpha"] = 1
        ax.plot(x[mask], y[mask], **unmasked_kwargs, zorder=1)
    # plot right portion of masked data
    ax.plot(x[mask_redge:], y[mask_redge:], **masked_kwargs)
    if plotmaskrange:
        ax.axvline(x[mask_ledge], **maskrangeline_kwargs)
        ax.axvline(x[mask_redge], **maskrangeline_kwargs)
```

```{code-cell} ipython3
samplesfile = "../../data/mcmc/mcmc_tddft"

datasets = [ 
    'abs residual - q = 1.55 - 80% peak threshold',
    'abs residual - q = 0.78 - 80% peak threshold',
    'abs residual - q = [0.78 1.55] - 80% peak threshold',
]
```

```{code-cell} ipython3
# import h5py
# dataset = "abs residual - q = 1.55 - 80% peak threshold"
# with h5py.File("../../data/mcmc/mcmc_tddft",'a') as f_dest:
#     with h5py.File("../../data/mcmc/mcmc_tddft_singleq",'r') as f_src:
#             f_src.copy(f_src[dataset],f_dest)

with h5py.File("../../data/mcmc/mcmc_tddft",'r') as f:
    print(list(f.keys()))
#     for attr in f[dataset].attrs:
#         print(attr)
```

# Inference plots

```{code-cell} ipython3
fig, axs = plt.subplots(3, 2, layout="constrained", sharex="col", figsize=(6, 6))

# colors for different datasets
colors=["tab:orange", "C2", "C0"]

with h5py.File(samplesfile, "r") as f:
    for i, dset in enumerate(np.asarray(datasets)[[0, 1]]):
        backend = emcee.backends.HDFBackend(samplesfile, name=dset)
        flat_samples = flat_mcmc_samples(backend)
        # randomly pick 100 samples from our MCMC sampling data
        inds = rng.integers(len(flat_samples), size=10)
        # Get data mask
        if i==2:
            mask = f[dset].attrs["frequency grid masks (for each wavenumber)"]
        else:
            mask = f[dset].attrs["frequency grid mask"]
        # plot collision model + ELF for different parameters from MCMC sampling
        for ind in inds:
            # individual MCMC sample of collision model parameters
            sample = flat_samples[ind]
            # plot collision frequencies
            plotmaskeddata(
                axs[i, 0],
                freq_grid,
                collisionfreq.real(freq_grid / AtomicUnits.energy, sample),
                mask,
                unmasked_kwargs={"color": colors[i], "alpha": 0.2, "lw": 1},
                plotmaskrange=True,
                maskrangeline_kwargs={"color": "black", "lw": 0.5, "ls": (0, (1, 10))}
            )
            # plot ELFs
            plotmaskeddata(
                axs[i, 1],
                freq_grid,
                elfmodel(wavenum[i] * AtomicUnits.length, freq_grid / AtomicUnits.energy, sample),
                mask,
                unmasked_kwargs={"color": colors[i], "alpha": 0.2, "lw": 1},
                plotmaskrange=True,
                maskrangeline_kwargs={"color": "black", "lw": 0.5, "ls": (0, (1, 10))}
            )
        # # plot grid hatch
        # for ax in axs[i, :]:
        #     ax.fill_between(
        #         freq_grid,
        #         0,
        #         1, 
        #         where=mask,
        #         color="black",
        #         alpha=1, 
        #         transform=ax.get_xaxis_transform(),
        #         facecolor=("white", 0),
        #         linestyle="--",
        #         #hatch=".",
        #         zorder=-1
        #     )
    # last data set has special structure
    dset = datasets[-1]
    backend = emcee.backends.HDFBackend(samplesfile, name=dset)
    flat_samples = flat_mcmc_samples(backend)
    # randomly pick 100 samples from our MCMC sampling data
    inds = rng.integers(len(flat_samples), size=10)
    # Get data mask
    mask = f[dset].attrs["frequency grid masks (for each wavenumber)"]
    # plot collision model + ELF for different parameters from MCMC sampling
    for qind in range(len(wavenum)):
        for ind in inds:
            # individual MCMC sample of collision model parameters
            sample = flat_samples[ind]
            # plot collision frequencies
            plotmaskeddata(
                axs[2, 0],
                freq_grid,
                collisionfreq.real(freq_grid / AtomicUnits.energy, sample),
                mask[qind],
                unmasked_kwargs={"color": colors[2], "alpha": 0.1 + qind/10, "lw": 1},
                plotmaskrange=True,
                maskrangeline_kwargs={"color": "black", "lw": 0.5, "ls": (0, (1, 10))}
            )
            # plot ELFs
            # order of wavenums is different than MCMC data
            y = elfmodel(wavenum[1 - qind] * AtomicUnits.length, freq_grid / AtomicUnits.energy, sample)
            plotmaskeddata(
                axs[2, 1],
                freq_grid,
                y / np.max(y), # normalized ELF
                mask[qind],
                unmasked_kwargs={"color": colors[2], "alpha": 0.1, "lw": 1},
                plotmaskrange=True,
                maskrangeline_kwargs={"color": "black", "lw": 0.5, "ls": (0, (1, 10))}
            )
        # plot grid hatch
        # hatch = [".", "."]
        # for ax in axs[2, :]:
        #     ax.fill_between(
        #         freq_grid,
        #         0,
        #         1, 
        #         where=mask[qind],
        #         color="black",
        #         alpha=1, 
        #         transform=ax.get_xaxis_transform(),
        #         facecolor=("white", 0),
        #         #hatch=hatch[qind],
        #         zorder=-1
        #     )


for i in range(len(axs)):
    # sample label
    axs[i, 0].plot(0, 0, colors[i], alpha=0.4, lw=1, label="Sample")

    # plot true collision frequency and ELF
    if i == 0:
        label="TDDFT"
    else:
        label=None

    if i==2:
        # plot both ELFs on one plot and normalize them
        pass
        # axs[i, 2].plot(freq_grid, normalized_tddft_ELF.T, ls="--", color="black", lw=1.5)
    else:
        axs[i, 1].plot(freq_grid, tddft_ELF[i], ls="--", color="black", lw=1.5, label=label)

    # axis scales and limits
    axs[i, 0].set_xscale("log")
    # axs[i, 0].set_ylim(0, 0.3)
    
    # axs[i, 1].set_ylim(0, 0.99)

    # tick params
    axs[i, 1].tick_params(left=False, labelleft=False, right=True, labelright=True)

# axis lims
axs[0, 0].set_ylim(0, 1)
axs[2, 1].set_xlim(0, 40)

# labels and titles
axs[0, 0].set_title("Collision Freq. (at. u.)")
axs[0, 0].set_ylabel("(a) ", rotation="horizontal", horizontalalignment="right")
axs[0, 0].legend(frameon=False, loc="upper left")
axs[0, 1].set_title("ELF")
axs[1, 0].legend(frameon=False)
axs[1, 0].set_ylabel("(b) ", rotation="horizontal", horizontalalignment="right")
axs[2, 0].legend(frameon=False)
axs[2, 0].set_ylabel("(c) ", rotation="horizontal", horizontalalignment="right")
axs[2, 0].set_xlabel("Frequency (eV)")
axs[2, 1].set_xlabel("Frequency (eV)");

# plt.savefig("../../reports/figures/ideal_mcmc_res-datarange-changes")
```

```{code-cell} ipython3

```

```{code-cell} ipython3
import matplotlib.ticker as ticker
fig, axs = plt.subplots(3, 2, layout="constrained", sharex="col", figsize=(6, 6))
plt.rcParams['hatch.linewidth'] = 0.3

# colors for different datasets
colors=["tab:orange", "C2", "C0"]

with h5py.File(samplesfile, "r") as f:
    for i, dset in enumerate(np.asarray(datasets)[[0, 1]]):
        backend = emcee.backends.HDFBackend(samplesfile, name=dset)
        flat_samples = flat_mcmc_samples(backend)
        # randomly pick 100 samples from our MCMC sampling data
        inds = rng.integers(len(flat_samples), size=50)
        # Get data mask
        if i==2:
            mask = f[dset].attrs["frequency grid masks (for each wavenumber)"]
        else:
            mask = f[dset].attrs["frequency grid mask"]
        # plot collision model + ELF for different parameters from MCMC sampling
        for ind in inds:
            # individual MCMC sample of collision model parameters
            sample = flat_samples[ind]
            # plot collision frequencies
            plotmaskeddata(
                axs[i, 0],
                freq_grid,
                collisionfreq.real(freq_grid / AtomicUnits.energy, sample),
                mask,
                bglines=True,
                unmasked_kwargs={"color": "tab:orange", "alpha": 0.2, "lw": 1},
            )
            # plot ELFs
            plotmaskeddata(
                axs[i, 1],
                freq_grid,
                elfmodel(wavenum[i] * AtomicUnits.length, freq_grid / AtomicUnits.energy, sample),
                mask,
                bglines=True,
                unmasked_kwargs={"color": "tab:orange", "alpha": 0.2, "lw": 1},
            )
        # plot grid hatch
        for ax in axs[i, :]:
            ax.fill_between(
                freq_grid,
                0,
                1, 
                where=mask,
                color="C0",
                alpha=0.1, 
                transform=ax.get_xaxis_transform(),
                # facecolor="none",
                # hatch="/",
                zorder=-1
            )
    # last data set has special structure
    dset = datasets[-1]
    backend = emcee.backends.HDFBackend(samplesfile, name=dset)
    flat_samples = flat_mcmc_samples(backend)
    # randomly pick 100 samples from our MCMC sampling data
    inds = rng.integers(len(flat_samples), size=50)
    # Get data mask
    mask = f[dset].attrs["frequency grid masks (for each wavenumber)"]
    # plot collision model + ELF for different parameters from MCMC sampling
    hatch = ["/", "\\"]
    for qind in range(len(wavenum)):
        for ind in inds:
            # individual MCMC sample of collision model parameters
            sample = flat_samples[ind]
            # plot collision frequencies only once
            if qind == 1:
                plotmaskeddata(
                    axs[2, 0],
                    freq_grid,
                    collisionfreq.real(freq_grid / AtomicUnits.energy, sample),
                    mask[qind],
                    bglines=True,
                    unmasked_kwargs={"color": "tab:orange", "alpha": 0.1, "lw": 1},
                )
            # plot ELFs
            # order of wavenums is different than MCMC data
            y = elfmodel(wavenum[1 - qind] * AtomicUnits.length, freq_grid / AtomicUnits.energy, sample)
            plotmaskeddata(
                axs[2, 1],
                freq_grid,
                y / np.max(y), # normalized ELF
                mask[qind],
                bglines=True,
                unmasked_kwargs={"color": "tab:orange", "alpha": 0.1, "lw": 1},
            )
        # plot hatch
        for ax in axs[2, :]:
            ax.fill_between(
                freq_grid,
                0,
                1, 
                where=mask[1 - qind],
                color="C0",
                alpha=0.1, 
                transform=ax.get_xaxis_transform(),
                # facecolor="none",
                # hatch=hatch[qind],
                zorder=-1
            )

# sample label
    axs[0, 0].plot(0, 0, "tab:orange", alpha=0.4, lw=1, label="Sample")
    axs[0, 0].plot(0, 0, "black", lw=1.5, label="TDDFT", ls="--")
for i in range(len(axs)):
    # plot TDDFT ELF

    if i==2:
        # plot both ELFs on one plot and normalize them
        axs[2, 1].plot(freq_grid, normalized_tddft_ELF.T, ls="--", color="black", lw=1.5)
    else:
        axs[i, 1].plot(freq_grid, tddft_ELF[i], ls="--", color="black", lw=1.5)

    # axis scales and limits
    axs[i, 0].set_xscale("log")

    # move ticks
    axs[i, 1].tick_params(left=False, labelleft=False, right=True, labelright=True)

    # default number of digits
    axs[i, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# axis lims
for ax in axs.flatten():
    ax.set_ylim(0)
axs[0, 0].set_ylim(0, 1)
axs[2, 1].set_xlim(0, 40)

# labels and titles
axs[0, 0].set_title("Collision Freq. (at. u.)")
axs[0, 0].set_ylabel("(a) ", rotation="horizontal", horizontalalignment="right")
axs[0, 0].legend(frameon=False, loc="upper left")
axs[0, 1].set_title("ELF")
# axs[1, 0].legend(frameon=False)
axs[1, 0].set_ylabel("(b) ", rotation="horizontal", horizontalalignment="right")
# axs[2, 0].legend(frameon=False)
axs[2, 0].set_ylabel("(c) ", rotation="horizontal", horizontalalignment="right")
axs[2, 0].set_xlabel("Frequency (eV)")
axs[2, 1].set_xlabel("Frequency (eV)");

# wave numbers
axs[0, 1].text(1, 1, r"$q = 1.55$ A$^{-1}$", fontsize="small")
axs[1, 1].text(39, 3.7, r"$q = 0.78$ A$^{-1}$", fontsize="small", ha="right", va="top")

# ticks
axs[2, 0].set_xticks([1e-1, 1e1, 1e3])
# plt.savefig("../../reports/figures/tddft_inference")
```

## Aluminum, nonideal DOS

```{code-cell} ipython3
samplesfile = "../../data/mcmc/mcmc_tddft"

dataset = "abs residual - q = 1.55 - 80% peak threshold - nonideal dos"
```

```{code-cell} ipython3
# temperature
teV = 1
t = teV / AtomicUnits.energy

# density
d_ang = 0.18071 # 1/[angstroms]**3
d = d_ang * AtomicUnits.length**3

# ionization state (at T = 1 eV)
Z = 3

# chemical potential
cp = 0.3212 # [at u]

# read in data
AAdata = np.loadtxt("../../data/external/Al-1 _vwf.txt", skiprows=9, usecols=[0, 9], unpack=True)

# function for DOS ratio
dos_fn = lambda x : np.interp(x, np.sqrt(2 * AAdata[0] / AtomicUnits.energy), AAdata[1])


# electron data
nonideal_electrons = ElectronGas(t, d, dos_fn, cp)

# dielectric function
dielectric = Mermin(nonideal_electrons)

# wavenumber
wavenum = 1.55 # 1/A
```

```{code-cell} ipython3
freq_grid, tddft_ELF = np.loadtxt("../../data/processed/tddft_elf.txt", unpack=True, usecols=[0, 3])
```

```{code-cell} ipython3
# define our collision frequency function
collisionfreq = BornLogPeak(nonideal_electrons.temperature, nonideal_electrons.density, nonideal_electrons.chemicalpot, Z)

def elfmodel(freq, params):
    return elec_loss_fn(
        dielectric,
        wavenum * AtomicUnits.length,
        freq,
        lambda x: collisionfreq(x, params)
    )
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, layout="constrained", sharex="col", figsize=(5.5, 2.5))

with h5py.File(samplesfile, "r") as f:
    backend = emcee.backends.HDFBackend(samplesfile, name=dataset)
    flat_samples = flat_mcmc_samples(backend)
    # randomly pick 100 samples from our MCMC sampling data
    inds = rng.integers(len(flat_samples), size=50)
    # Get data mask
    mask = f[dataset].attrs["frequency grid mask"]
    # plot collision model + ELF for different parameters from MCMC sampling
    for ind in inds:
        # individual MCMC sample of collision model parameters
        sample = flat_samples[ind]
        # plot collision frequencies
        plotmaskeddata(
            axs[0],
            freq_grid,
            collisionfreq.real(freq_grid / AtomicUnits.energy, sample),
            mask,
            bglines=True,
            unmasked_kwargs={"color": "tab:orange", "alpha": 0.2, "lw": 1},
        )
        # plot ELFs
        plotmaskeddata(
            axs[1],
            freq_grid,
            elfmodel(
                freq_grid / AtomicUnits.energy,
                sample
            ),
            mask,
            bglines=True,
            unmasked_kwargs={"color": "tab:orange", "alpha": 0.2, "lw": 1},
        )
    # plot grid hatch
    for ax in axs:
        ax.fill_between(
            freq_grid,
            0,
            1, 
            where=mask,
            color="C0",
            alpha=0.1, 
            transform=ax.get_xaxis_transform(),
            zorder=-1
        )

# sample label
axs[0].plot(0, 0, "tab:orange", alpha=0.4, lw=1, label="Sample")
axs[0].plot(0, 0, "black", lw=1.5, label="TDDFT", ls="--")

# plot TDDFT ELF
axs[1].plot(freq_grid, tddft_ELF, ls="--", color="black", lw=1.5)

# axis scales and limits
axs[0].set_xscale("log")

# move ticks
axs[1].tick_params(left=False, labelleft=False, right=True, labelright=True)

# axis lims
for ax in axs:
    ax.set_ylim(0)
# axs[0, 0].set_ylim(0, 3)
axs[0].set_ylim(0, 0.72)
axs[1].set_xlim(0, 50)

# labels and titles
axs[0].set_title("Collision Freq. (at. u.)")
# axs[0, 0].set_ylabel("(a) ", rotation="horizontal", horizontalalignment="right")
axs[0].legend(frameon=False, loc="best", borderaxespad=0.1, fontsize=14)
axs[1].set_title("ELF")

axs[0].set_xlabel("Frequency (eV)")
axs[1].set_xlabel("Frequency (eV)");

# DOS
# axs[1].text(95, 0.82, r"Non-Ideal DOS", fontsize="small", ha="right", va="top")

# ticks
axs[0].set_xticks([1e-1, 1e1, 1e3])
# plt.savefig("../../reports/figures/tddft_Al_inference_nonideal")
```

# Iron

```{code-cell} ipython3
samplesfile = "../../data/mcmc/mcmc_tddft_iron"

datasets = [
    "abs residual - q = 1.1 - 80% peak threshold (ff peak only)",
    "abs residual - q = 1.1 - 80% peak threshold (ff peak only) - nonideal dos"
]
```

```{code-cell} ipython3
# temperature
teV = 1
t = teV / AtomicUnits.energy

# density
d = 0.1 # au

# ionization state (at T = 1 eV)
Z = 8

# chemical potential
cp = 0.383 # au

# read in data
AAdata = np.loadtxt("../../data/external/Fe-1eV-dos.txt", skiprows=2, usecols=[0, 1, 2], unpack=True)

# function for DOS ratio
dos_fn = lambda x : np.interp(x, np.sqrt(2 * AAdata[0]), AAdata[1] / AAdata[2])

# electron data
nonideal_electrons = ElectronGas(t, d, dos_fn, cp)
ideal_electrons = ElectronGas(t, d)

# # dielectric function
# dielectric = Mermin(electrons)

# wavenumber
wavenum = 1.1 # 1/A
```

```{code-cell} ipython3
# define our collision frequency function
collisionfreq = [
    BornLogPeak(ideal_electrons.temperature, ideal_electrons.density, ideal_electrons.chemicalpot, Z),
    BornLogPeak(nonideal_electrons.temperature, nonideal_electrons.density, nonideal_electrons.chemicalpot, Z)
]
electrons = [
    ideal_electrons,
    nonideal_electrons
]

def elfmodel(wavenum, freq, params, electrons, collfreq):
    return elec_loss_fn(
        Mermin(electrons),
        wavenum,
        freq,
        lambda x: collfreq(x, params)
    )
```

```{code-cell} ipython3
freq_grid, tddft_ELF  = np.loadtxt("../../data/processed/tddft_iron_elf.txt", unpack=True)
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, layout="constrained", sharex="col", figsize=(6, 4))

with h5py.File(samplesfile, "r") as f:
    for i, dset in enumerate(datasets):
        backend = emcee.backends.HDFBackend(samplesfile, name=dset)
        flat_samples = flat_mcmc_samples(backend)
        # randomly pick 100 samples from our MCMC sampling data
        inds = rng.integers(len(flat_samples), size=50)
        # Get data mask
        mask = f[dset].attrs["frequency grid mask"]
        # plot collision model + ELF for different parameters from MCMC sampling
        for ind in inds:
            # individual MCMC sample of collision model parameters
            sample = flat_samples[ind]
            # plot collision frequencies
            plotmaskeddata(
                axs[i, 0],
                freq_grid,
                collisionfreq[i].real(freq_grid / AtomicUnits.energy, sample),
                mask,
                bglines=True,
                unmasked_kwargs={"color": "tab:orange", "alpha": 0.2, "lw": 1},
            )
            # plot ELFs
            plotmaskeddata(
                axs[i, 1],
                freq_grid,
                elfmodel(
                    wavenum * AtomicUnits.length,
                    freq_grid / AtomicUnits.energy,
                    sample,
                    electrons[i],
                    collisionfreq[i],
                ),
                mask,
                bglines=True,
                unmasked_kwargs={"color": "tab:orange", "alpha": 0.2, "lw": 1},
            )
        # plot grid hatch
        for ax in axs[i, :]:
            ax.fill_between(
                freq_grid,
                0,
                1, 
                where=mask,
                color="C0",
                alpha=0.1, 
                transform=ax.get_xaxis_transform(),
                zorder=-1
            )

# sample label
    axs[0, 0].plot(0, 0, "tab:orange", alpha=0.4, lw=1, label="Sample")
    axs[0, 0].plot(0, 0, "black", lw=1.5, label="TDDFT", ls="--")
for i in range(len(axs)):
    # plot TDDFT ELF
    axs[i, 1].plot(freq_grid, tddft_ELF, ls="--", color="black", lw=1.5)

    # axis scales and limits
    axs[i, 0].set_xscale("log")

    # move ticks
    axs[i, 1].tick_params(left=False, labelleft=False, right=True, labelright=True)

# axis lims
for ax in axs.flatten():
    ax.set_ylim(0)
axs[0, 0].set_ylim(0, 3.5)
axs[1, 0].set_ylim(0, 3.5)
axs[1, 1].set_xlim(0, 100)

# labels and titles
axs[0, 0].set_title("Collision Freq. (at. u.)")
axs[0, 0].set_ylabel("(a) ", rotation="horizontal", horizontalalignment="right")
axs[0, 0].legend(frameon=False, loc="best", borderaxespad=0.1, fontsize=14)
axs[0, 1].set_title("ELF")
# axs[0, 1].legend(frameon=False)
axs[1, 0].set_ylabel("(b) ", rotation="horizontal", horizontalalignment="right")
axs[1, 0].set_xlabel("Frequency (eV)")
axs[1, 1].set_xlabel("Frequency (eV)");

# DOS
axs[0, 1].text(95, 0.75, r"Ideal DOS", fontsize="small", ha="right")
axs[1, 1].text(95, 0.82, r"Non-Ideal DOS", fontsize="small", ha="right", va="top")

# ticks
axs[1, 0].set_xticks([1e-1, 1e1, 1e3])
# axs[0, 0].set_yticks([0, 1, 2, 3])
# axs[1, 0].set_yticks([0, 1, 2, 3])
# plt.savefig("../../reports/figures/tddft_iron_inference")
```

```{code-cell} ipython3

```
