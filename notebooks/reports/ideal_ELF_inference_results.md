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

# Define the electronic state

```{code-cell} ipython3
# temperature
teV = 1
t = teV / AtomicUnits.energy

# density
d_ang = 0.18071 # 1/[angstroms]**3
d = d_ang * AtomicUnits.length**3

# ionization state (at T = 1 eV)
Z = 3

# electron data
electrons = ElectronGas(t, d, DOSratio=None, chemicalpot=None)

# dielectric function
dielectric = Mermin(electrons)

# Wavenumbers used in analysis
wavenum = 1.55 # 1/[angstrom]
```

# Get ideal ELF data

```{code-cell} ipython3
idealELFdata = np.loadtxt("../../data/raw/mermin_ELF_collision_model_fit_to_AA.txt", unpack=True)
freq_grid = idealELFdata[0]
truecollmodel = idealELFdata[1] + 1j * idealELFdata[2]
trueELF = idealELFdata[3]
```

# Define collision frequency and ELF models

```{code-cell} ipython3
# define our collision frequency function
collisionfreq = BornLogPeak(electrons.temperature, electrons.density, electrons.chemicalpot, Z)

def elfmodel(freq, params):
    return elec_loss_fn(
        dielectric,
        wavenum * AtomicUnits.length,
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
    }
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
    # plot white lines in the background
    if bglines:
        unmasked_kwargs["color"] = "white"
        unmasked_kwargs["alpha"] = 1
        ax.plot(x[mask], y[mask], **unmasked_kwargs, zorder=1)
    # plot right portion of masked data
    ax.plot(x[mask_redge:], y[mask_redge:], **masked_kwargs)
```

```{code-cell} ipython3
def adjust_lightness(color, amount=0.5):
    """https://stackoverflow.com/a/49601444/22218756"""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
```

```{code-cell} ipython3
samplesfile = "../../data/mcmc/mcmc_modeldata_bornlogpeak"

datasets = [
    "rel residual - q = 1.55 - full freq grid",
    "rel residual - q = 1.55 - 99% peak threshold",
    "rel residual - q = 1.55 - 80% peak threshold",
    "abs residual - q = 1.55 - 99% peak threshold",
    "abs residual - q = 1.55 - 80% peak threshold"
]
```

# Relative residual, full frequency grid

```{code-cell} ipython3
fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
# inset axes
x1, x2, y1, y2 = 5e-1, 45, 1e-1, 1
axins = ax[1].inset_axes(
    [0.15, 0.09, 0.45, 0.45],
    xlim=(x1, x2), ylim=(y1, y2)
)
# make inset axes tick labels smaller
axins.tick_params(labelsize=10)
axins.xaxis.set_major_locator(plt.MaxNLocator(2))
axins.yaxis.set_major_locator(plt.MaxNLocator(2))

backend = emcee.backends.HDFBackend(samplesfile, name=datasets[0])
fullgrid_flat_samples = flat_mcmc_samples(backend)

# randomly pick 100 samples from our MCMC sampling data
inds = rng.integers(len(fullgrid_flat_samples), size=100)

# plot collision function for different parameters from MCMC sampling
for ind in inds:
    sample = fullgrid_flat_samples[ind]
    ax[0].plot(
        freq_grid, 
        collisionfreq(freq_grid / AtomicUnits.energy, sample).real, 
        "tab:orange", 
        alpha=0.1,
        lw=1
    )

    # ELF
    elf = elfmodel(freq_grid / AtomicUnits.energy, sample)
    ax[1].plot(
        freq_grid,
        elf, 
        "tab:orange", 
        alpha=0.1,
        lw=1
    )
    axins.plot(
        freq_grid,
        elf,
        "tab:orange",
        alpha=0.1,
        lw=1
    )

# sample label plot
ax[0].plot(0, 0, "tab:orange", alpha=0.4, lw=1, label="Posterior Sample")

# inputs
ax[0].semilogx(freq_grid, truecollmodel.real, ls="--", color="black", label="True")
ax[1].plot(freq_grid, trueELF, ls="--", color="black")
axins.plot(freq_grid, trueELF, ls="--", color="black")

ax[0].set_xscale("log")
ax[0].set_ylabel("Collision Frequency (at. u.)")
ax[1].set_ylabel("ELF")
ax[0].legend()
ax[1].set_xlabel("Frequency (eV)")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].indicate_inset_zoom(axins, edgecolor="black", lw=1)

plt.tight_layout()
# plt.savefig("../../reports/figures/ideal_mcmc_relres_fullgrid")
```

```{code-cell} ipython3
fig, ax = plt.subplots(2, 1, figsize=(6, 8))
# inset axes
x1, x2, y1, y2 = 5e-1, 45, 1e-1, 1
axins = ax[1].inset_axes(
    [0.15, 0.09, 0.45, 0.45],
    xlim=(x1, x2), ylim=(y1, y2)
)
# make inset axes tick labels smaller
axins.tick_params(labelsize=10)
axins.xaxis.set_major_locator(plt.MaxNLocator(2))
axins.yaxis.set_major_locator(plt.MaxNLocator(2))

backend = emcee.backends.HDFBackend(samplesfile, name=datasets[0])
fullgrid_flat_samples = flat_mcmc_samples(backend)

# randomly pick N samples from our MCMC sampling data
N = 50
inds = rng.integers(len(fullgrid_flat_samples), size=N)
colls = [collisionfreq.real(freq_grid / AtomicUnits.energy, sample)[0] for sample in fullgrid_flat_samples[inds]]

# sort based on DC limit
sortedinds = np.argsort(colls)
# plot collision function for different parameters from MCMC sampling
colors = plt.cm.Spectral(np.linspace(0, 1, N))
for i, ind in enumerate(inds[sortedinds]):
    sample = fullgrid_flat_samples[ind]
    p = ax[0].plot(
        freq_grid, 
        collisionfreq.real(freq_grid / AtomicUnits.energy, sample),
        alpha=0.5,
        lw=1,
        color=colors[i]
    )

    # ELF
    elf = elfmodel(freq_grid / AtomicUnits.energy, sample)
    ax[1].plot(
        freq_grid,
        elf,
        color=p[-1].get_color(),
        alpha=0.5,
        lw=1
    )
    axins.plot(
        freq_grid,
        elf,
        color=p[-1].get_color(),
        alpha=0.5,
        lw=1
    )

# sample label plot
# ax[0].plot(0, 0, "tab:orange", alpha=0.4, lw=1, label="Posterior Sample")

# inputs
ax[0].semilogx(freq_grid, truecollmodel.real, ls="--", color="black", label="True")
ax[1].plot(freq_grid, trueELF, ls="--", color="black")
axins.plot(freq_grid, trueELF, ls="--", color="black")

ax[0].set_xscale("log")
ax[0].set_ylabel("Collision Frequency (at. u.)")
ax[1].set_ylabel("ELF")
ax[0].legend()
ax[1].set_xlabel("Frequency (eV)")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].indicate_inset_zoom(axins, edgecolor="black", lw=1)
plt.savefig("../../reports/figures/ideal_mcmc_relres_fullgrid_rainbow")
```

# Changing amount of data considered and residual type

```{code-cell} ipython3
fig, axs = plt.subplots(3, 2, layout="constrained", sharex="col", figsize=(6, 6))
plt.rcParams['hatch.linewidth'] = 0.3

# colors for different datasets
colors=["tab:orange", "C2", "C0"]

with h5py.File(samplesfile, "a") as f:
    for i, dset in enumerate(np.asarray(datasets)[[1, 2, 4]]):
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
                collisionfreq(freq_grid / AtomicUnits.energy, sample).real,
                mask,
                bglines=True,
                unmasked_kwargs={"color": "tab:orange", "alpha": 0.2, "lw": 1}
            )
            # plot ELFs
            plotmaskeddata(
                axs[i, 1],
                freq_grid,
                elfmodel(freq_grid / AtomicUnits.energy, sample),
                mask,
                bglines=True,
                unmasked_kwargs={"color": "tab:orange", "alpha": 0.2, "lw": 1}
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
# note: a transparent line for the label looks funky - adjust lightness but not transparency instead
axs[0, 0].plot(0, 0, color=adjust_lightness("tab:orange", 1.6), lw=1, label="Sample")

for i in range(len(axs)):
    # plot true collision frequency and ELF
    if i == 0:
        label="True"
    else:
        label=None
    axs[i, 0].plot(freq_grid, truecollmodel.real, ls="--", color="black", label=label, lw=1.5)
    axs[i, 1].plot(freq_grid, trueELF, ls="--", color="black", lw=1.5)

    # axis scales and limits
    axs[i, 0].set_xscale("log")
    axs[i, 0].set_ylim(0, 0.5)
    axs[i, 1].set_xlim(0, 70)
    axs[i, 1].set_ylim(0, 1)

    # tick params
    axs[i, 1].tick_params(left=False, labelleft=False, right=True, labelright=True)

# labels and titles
axs[0, 0].set_title("Collision Freq. (at. u.)")
axs[0, 0].set_ylabel("(a) ", rotation="horizontal", horizontalalignment="right")
axs[0, 0].legend(frameon=False)
axs[0, 1].set_title("ELF")
# axs[1, 0].legend(frameon=False)
axs[1, 0].set_ylabel("(b) ", rotation="horizontal", horizontalalignment="right")
# axs[2, 0].legend(frameon=False)
axs[2, 0].set_ylabel("(c) ", rotation="horizontal", horizontalalignment="right")
axs[2, 0].set_xlabel("Frequency (eV)")
axs[2, 1].set_xlabel("Frequency (eV)");

# ticks
axs[2, 0].set_xticks([1e-1, 1e1, 1e3])

# plt.savefig("../../reports/figures/ideal_mcmc_res-datarange-changes")
```

```{code-cell} ipython3
fig, axs = plt.subplots(3, 2, layout="constrained", sharex="col", figsize=(6, 6))
plt.rcParams['hatch.linewidth'] = 0.3

# colors for different datasets
colors=["tab:orange", "C2", "C0"]

with h5py.File(samplesfile, "a") as f:
    for i, dset in enumerate(np.asarray(datasets)[[1, 2, 4]]):
        backend = emcee.backends.HDFBackend(samplesfile, name=dset)
        flat_samples = flat_mcmc_samples(backend)
        # Get data mask
        mask = f[dset].attrs["frequency grid mask"]
        # randomly pick N samples from our MCMC sampling data
        N = 50
        inds = rng.integers(len(flat_samples), size=N)
        colls = [collisionfreq.real(freq_grid / AtomicUnits.energy, sample)[0] for sample in flat_samples[inds]]

        # sort based on DC limit
        sortedinds = np.argsort(colls)
        # plot collision function for different parameters from MCMC sampling
        colors = plt.cm.Spectral(np.linspace(0, 1, N))
        for j, ind in enumerate(inds[sortedinds]):
        # # plot collision model + ELF for different parameters from MCMC sampling
        # for ind in inds:
            # individual MCMC sample of collision model parameters
            sample = flat_samples[ind]
            # plot collision frequencies
            plotmaskeddata(
                axs[i, 0],
                freq_grid,
                collisionfreq(freq_grid / AtomicUnits.energy, sample).real,
                mask,
                unmasked_kwargs={"color": colors[j], "alpha": 0.2, "lw": 1}
            )
            # plot ELFs
            plotmaskeddata(
                axs[i, 1],
                freq_grid,
                elfmodel(freq_grid / AtomicUnits.energy, sample),
                mask,
                unmasked_kwargs={"color": colors[j], "alpha": 0.2, "lw": 1}
            )
        # plot grid hatch
        for ax in axs[i, :]:
            ax.fill_between(
                freq_grid,
                0,
                1, 
                where=mask,
                color="black",
                alpha=0.5, 
                transform=ax.get_xaxis_transform(),
                facecolor=("white", 0),
                hatch="/",
                zorder=-1
            )


# figure details
for i in range(len(axs)):
    # sample label
    # axs[i, 0].plot(0, 0, colors[i], alpha=0.4, lw=1, label="Sample")

    # plot true collision frequency and ELF
    if i == 0:
        label="True"
    else:
        label=None
    axs[i, 0].plot(freq_grid, truecollmodel.real, ls="--", color="black", label=label, lw=1.5)
    axs[i, 1].plot(freq_grid, trueELF, ls="--", color="black", lw=1.5)

    # axis scales and limits
    axs[i, 0].set_xscale("log")
    axs[i, 0].set_ylim(0, 0.5)
    axs[i, 1].set_xlim(0, 70)
    axs[i, 1].set_ylim(0, 0.99)

    # tick params
    axs[i, 1].tick_params(left=False, labelleft=False, right=True, labelright=True)

# labels and titles
axs[0, 0].set_title("Collision Freq. (at. u.)")
axs[0, 0].set_ylabel("(a) ", rotation="horizontal", horizontalalignment="right")
axs[0, 0].legend(frameon=False)
axs[0, 1].set_title("ELF")
axs[1, 0].legend(frameon=False)
axs[1, 0].set_ylabel("(b) ", rotation="horizontal", horizontalalignment="right")
axs[2, 0].legend(frameon=False)
axs[2, 0].set_ylabel("(c) ", rotation="horizontal", horizontalalignment="right")
axs[2, 0].set_xlabel("Frequency (eV)")
axs[2, 1].set_xlabel("Frequency (eV)");

# plt.savefig("../../reports/figures/ideal_mcmc_res-datarange-changes_rainbow")
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
