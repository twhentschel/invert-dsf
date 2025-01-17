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

# Dynamic conductivities from collision frequencies inferred from TD-DFT DSF/ELF data
The dynamic conductivity can be expressed in terms of the dielectric function
$$ \sigma (\omega) = \frac{\omega}{4 \pi} \mathrm{Im}\{ \epsilon(q=0, \omega)\}$$

We can use the Mermin dielectric function in this formula along with the collision frequency model and the MCMC samples of the collision frequency model parameters.

```{code-cell} ipython3
import numpy as np
rng = np.random.default_rng(seed=41)
import matplotlib.pyplot as plt
plt.style.use(["seaborn-v0_8-paper", "paper-style.mplstyle"])
import seaborn as sns

import emcee 
from uegdielectric import ElectronGas
from uegdielectric.dielectric import Mermin

from src.inference.collision_models import BornLogPeak
from src.inference.mcmc_inference import flat_mcmc_samples
from src.utilities import AtomicUnits

import warnings
warnings.filterwarnings('ignore')
```

```{code-cell} ipython3
samplesfile = "../../data/mcmc/mcmc_tddft"
dataset = "abs residual - q = [0.78 1.55] - 80% peak threshold - TDDFT peak normalized"

backend = emcee.backends.HDFBackend(samplesfile, name=dataset)
flat_samples = flat_mcmc_samples(backend)
```

```{code-cell} ipython3
# temperature
teV = 1
t = teV / AtomicUnits.energy

# density
d_ang = 0.18071 # 1/[angstroms]**3
d = d_ang * AtomicUnits.length**3

# # chemical potential
# cp = 0.3212 # [at u]

# ionization state (at T = 1 eV)
Z = 3

# electron data
electrons = ElectronGas(t, d)

# dielectric function
dielectric = Mermin(electrons)
```

```{code-cell} ipython3
# define our collision frequency function
collisionfreq = BornLogPeak(electrons.temperature, electrons.density, electrons.chemicalpot, Z)
```

```{code-cell} ipython3
def conductivity(freq, params, wavenum=0):
    return (
        freq
        / 4
        / np.pi
        * dielectric(wavenum, freq, lambda x: collisionfreq(x, params)).imag
    )
```

```{code-cell} ipython3
x = np.geomspace(1e-1, 1e3, 200)
N = 50
inds = rng.integers(len(flat_samples), size=N)
conductivities = [conductivity(x / AtomicUnits.energy, sample, 1e-2) for sample in flat_samples[inds]]
conductivities.sort(key=lambda arr : arr[0])
with sns.color_palette("crest", n_colors=N):
    for y in conductivities:
        plt.loglog(x, y, alpha=.3)

# plot AA conductivity
x, AAconductivity = np.loadtxt("../../data/external/KG for collision paper.txt", unpack=True, skiprows=1)
plt.plot(x, AAconductivity, color="purple", ls="--")

plt.xlabel(r"Frequency (eV)")
plt.ylabel("Conductivity (at. u.)")

plt.ylim(1e-5, 1.2e0)
plt.xlim(1e-1, 8e2)
plt.tight_layout()
plt.savefig("../../reports/figures/dynamic_conductivity_inference_correct_normalization.tif")
```

```{code-cell} ipython3
def dcconductivity(density, params):
    return density / collisionfreq.real(0, params)

dccond_samples = dcconductivity(electrons.density, flat_samples.T)
mean = np.mean(dccond_samples)
stddev = np.std(dccond_samples)
print(f"mean(DC conductivity) = {mean} [au]")
print(f"Std. Dev.(DC conductivity) = {stddev} [au]")
cond_au2SI = 4599848.13 # S/m
print(f"mean(DC conductivity) = {mean * cond_au2SI:e} [S/m]")
print(f"Std. Dev.(DC conductivity) = {stddev  * cond_au2SI:e} [S/m]")
print(f" min = {np.min(dccond_samples) * cond_au2SI:e}, max = {np.max(dccond_samples) * cond_au2SI:e}")
print("DC collision freq spread")
DCcoll = collisionfreq.real(1e-5, flat_samples.T)
print(f" min = {np.min(DCcoll)}; max = {np.max(DCcoll)}")
```

## DC conductivities

### $q = 1.55$ 1/A

```{code-cell} ipython3
samplesfile = "../../data/mcmc/mcmc_tddft"
dataset = "abs residual - q = 1.55 - 80% peak threshold"

backend = emcee.backends.HDFBackend(samplesfile, name=dataset)
flat_samples = flat_mcmc_samples(backend)
```

```{code-cell} ipython3
x = np.geomspace(1e-1, 1e3, 200)
N = 50
inds = rng.integers(len(flat_samples), size=N)
conductivities = [conductivity(x / AtomicUnits.energy, sample, 1e-2) for sample in flat_samples[inds]]
conductivities.sort(key=lambda arr : arr[0])
with sns.color_palette("crest", n_colors=N):
    for y in conductivities:
        plt.loglog(x, y, alpha=.3)

plt.xlabel(r"Frequency (eV)")
plt.ylabel("Conductivity (at. u.)")

plt.ylim(1e-5)
plt.xlim(1e-1, 8e2)
plt.tight_layout()
```

```{code-cell} ipython3
dccond_samples = [c[0] for c in conductivities]
mean = np.mean(dccond_samples)
stddev = np.std(dccond_samples)
print(f"mean(DC conductivity) = {mean} [au]")
print(f"Std. Dev.(DC conductivity) = {stddev} [au]")
```

Compare with alternative DC conductivity formulas (See NB 6.0)

```{code-cell} ipython3
dccond_samples = dcconductivity(electrons.density, flat_samples.T)
mean = np.mean(dccond_samples)
stddev = np.std(dccond_samples)
print(f"mean(DC conductivity) = {mean} [au]")
print(f"Std. Dev.(DC conductivity) = {stddev} [au]")
cond_au2SI = 4599848.13 # S/m
print(f"mean(DC conductivity) = {mean * cond_au2SI:e} [S/m]")
print(f"Std. Dev.(DC conductivity) = {stddev  * cond_au2SI:e} [S/m]")
print(f" min = {np.min(dccond_samples) * cond_au2SI:e}, max = {np.max(dccond_samples) * cond_au2SI:e}")
print("DC collision freq spread")
DCcoll = collisionfreq.real(1e-6, flat_samples.T)
print(f" min = {np.min(DCcoll)}; max = {np.max(DCcoll)}")
```

### $q = 0.78$ 1/A

```{code-cell} ipython3
samplesfile = "../../data/mcmc/mcmc_tddft"
dataset = "abs residual - q = 0.78 - 80% peak threshold"

backend = emcee.backends.HDFBackend(samplesfile, name=dataset)
flat_samples = flat_mcmc_samples(backend)
```

```{code-cell} ipython3
x = np.geomspace(1e-3, 1e3, 100)
N = 50
inds = rng.integers(len(flat_samples), size=N)
conductivities = [conductivity(x / AtomicUnits.energy, sample, 1e-4) for sample in flat_samples[inds]]
conductivities.sort(key=lambda arr : arr[0])
with sns.color_palette("crest", n_colors=N):
    for y in conductivities:
        plt.loglog(x, y, alpha=.3)

plt.xlabel(r"Frequency (eV)")
plt.ylabel("Conductivity (at. u.)")

plt.ylim(1e-5)
plt.xlim(1e-3, 8e2)
plt.tight_layout()
```

```{code-cell} ipython3
N = 50
inds = rng.integers(len(flat_samples), size=N)
dccond_samples = [conductivity(np.asarray([1e-4]), sample, 1e-3)[0] for sample in flat_samples[inds]]
dccond_samples = [c[0] for c in conductivities]
mean = np.mean(dccond_samples)
stddev = np.std(dccond_samples)
print(f"mean(DC conductivity) = {mean} [au]")
print(f"Std. Dev.(DC conductivity) = {stddev} [au]")
```

Compare with alternative DC conductivity formulas (See NB 6.0)

```{code-cell} ipython3
dccond_samples = dcconductivity(electrons.density, flat_samples.T)
mean = np.mean(dccond_samples)
stddev = np.std(dccond_samples)
print(f"mean(DC conductivity) = {mean} [au]")
print(f"Std. Dev.(DC conductivity) = {stddev} [au]")
cond_au2SI = 4599848.13 # S/m
print(f"mean(DC conductivity) = {mean * cond_au2SI:e} [S/m]")
print(f"Std. Dev.(DC conductivity) = {stddev  * cond_au2SI:e} [S/m]")
print(f" min = {np.min(dccond_samples) * cond_au2SI:e}, max = {np.max(dccond_samples) * cond_au2SI:e}")
print("DC collision freq spread")
DCcoll = collisionfreq.real(1e-5, flat_samples.T)
print(f" min = {np.min(DCcoll)}; max = {np.max(DCcoll)}")
```

```{code-cell} ipython3
freq = np.geomspace(1e-1, 1e3, 100)
N = 100
for i in range(N):
    plt.loglog(freq, collisionfreq.real(freq/ AtomicUnits.energy, flat_samples[i]), alpha=0.1, c="orange")
# plot mean and standard deviation
mean = np.zeros_like(freq)
stddev = np.zeros_like(freq)
for i in range(len(freq)):
    mean[i] = np.mean(collisionfreq.real(freq[i]/ AtomicUnits.energy, flat_samples.T))
    stddev[i] = np.std(collisionfreq.real(freq[i]/ AtomicUnits.energy, flat_samples.T))
plt.plot(freq, mean, c="k", ls = "--", lw=0.5, label="mean")
plt.plot(freq, mean + stddev, c="k", ls="-", lw=0.5, label="max(0, mean +/-std. dev.)")
plt.plot(freq, np.maximum(1e-5, mean - stddev), c="k", ls="-", lw=0.5)
```

```{code-cell} ipython3

```
