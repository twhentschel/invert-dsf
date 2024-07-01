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
rng = np.random.default_rng()
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
dataset = "abs residual - q = [0.78 1.55] - 80% peak threshold"

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

# chemical potential
cp = 0.3212 # [at u]

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
        plt.loglog(x, y, alpha=.1)

plt.xlabel(r"Frequency (eV)")
plt.ylabel("Conductivity (at. u.)")

plt.ylim(1e-5)
plt.xlim(1e-1, 8e2)
plt.tight_layout()
# plt.savefig("../../reports/figures/dynamic_conductivity_inference")
```

```{code-cell} ipython3

```
