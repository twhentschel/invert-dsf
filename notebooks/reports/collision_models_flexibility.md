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
import matplotlib.pyplot as plt
plt.style.use(["seaborn-v0_8-paper", "paper-style.mplstyle"])
from scipy.optimize import curve_fit

from src.inference import collision_models as models
from src.utilities import AtomicUnits, elec_loss_fn
from uegdielectric import ElectronGas, dielectric
```

```{code-cell} ipython3
datafile = "../../data/external/Al-1 _vwf.txt"

data = np.loadtxt(datafile, skiprows=9, usecols=[0, 1, 3, 5, 9], unpack=True)
dos = data[-1]
# remove dos info from data matrix
data = data[:-1]
print(f"shape of data = {data.shape}")
```

```{code-cell} ipython3
# electron information
# temperature
teV = 1
t = teV / AtomicUnits.energy

# density
d_ang = 0.18071 # angstroms
d = d_ang * AtomicUnits.length**3

# chemical potential
m = 0.3212

# wavenumber
q_invang = 1.55 # 1 / angstroms
q = q_invang * AtomicUnits.length

# density of states ratio function
dos_fn = lambda x : np.interp(x, np.sqrt(2 * data[0] / AtomicUnits.energy), dos)
```

```{code-cell} ipython3
fig, ax = plt.subplots(2, 1, figsize=(6,8))
# collision frequency theories
collnames = ["Born", "T-matrix", "T-matrix + inel. collisions"]
# plot the real part only and our fitted model first
for i, colldata in enumerate(data[1:]):
    # fit the model to the data
    # Use the relative residual in the fit: (ydata - model) / ydata
    popt, _ = curve_fit(models.collision_activate_decay, data[0] / AtomicUnits.energy, colldata, p0=(1,1,1,1), sigma=colldata, bounds=(0, np.inf))
    # plot the data
    p = ax[0].plot(data[0], colldata, label=f"{collnames[i]}")
    # plot the model with optimized parameters
    ax[0].plot(
        data[0], 
        models.collision_activate_decay(data[0] / AtomicUnits.energy, *popt), 
        ls="--", 
        color=p[-1].get_color(),
        alpha=0.8
    )

    # plot the ELF
    if i == 0:
        mermin = dielectric.Mermin(ElectronGas(t, d))
        rpaelf = elec_loss_fn(
            mermin, 
            q, 
            data[0] / AtomicUnits.energy
        )
        ax[1].plot(data[0], rpaelf, color="gray", ls="-.", label="RPA")
    else:
        mermin = dielectric.Mermin(ElectronGas(t, d, dos_fn, m))
    elf = elec_loss_fn(
        mermin, 
        q, 
        data[0] / AtomicUnits.energy, 
        lambda x : np.interp(x, data[0] / AtomicUnits.energy, colldata)
    )
    ax[1].plot(
        data[0],
        elf
    )

ax[0].set_ylim(2e-3, 5e-1)
ax[0].set_xlim(data[0, 0], data[0, -1])
ax[0].legend(frameon=False)
ax[0].set_ylabel("Collision frequency (at. u.)")
ax[0].set_xscale("log")
ax[0].set_yscale("log")

ax[1].set_xlim(0, 50)
ax[1].set_ylabel("ELF (at. u.)")
ax[1].set_xlabel(r"$\hbar \omega$ (eV)")

plt.tight_layout()
# plt.savefig("../../reports/figures/model-fit-AA-collisions")
```

```{code-cell} ipython3

```
