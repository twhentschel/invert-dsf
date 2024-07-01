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
from scipy import optimize

from src.inference.collision_models import BornLogPeak
from src.utilities import AtomicUnits, elec_loss_fn, kramerskronig
from uegdielectric import ElectronGas, dielectric
```

```{code-cell} ipython3
datafile = "../../data/external/Al-1 _vwf.txt"

# last column of data is not needed for this
data = np.loadtxt(datafile, skiprows=9, usecols=[0, 3, 4, 5, 6, 7, 8], unpack=True)
# limit the frequency grid to be between 0.1 and 500 eV
freqev = data[0]
gridtrunc = (0.1 < freqev) & (freqev < 1e3)
data = data[:, gridtrunc]
data = data[:, ::2]
freqev = data[0]
# convert grid to au
freq = freqev / AtomicUnits.energy
# compute imaginary part from real part using kramers-kronig
# data[2] = kramerskronig(freq, lambda x: np.interp(x, freq, data[1]))
# data[4] = kramerskronig(freq, lambda x: np.interp(x, freq, data[3]))
# data[6] = kramerskronig(freq, lambda x: np.interp(x, freq, data[5]))
# extranct collision frequencies
Tmat = data[1] + 1j * data[2]
Tplus = data[3] + 1j * data[4]
KG = data[5] + 1j * data[6]

print(f"shape of data = {KG.shape}")
```

```{code-cell} ipython3
# temperature
teV = 1
t = teV / AtomicUnits.energy

# density
d_ang = 0.18071 # 1/[angstroms]**3
d = d_ang * AtomicUnits.length**3

# charge state
Z = 3

electrons = ElectronGas(t, d)
print(f"inverse screening length = {models.inverse_screening_length(electrons.temperature, electrons.density)}")
print(f"chemical potential = {electrons.chemicalpot}")

wavenum = 1.55 # 1/[angstrom]
```

# Create Born collision freqeuncy data

```{code-cell} ipython3
born = models.ScreenedBorn(electrons.temperature, electrons.density, electrons.chemicalpot, Z)(freq)
```

# Define collision frequency model

```{code-cell} ipython3
collisionfreq = models.BornLogPeak(electrons.temperature, electrons.density, electrons.chemicalpot, Z)
# write real part in form acceptable by scipy.optimize.curve_fit
def model(x, p0, p1, p2, p3, p4):
    return collisionfreq.real(x, (p0, p1, p2, p3, p4))
```

# Fit model to data and also plot corresponding ELFs

```{code-cell} ipython3
fig, ax = plt.subplots(2, 1, figsize=(6, 9))
# plot the real part only and our fitted model first
names = ["Born", "T-matrix", "T-matrix + inel.", "KG"]
collisionfreqs = [born, Tmat, Tplus, KG]
for colldata, name in zip(collisionfreqs, names):
    # fit the model to the data
    # sigma is the data so we consider the relative residual in the fit: (ydata - model) / ydata
    popt, pcov = curve_fit(model, freq, colldata.real, p0=(1, 1, 1, 1, 1), sigma=colldata.real, bounds=(0, np.inf))
    # print(name)
    # print(f"optimized parameters: {popt}")
    # print(f"condition number of cov. matrix: {np.linalg.cond(pcov)}")
    # plot the data
    p = ax[0].plot(freqev, colldata.real)
    # plot the model with optimized parameters
    ax[0].plot(freqev, model(freq, *popt), color=p[-1].get_color(), ls="--", alpha=0.6)

    # ELF
    # plot the true ELF 
    trueelf = elec_loss_fn(
        mermin(wavenum * AtomicUnits.length, freq, lambda x: np.interp(x, freq, colldata))
    )
    p = ax[1].plot(
        freqev,
        trueelf,
        label=name
    )
    # plot ELF with fitted model
    modelelf = elec_loss_fn(
        mermin,
        wavenum * AtomicUnits.length,
        freq,
        lambda x: collisionfreq(x, popt)
    )
    p = ax[1].plot(
        freqev,
        modelelf,
        ls="--",
        color=p[-1].get_color(),
        alpha=0.6
    )

# plot RPA ELF
ax[1].plot(
    freqev,
    elec_loss_fn(mermin(wavenum * AtomicUnits.length, freq)),
    ls="-.",
    color="grey",
    label="RPA"
)
    

ax[0].set_xscale("log")
ax[0].set_yscale("log")
# ax[0].set_ylabel(r"Re$\{\nu(\omega)\}$ (at. u.)")
ax[0].set_ylabel("Collision Frequency (at. u.)")
ax[1].set_xlim(0, 40)
ax[1].set_ylabel("ELF")
ax[1].set_xlabel("Frequency (eV)")
ax[1].legend()
plt.tight_layout()
plt.savefig("../../reports/figures/model_flexibility")
```

# Comparing my Kramers-Kronig method with the imaginary parts in the data

```{code-cell} ipython3
p = plt.plot(freqev, data[2], label="T-matrix (data)")
plt.plot(freqev, kramerskronig(freq, lambda x: np.interp(x, freq, data[1])), ls="--")
p = plt.plot(freqev, data[4], label="T+ (data)")
plt.plot(freqev, kramerskronig(freq, lambda x: np.interp(x, freq, data[3])), ls="--")
p = plt.plot(freqev, data[6], label="KG (data)")
plt.semilogx(freqev, kramerskronig(freq, lambda x: np.interp(x, freq, data[5])), ls="--")
plt.ylabel(r"Im$\{\nu(\omega)\}$ (at. u.)")
plt.xlabel("Frequency (eV)")
plt.legend()
```

```{code-cell} ipython3

```
