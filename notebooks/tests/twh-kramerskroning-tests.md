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
# Load the "autoreload" extension so that code can change
%load_ext autoreload

# always reload modules so that as you change code in src, it gets loaded
%autoreload 2
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

import src.utilities as utils
from src.inference.collision_models import collision_activate_decay, collision_activate_decay_imag

import warnings
warnings.filterwarnings('ignore')
```

```{code-cell} ipython3
def kramerkronigFT(fr):
    # make fr symmetric about 0 (assumption that x=0 corresponds to fr[0])
    fr_ext = np.concatenate((fr[::-1], fr))
    fi_ext = hilbert(fr_ext).imag
    fi = fi_ext[len(fr):]
    return fi
```

# Compute imaginary part of example function

```{code-cell} ipython3
params = [4.910e-01, 0.1, 2.831e-01, 8.525e+00]
xlog = np.geomspace(1e-3, 1e3, 100)
xlogdense = np.geomspace(1e-3, 1e6, 1_000)
plt.semilogx(
    xlog,
    collision_activate_decay(xlog, *params),
    ls="--",
    label="real"
)
plt.plot(
    xlog,
    collision_activate_decay_imag(xlog, *params),
    label="adaptive",
    lw=3
)
plt.plot(
    xlog,
    utils.kramerskronig_arr(xlog, collision_activate_decay(xlog, *params)),
    label="Fixed samples (100 pts)"
)
plt.plot(
    xlogdense,
    utils.kramerskronig_arr(xlogdense, collision_activate_decay(xlogdense, *params)),
    label="Fixed samples (1,000 pts)",
    ls="-."
)
xFT = np.linspace(-1e3, 1e3, 100_000)
plt.plot(
    xFT,
    hilbert(collision_activate_decay(np.abs(xFT), *params)).imag,
    label="Hilbert (10,000 pts)",
    ls="--"
)

plt.legend()
plt.ylabel("collision freq")
plt.xlabel("freq.")
plt.xlim(1e-3, 1e3)
```

# Tests for known complex functions satisfying Kramers-Kronig relations

```{code-cell} ipython3
# real part of known function that obeys Kramers-Kronig
def f1real(x):
    return 1 / (1 + x**2)

# imaginary part of known function that obeys Kramers-Kronig
def f1imag(x):
    return x / (1 + x**2)

def f2real(x):
    return ( 1- x**2) / (x**4 - x**2 + 1)

def f2imag(x):
    return x / (x**4 - x**2 + 1)
```

```{code-cell} ipython3
x = np.linspace(-10, 10, 1000)
plt.plot(x, f1real(x), c="C0", ls="--", label="Re(f1)")
plt.plot(x, f1imag(x), c="C0", ls="-", label="Im(f1)")
plt.plot(x, f2real(x), c="C1", ls="--", label="Re(f2)")
plt.plot(x, f2imag(x), c="C1", ls="-", label="Im(f2)")
plt.legend()
```

```{code-cell} ipython3
x = np.linspace(0, 10, 150)
xsparse = np.linspace(1e-6, 10, 100)
xarray = np.linspace(0, 20, 100)
xdense = np.linspace(0, 20, 1000)
xFT = np.linspace(-100, 100, 10000)

plt.plot(x, f1imag(x), ls="-", lw=3.5, label="Im(f1)")
plt.plot(xarray, utils.kramerskronig_arr(xarray, f1real(xarray)), ls="--", label="fixed samples (100 pts)")
plt.plot(xdense, utils.kramerskronig_arr(xdense, f1real(xdense)), ls="--", label="fixed samples (1,000 pts)")
plt.plot(xsparse, utils.kramerskronig(xsparse, f1real), ls="--", label="adaptive")
plt.plot(xFT, hilbert(f1real(xFT)).imag, ls="-.", label="Hilbert (10,000 pts)", c="tan")

plt.xlim(0, 10)
plt.ylim(0, 0.7)
plt.legend()
```

```{code-cell} ipython3
x = np.linspace(0, 6, 150)
xsparse = np.linspace(1e-6, 6, 100)
xarray = np.linspace(0, 10, 100)
xdense = np.linspace(0, 10, 1000)
xFT = np.linspace(-100, 100, 10000)

plt.plot(x, f2imag(x), ls="-", lw=3.5, label="Im(f2)")
plt.plot(xarray, utils.kramerskronig_arr(xarray, f2real(xarray)), ls="--", label="fixed samples (100 pts)")
plt.plot(xdense, utils.kramerskronig_arr(xdense, f2real(xdense)), ls="--", label="fixed samples (1,000 pts)")
plt.plot(xsparse, utils.kramerskronig(xsparse, f2real), ls="--", label="adaptive")
plt.plot(xFT, hilbert(f2real(xFT)).imag, ls="-.", label="Hilbert (10,000 pts)", c="tan")

plt.xlim(0, 6)
plt.ylim(0, 1.2)
plt.legend()
```

## Hilbert Transform tests

```{code-cell} ipython3
xplot = np.linspace(-10, 10, 1000)
xFT = np.linspace(-1000, 1000, 10000)
plt.plot(xplot, f2real(xplot), ls="-", label="real(f2)")
plt.plot(xFT, -hilbert(f2imag(xFT)).imag, ls="-.", label="Hilbert(imag(f2))")
# # plt.plot(x, kramerskronigfn(x, f2real), ls="--", label="adaptive")
plt.plot(xplot, f1real(xplot), ls="-", label="real(f1)")
plt.plot(xFT, -hilbert(f1imag(xFT)).imag, ls="--", label="Hilbert(imag(f1))")
plt.xlim(-10, 10)
plt.legend()
```

```{code-cell} ipython3
xplot = np.linspace(-10, 10, 1000)
xFT = np.linspace(-100, 100, 10000)
plt.plot(xplot, f2imag(xplot), ls="-", label="imag(f2)")
plt.plot(xFT, hilbert(f2real(xFT)).imag, ls="-.", label="Hilbert(real(f2))")
# plt.plot(x, kramerskronigfn(x, f2real), ls="--", label="adaptive")
plt.plot(xplot, f1imag(xplot), ls="-", label="imag(f1)")
plt.plot(xFT, hilbert(f1real(xFT)).imag, ls="--", label="Hilbert(real(f1))")
plt.xlim(-10, 10)
plt.legend()
```

# Timings

+++

Fixed samples integration

```{code-cell} ipython3
x = np.linspace(0, 10, 100)
print("100-point integration")
%timeit utils.kramerskronig_arr(x, collision_activate_decay(x, *params))
x = np.linspace(0, 10, 1000)
print("1000-point integration")
%timeit utils.kramerskronig_arr(x, collision_activate_decay(x, *params))
```

Adaptive quadrature (100 calculation points)

```{code-cell} ipython3
x = np.linspace(1e-3, 10, 100)
%timeit utils.kramerskronig(x, lambda x: collision_activate_decay(x, *params))
```

Adaptive quadrature w/ Cython LowLevelCallable (100 calculation points)

```{code-cell} ipython3
x = np.geomspace(1e-3, 1e3, 100)
%timeit collision_activate_decay_imag(x, *params)
```

Hilbert Transform (10,000 points)

```{code-cell} ipython3
x = np.linspace(-100, 100, 10_000)
%timeit hilbert(collision_activate_decay(x, *params)).imag
```

```{code-cell} ipython3

```
