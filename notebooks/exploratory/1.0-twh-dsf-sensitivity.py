# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DSF sensitivty to the collision frequency
#
# We examine how sensitive the dyanmic structure factor (DSF) is to the choice of collision frequency, $\nu(\omega)$, where $\omega$ is the energy (frequency) transferred during the scattering experiment. We are specifically interested in the high-frequency (that is, high-$\omega$) limit. We use the Mermin dielectric function, $\epsilon(q, \omega; \nu[\omega])$, to compute the DSF $S(q, \omega)$ via the formula
#
# $$ S(q, \omega) = \frac{1}{1 - \exp(-\omega / k_B T)}\frac{q^2}{4 \pi n_e} \mathrm{Im}\bigg\{\frac{-1}{\epsilon(q, \omega)}\bigg\},$$
#
# where $q$ is the momentum (wavenumber) transferred during the scattering experiment, $T$ is the (electron) temperature, and $n_e$ is the electron density. Because the only part of the DSF that depends on the collision frequency is the term $\mathrm{Im}\{-1/\epsilon(q, \omega)\}$ (called the __electron loss function [ELF]__), we focus on the sensitivity of that. We can also write the ELF as
#
# $$ \mathrm{Im}\bigg(\frac{-1}{\epsilon}\bigg ) =   \frac{\mathrm{Im}(\epsilon)}{\mathrm{Re}(\epsilon)^2 + \mathrm{Im}(\epsilon)^2} . $$
#
# Unless specified, everything will be written/computed in atomic units (see [this Wikipedia article](https://en.wikipedia.org/wiki/Hartree_atomic_units) for more information).

# %% [markdown]
# ## Motivation to explore the DSF sensitivity
#
# The collision frequency enters the Mermin dielectric function in a very specific way. Consider the Mermin dielectric function:
# $$ \epsilon(q, \omega; \nu[\omega]) = 1 + \frac{(\omega+i\nu)[\epsilon^0(q, \omega+i\nu) - 1]}{\omega + i\nu \frac{\epsilon^0(q, \omega+i\nu) - 1}{\epsilon^0(q,0)-1}},$$
#
# where $\epsilon^0$ is the random phase approximation (RPA) dielectric function. When $\omega >> \nu(\omega)$, we have (roughly) $\epsilon(q, \omega; \nu) \approx \epsilon^0(q, \omega+i\nu) \approx \epsilon^0(q, \omega)$. So the Mermin dielectric, and thus the ELF and the DSF, should be mostly indepedent of the details on $\nu$ for large $\omega$'s. (Note we are assuming that $\big | \frac{\epsilon^0(q, \omega+i\nu) - 1}{\epsilon^0(q,0)-1} \big | \leq 1$)

# %%
import numpy as np
import matplotlib.pyplot as plt
from uegdielectric import ElectronGas, dielectric


# %%
# Load the "autoreload" extension so that code can change
# %load_ext autoreload

# Always reload modules so that as you change code in src, it gets loaded
# %autoreload 2

# %% [markdown]
# ## ELF


# %%
def elf(q, ω, ν, e):
    """
    Computes the ELF

    q: array-like or float
        wavenumber
    ω: array-like or float
        frequency
    ν: function, float, or None
        collision frequency
    e: ElectronGas object
        ElectronGas instance containing parameters of the material
    """
    # dielectric model
    model = dielectric.Mermin(e, ν)
    # evaluate dielectric at q and ω points
    vals = model(q, ω)

    return vals.imag / (vals.real**2 + vals.imag**2)


# %% [markdown]
# ## Parameters

# %% [markdown]
# Define some constants that will be useful to convert to atomic units (au)

# %%
# atomic unit of energy: 1 hartree = 27.2114 eV (electron volts)
Ha = 27.2114  # eV
# atomic unit of length: 1 bohr radius = 0.529 A (angstroms)
a0 = 0.529177  # A
# Boltzmann's constant -- used to convert temperature (K, kelvin) to thermal energy (eV)
kB = 8.61733e-5  # eV/K

# %% [markdown]
# Electron temperatures and densities

# %%
# thermal energy (kBT) in eV
t_eV = np.array([0.1, 1, 10])  # [eV]
# convert to automic units
t = t_eV / Ha

# electron density (1.8071E23 is the electron density of solid aluminum at a thermal energy of 1 eV)
den_cc = 1.8071e23 * np.array([1 / 10, 1, 10])  # [electrons]/[cm^3]
# convert to atomic units (convert a0 to [cm] first)
d = den_cc * (a0 * 10**-8) ** 3  # [au]

# %% [markdown]
# Wavenumbers

# %%
q = [0.5, 1, 1.5]  # [au]


# %% [markdown]
# ## Collision frequency model
#
# Here we start with a simple Lorentzian model centered at $\omega = 0$ for $\nu$. This choice is based on the form of the Drude conductivity __[confirm this]__


# %%
def ν_lorentz(ω, width=1, height=1):
    """
    Lorentzian model for the collision frequency

    ω: dependent variable
    width: width of lorentzian peak
    height: height of lorentzian peak
    """
    return height / (1 + ω**2 / width**2)


# %%
xlog = np.geomspace(1e-1, 1e3, 100)
plt.plot(xlog, ν_lorentz(xlog, width=10))
plt.xscale("log")
plt.ylabel("Lorentzian collision frequency")
plt.xlabel(r"$\omega$")

# %%
xlog = np.geomspace(1e-1, 1e3, 100)
plt.plot(xlog, ν_lorentz(xlog, width=10))
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Lorentzian collision frequency")
plt.xlabel(r"$\omega$")


# %% [markdown]
# Now, we will modify the simple Lorentzian model so that we have more control over the high-$\omega$ behavior. For that, we will scale the Lorentzian by the polynomial $(1 + (\omega/\gamma))^\alpha$, where $\gamma$ is the `width` of the Lorentzian, and $\alpha$ roughly controls how fast the high-frequency tail rises. There are many other approaches we can take for this, like using a "generalized" Lorentzian function $\nu(\omega) = h / (1 + (\omega/\gamma)^\alpha)$, but one example will suffice to demonstrate the plausability of the ELF's (and DSF's) sensitivity.


# %%
def ν_model(ω, width=1, height=1, rise=0):
    """
    Collision frequency model, as a modified version of the Lorentzian model.
    `rise` roughly controls how fast the high frequency tails "rise", with 0
    coresponding to a normal Lorentzian function.
    """
    return ν_lorentz(ω, width, height) * (1 + (ω / width)) ** rise


# %%
xlog = np.geomspace(1e-1, 1e3, 100)
# test out different values for the rise parameter
risevals = [0, 0.5, 1, 1.5, 1.75, 2]
for rise in risevals:
    plt.plot(
        xlog,
        ν_model(xlog, width=10, height=1, rise=rise),
        label=r"$\alpha$ = {:.2f}".format(rise),
    )
plt.xscale("log")
plt.yscale("linear")
plt.ylabel("Lorentzian collision frequency")
plt.xlabel(r"$\omega$")
plt.legend()

# %%
xlog = np.geomspace(1e-1, 1e3, 100)
# test out different values for the rise parameter
risevals = [0, 0.5, 1, 1.5, 1.75, 2]
for rise in risevals:
    plt.plot(
        xlog,
        ν_model(xlog, width=10, rise=rise),
        label=r"$\alpha$ = {:.2f}".format(rise),
    )
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Lorentzian collision frequency")
plt.xlabel(r"$\omega$")
plt.legend()

# %% [markdown]
# ### Imaginary part
#
# The collision frequency has the special property that it obeys Kramers-Kronig relationship. This means that the real and imaginary parts of the collision frequency are connected through the Kramers-Kronig transform. This function is stored in `src/utilities`.

# %%
from src.utilities import collfreqimag

# %%
x = np.linspace(1e-1, 5e2, 2000)
νr = ν_lorentz(x, width=10)
νi = collfreqimag(νr)

# %%
plt.plot(x, νr)
plt.plot(x, νi)
plt.xscale("log")
plt.yscale("log")


# %% [markdown]
# ## Sensitivity analysis
#
# We will calculate the ELF using collision frequencies with different high-$\omega$ behavior, and also consider a few different density and temperature conditions.


# %%
def plotsensitivity(q, t, d, subregion):
    """
    Plots for sensitivity analysis as a function of the wavenumber (q),
    temperature (t), and density (d) (all in [au]).

    subregion is a four-tuple (x1, x2, y1, y2) of a subregion of the
    original plot
    """
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    # inset axes focusing on ELF peak
    x1, x2, y1, y2 = subregion  # subregion of the original plot
    axins = axs[0].inset_axes(
        [0.05, 0.05, 0.4, 0.7], xlim=(x1, x2), ylim=(y1, y2)
    )

    # define the electron gas
    eg = ElectronGas(t, d)
    # define the parameters controlling the high-ω tail behavior of ν
    tailparams = [0, 0.5, 1, 1.5, 2]

    # frequency range
    ω = np.linspace(
        1e-1, 1e2, 2000
    )  # [au] - will want to look on a logscale so don't include 0

    # plot the ELF for different tail parameters
    for param in tailparams:
        ν = lambda ω_: ν_model(ω_, width=10, rise=param)
        elfcalc = elf(q, ω, ν, eg)

        # ELF plot
        pl = axs[0].loglog(ω, elfcalc)

        # ELF inset - semilogx focused on peak
        axins.plot(ω, elfcalc)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axs[0].indicate_inset_zoom(axins, edgecolor="grey")

        # collision frequency plot
        axs[1].loglog(ω, ν(ω), c=pl[-1].get_color())

    axs[0].set_ylabel("ELF")
    axs[1].set_ylabel("Collision Freq.")
    axs[1].set_xlabel(r"$\omega$")

    return axs


# %%
axs = plotsensitivity(q[0], t[1], d[1], [0.1, 1, 0.2, 0.6])

# %%
# lower temperature
axs = plotsensitivity(q[0], t[0], d[1], [0.1, 1, 0.2, 0.6])
axs[0].set_title("Lower temperature")

# %%
# high density
axs = plotsensitivity(q[0], t[1], d[2], [0.1, 3, 0.2, 2])
axs[0].set_title("higher density")

# %%
# higher wavenumber
axs = plotsensitivity(q[1], t[1], d[2], [1, 4, 0.2, 1.2])
axs[0].set_title("higher wavenumber")

# %%
