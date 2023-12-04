# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Collision frequency models
#
# We ultimately want to infer a collision frequency from DSF data, but to do that we need to assume some form for the collision frequency. This notebook explores some of the functional forms we may or may consider in our formal analysis.

# %%
import numpy as np
import matplotlib.pyplot as plt
from src.utilities import collfreqimag


# %% [markdown]
# ## Model 1: Lorenzian * Sigmoid
#
# This model is based on the Drude form for the real part of collision frequency which is simply a Lorentzian function
# $$ \nu_\mathrm{L}(\omega) = \frac{\nu_0}{1 + (\omega / \gamma)^2},$$
# where $\nu_0$ is the height of the Lorentzian peak, $\gamma$ is the effective width, and $\omega$ is the frequency or energy of the electrons that will collide with ions in this situation.
#
# At higher frequencies, it is possible to excite other collision processes that are not accounted for by a simple Drude (_i.e._ billiard-ball-like) picture of electron-ion collisions. To model this, we modifiy the Lorentzian model by including a term that "turns on" at a specific frequency $\omega$. We pick the logistic sigmoid function to model the "turning on" feature, but any type of sigmoid can probably do the trick:
#
# $$ \nu_\mathrm{L, s}(\omega)= \nu_\mathrm{L}(\omega; \nu_0, \gamma) \left(1 + \frac{h}{1 + e^{-\alpha(\omega - \omega_0)}}\right),$$
#
# where $h$ is the height of the sigmoid function, $\alpha$ is a scaling function that governs how quickly the sigmoid activates, and $\omega_0$ represents the frequency at which it turns on. We will plot it to get a feel for its behavior.

# %%
def logistic(x, activate=0, scale=1):
    return 1 / (1 + np.exp(-scale * (x - activate)))


# %%
def lorentzian(x, center=0, width=1):
    return 1 / (1 + ((x - center)/width)**2)


# %%
def ν_model_lorsig(ω, lheight = 1, lwidth = 1, sheight = 1, sactivate = 0, sscale=1):
    return lheight * lorentzian(ω, width=lwidth) * (1 + sheight * logistic(ω, sactivate, sscale))


# %%
def plotmodel(lheight=1, lwidth=10, sheight=1, sactivate=1, sscale=1):
    fig, ax = plt.subplots()
    ω = np.geomspace(1e-3, 1e3, 200)
    ax.loglog(ω, ν_model_lorsig(ω, lheight, lwidth, sheight, sactivate, sscale))
    ax.plot(ω, lheight * lorentzian(ω, width=lwidth), ls='--', c='k', label="Lorentzian")
    ax.plot(ω, sheight * logistic(ω, sactivate, sscale) + 1e-3, ls='--', c='gray', label="logisitc + 1e-3")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("Collision frequency")
    ax.legend()


# %%
sheight=10
sactivate=1e-1
sscale=100
plotmodel(sheight=sheight, sactivate=sactivate, sscale=sscale)

# %%
sheight=1.5
sactivate=1
sscale=1
plotmodel(sheight=sheight, sactivate=sactivate, sscale=sscale)

# %%
sheight=10
sactivate=2e1
sscale=0.5
plotmodel(sheight=sheight, sactivate=sactivate, sscale=sscale)
