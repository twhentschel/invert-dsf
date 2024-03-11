# Invert DSF

The dynamic structure factor (DSF) describes electronic density correlations in a material, and is measured in X-ray Thomson
scattering experiments. Physical models for the DSF that depend on material conditions, like temperature and density,
can be fit to experimental data to infer these parameters. In the language of inverse problems, the DSF represents a _forward model_
whereas the fitting of this forward model to the data is called the _inverse problem_. The goal of this project is to demonstrate
a framework for inverting a specific DSF model for a given set of DSF data.

## DSF model
The specific model we consider is based on the Mermin dielectric function $\epsilon$. In general, the DSF (denoted as $S$) is closely
related to the dielectric function

$$ S(q, \omega) \propto \frac{(\hbar q)^2}{1 - \exp(-\hbar \omega / k_B T)} \mathrm{Im} \left[ \frac{-1}{ \epsilon(q, \omega; \nu(\omega), T, n_e)} \right] .$$

In terms of scattering experiments, $\hbar q$ and $\hbar \omega$ refer to the momentum and energy that is transferred from the
scattering photons to the target, respectively. In particular, we'll call $q$ the wavenumber and $\omega$ the frequency because
they also correspond to the mode of electron-density fluctuations excited during the scattering event.

We also wrote the Mermin dielectric in terms of $q$ and $\omega$, but it also has other parameters that correspond to the state of the
electrons in the scattering target: $T$ is the electron temperature, $n_e$ is the density, and $\nu(\omega)$ is a frequency-dependent
_electron-ion collision frequency_.

When using the Mermin dielectric to construct the DSF, we'll refer to this total model as the Mermin DSF model.

## Inverse problem - determining $\nu(\omega)$
The accuracy of the Mermin DSF predictions relies crucially on the electron-ion collision frequency $\nu(\omega)$ we give to our Mermin model.
There are physical approximations that can be made to come up with models for $\nu$, but it is hard to evaluate their accuracy because
this is not an easily measurable quantity. However, by taking advantage of the close connection between the Mermin dielectric (which depends on
$\nu$) and the DSF (which _is_ measureable), it seems possible that we might be able to extract some information about the collision frequency from
DSF data. This project addresses the feasibility of this idea.

## Analysis
Typically, inverse problems are _ill-posed_, meaning that, among other things,
the solution may not be unique. For example, inverting the DSF by minimizing a non-linear least-squares problem would only
return a single solution, not the potentially-many other collision frequencies consistent with the data. In this project,
we use __Bayesian inference__ to obtain a posterior distribution for the collision frequency for a given set of DSF data. The highlight of this
approach is that, since we have a posterior distribution for $\nu(\omega)$, we can quantify the uncertainty
associated with our inferred collision frequency.




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    |   ├── mcmc           <- Posterior distriubtion data from MCMC sampling
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for analysis.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   |                     the creator's initials, and a short `-` delimited description, e.g.
    │   |                     `1.0-twh-initial-data-exploration`. Notebooks are converts to markdown using 
    |   |                     `jupytext` to facilitate version control 
    |   ├── exploratory    <- notebooks containing initial explorations
    |   ├── tests          <- visual tests of certain features
    |   └── reports        <- notebooks containing polished analysis
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- Make this project pip installable with `pip install -e`
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── inference         <- Scripts to perform Bayesian inference. 
        |   |                    Posterior samples stored in (project)/data/mcmc           
        │   ├── collision_models.py
        │   └── mcmc_inference.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
