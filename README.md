# Invert DSF
==============================

Analysis for inverting the dynamic structure factor to obtain a collision frequency, based on the Mermin dielectric model.

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
    |   └── reports        <- notebooks containing polished analysis
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into a more useful form for inference
        │   └── build_features.py
        │
        ├── bayes         <- Scripts to define distributions and perform Bayesian inference. Posterior
        │   │                samples stored in (project)/data/mcmc 
        │   ├── distributions.py
        │   └── inference.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
