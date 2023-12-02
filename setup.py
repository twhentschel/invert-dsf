from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="""Analysis of inverting the dynamic structure factor to obtain
        a collision frequency, based on the Meermin dielectric function.""",
    author="Thomas Hentschel",
    license="MIT",
)
