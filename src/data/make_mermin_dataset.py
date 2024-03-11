import pandas as pd
import numpy as np

from numpy.typing import ArrayLike
from typing import Callable

from src.utilities import elec_loss_fn
from src.utilities import AtomicUnits

from uegdielectric import ElectronGas
from uegdielectric.dielectric import Mermin


class MerminELFData:
    """Class to make Mermin ELF data and store it in a file."""

    def __init__(
        self,
        temperature: float,
        density: float,
        wavenum: ArrayLike,
        frequency: ArrayLike,
        collisonrate: Callable = None,
    ) -> None:
        self.electronparams = ElectronGas(temperature, density)
        self.wavenum = wavenum
        self.frequency = frequency
        self.collisionrate = collisonrate
        self.ELF = elec_loss_fn(
            Mermin(self.electronparams),
            self.wavenum,
            self.frequency,
            collisonrate,
        )

    def writedata(self, filename, info=""):
        """Write electron loss function (ELF) data to file.

        By default, different values of `self.wavenum` are
        stored as columns in the file, while the rows correspond to
        `self.frequency` values.

        - info (str):
            extra information stored as part of a header string in the
            file.
        """
        # convert parameters to non-atomic units
        temp = self.electronparams.temperature * AtomicUnits.energy  # [eV]
        den = self.electronparams.density / AtomicUnits.length**3  # [1/A^3]
        q = self.wavenum / AtomicUnits.length  # [1/A]
        omega = self.frequency * AtomicUnits.energy  # [eV]

        # if len(self.wavenum) = m and len(self.frequency) = n, then
        # self.ELF.shape = (m, n)
        # we will also append collision rate data for saving
        collisiondata = self.collisionrate(self.frequency)
        ELFdata = np.vstack((collisiondata.real, collisiondata.imag, self.ELF))
        # store transpose ELFdata so columns correspond to wave numbers
        data = pd.DataFrame(
            data=ELFdata.T,
            index=omega,
            columns=[
                "coll. rate (real)",
                "coll. rate (imag)",
                *[f"q={x:.6f}" for x in q],
            ],
        )

        header = (
            "Mermin ELF data\n"
            + info
            + "\n"
            + "Parameters:\n___________\n"
            + f"temperature (eV): {temp:.3f}\n"
            + f"density (1/Angstrom^3): {den:.6f}\n\n"
            + "The first column is the frequency (eV), the second and third\n"
            + "are the real and imaginary parts of the collision rate (a.u.)\n"
            + "(as a function of the frequency) and the rest are the\n"
            + "values of the ELF for each wave number q (1/Angstrom) \n\n"
        )

        out = data.to_string(float_format="%.6f", header=header)
        # write data to file
        with open(filename, "w") as f:
            f.write(header + out)


if __name__ == "__main__":
    from src.inference.collision_models import (
        collision_activate_decay,
        collision_activate_decay_imag,
    )
    import json

    # thermal energy (kBT) in eV
    t_eV = 1  # [eV]
    # convert to automic units
    t = t_eV / AtomicUnits.energy

    # electron density (1.8071E23 is the electron density of solid aluminum at
    # a thermal energy of 1 eV)
    den_cc = 1.8071e23  # [electrons]/[cm^3]
    # convert to atomic units (convert [angstrom] to [cm] first)
    d = den_cc * (AtomicUnits.length * 10**-8) ** 3  # [au]

    # wave numbers
    q = np.asarray([0.5, 1, 1.5])  # [au]
    freq = np.geomspace(1e-3, 1e2, 101)

    # collision rate parameters
    params = {
        "lorentzian_height": 1,
        "lorentzian_powerlaw": 0.5,
        "logistic_activate": 0.1,
        "logistic_gradient": 100,
    }

    # define our collision rate function
    def collisonrate(freq, params):
        # real part
        recollision = collision_activate_decay(freq, **params)
        # imaginary part
        imcollision = collision_activate_decay_imag(freq, **params)

        return recollision + 1j * imcollision

    # Mermin ELF data object
    data = MerminELFData(t, d, q, freq, lambda x: collisonrate(x, params))

    # store collision rate params as info
    collision_info = (
        "\nCollision Rate Parameters:\n"
        + json.dumps(params)
        + "\n"
        + f"Function: {collision_activate_decay.__name__}\n"
    )

    # write data to file
    fname = "data/raw/mermin_ELF_test.txt"
    data.writedata(fname, info=collision_info)
