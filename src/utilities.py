"""Miscellaneous helper functions"""

import numpy as np
import scipy.fftpack as ft


def collfreqimag(collfreqreal):
    r"""
    Use the Kramers-Kronig transformation to compute the imaginary part of the
    collision frequency `collfreqimag` from the real part. Here, since the real
    part is symmetric about $\omega = 0$, the input is assumed to start at
    $\omega = 0$. The true Kramers-Kronig transformation is an integral over
    the whole real line, but if `collfreqreal` goes to zero before that, we can
    get away with doing an integral that goes until `collfreqreal` is close
    enough to zero. Thus, its important that the inputs `collfreqreal` are
    close to zero near the end of the array.

    Notes:
    This relies on the inverse Hilbert transform to transform the real values
    to the imaginary values. The implementation of the Hilbert transform uses
    the fast Fourier transform (FFT) algorithm, which essentially assumes a
    periodic function. In other words, our input values are periodically
    extended over the whole real line (so that
    collfreqreal(x) = collfreqreal(x + L) where L range of $\omega$ we are
    considering). For the best results, the start and ending values of
    `collfreqreal` should be near each other, otherwise a periodic extension
    of this function would result in sharp jumps near the boundaries and yield
    high frequency components in the Fourier transform. For the inputs assumed
    here, we artificially extend `collfreqreal` about $\omega = 0$ before
    feeding it to `scipy.fftpack.ihilbert`.

    collfreqreal: array
        Values of the real part of the function f.

    returns:
    collfreqimag: array
        Values of the imaginary part of the function f.
    """
    real_ext = np.concatenate((collfreqreal[::-1], collfreqreal))
    imag_ext = ft.ihilbert(real_ext)
    collfreqimag = imag_ext[len(collfreqreal):]
    return collfreqimag
