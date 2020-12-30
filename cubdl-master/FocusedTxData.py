# File:       FocusedTxData.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-04-18
import numpy as np


class FocusedTxData:
    """ A template class that contains the focused transmit data.

    FocusedTxData is a container or dataclass that holds all of the information describing
    a focused transmit acquisition. Users should create a subclass that reimplements
    __init__() according to how their data is stored.

    The required information is:
    idata       In-phase (real) data with shape (nxmits, nchans, nsamps)
    qdata       Quadrature (imag) data with shape (nxmits, nchans, nsamps)
    tx_ori      List of transmit origins with shape (N,3) [m]
    tx_dir      List of transmit directions with shape (N,2) [radians]
    ele_pos     Element positions with shape (N,3) [m]
    fc          Center frequency [Hz]
    fs          Sampling frequency [Hz]
    fdemod      Demodulation frequency, if data is demodulated [Hz]
    c           Speed of sound [m/s]
    time_zero   List of time zeroes for each acquisition [s]

    Correct implementation can be checked by using the validate() method.
    """

    def __init__(self):
        """ Users must re-implement this function to load their own data. """
        # Do not actually use FocusedTxData.__init__() as is.
        raise NotImplementedError

        # We provide the following as a visual example for a __init__() method.
        nxmits, nchans, nsamps = 2, 3, 4
        # Initialize the parameters that *must* be populated by child classes.
        self.idata = np.zeros((nxmits, nchans, nsamps), dtype="float32")
        self.qdata = np.zeros((nxmits, nchans, nsamps), dtype="float32")
        self.tx_ori = np.zeros((nxmits, 3), dtype="float32")
        self.tx_dir = np.zeros((nxmits, 2), dtype="float32")
        self.ele_pos = np.zeros((nchans, 3), dtype="float32")
        self.fc = 5e6
        self.fs = 20e6
        self.fdemod = 0
        self.c = 1540
        self.time_zero = np.zeros((nxmits,), dtype="float32")

    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, tx_ori, tx_dir, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nxmits, nchans, nsamps = self.idata.shape
        assert self.tx_ori.shape == (nxmits, 3)
        assert self.tx_dir.shape == (nxmits, 2)
        assert self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        assert self.time_zero.ndim == 1 and self.time_zero.size == nxmits
