# File:       PlaneWaveData.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-04-03
import numpy as np
import h5py
from scipy.signal import hilbert


class PlaneWaveData:
    """ A template class that contains the plane wave data.

    PlaneWaveData is a container or dataclass that holds all of the information
    describing a plane wave acquisition. Users should create a subclass that reimplements
    __init__() according to how their data is stored.

    The required information is:
    idata       In-phase (real) data with shape (nangles, nchans, nsamps)
    qdata       Quadrature (imag) data with shape (nangles, nchans, nsamps)
    angles      List of angles [radians]
    ele_pos     Element positions with shape (N,3) [m]
    fc          Center frequency [Hz]
    fs          Sampling frequency [Hz]
    fdemod      Demodulation frequency, if data is demodulated [Hz]
    c           Speed of sound [m/s]
    time_zero   List of time zeroes for each acquisition [s]
    
    Correct implementation can be checked by using the validate() method. See the
    PICMUSData class for a fully implemented example.
    """

    def __init__(self):
        """ Users must re-implement this function to load their own data. """
        # Do not actually use PlaneWaveData.__init__() as is.
        raise NotImplementedError

        # We provide the following as a visual example for a __init__() method.
        nangles, nchans, nsamps = 2, 3, 4
        # Initialize the parameters that *must* be populated by child classes.
        self.idata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.qdata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.angles = np.zeros((nangles,), dtype="float32")
        self.ele_pos = np.zeros((nchans, 3), dtype="float32")
        self.fc = 5e6
        self.fs = 20e6
        self.fdemod = 0
        self.c = 1540
        self.time_zero = np.zeros((nangles,), dtype="float32")

    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, angles, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nangles, nchans, nsamps = self.idata.shape
        assert self.angles.ndim == 1 and self.angles.size == nangles
        assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        assert self.time_zero.ndim == 1 and self.time_zero.size == nangles


class PICMUSData(PlaneWaveData):
    """ PICMUSData - Demonstration of how to use PlaneWaveData to load PICMUS data

    PICMUSData is a subclass of PlaneWaveData that loads the data from the PICMUS
    challenge from 2016 (https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016).
    PICMUSData re-implements the __init__() function of PlaneWaveData.
    """

    def __init__(self, database_path, acq, target, dtype):
        """ Load PICMUS dataset as a PlaneWaveData object. """
        # Make sure the selected dataset is valid
        assert any([acq == a for a in ["simulation", "experiments"]])
        assert any([target == t for t in ["contrast_speckle", "resolution_distorsion"]])
        assert any([dtype == d for d in ["rf", "iq"]])

        # Load PICMUS dataset
        fname = "%s/%s/%s/%s_%s_dataset_%s.hdf5" % (
            database_path,
            acq,
            target,
            target,
            acq[:4],
            dtype,
        )
        f = h5py.File(fname, "r")["US"]["US_DATASET0000"]
        self.idata = np.array(f["data"]["real"], dtype="float32")
        self.qdata = np.array(f["data"]["imag"], dtype="float32")
        self.angles = np.array(f["angles"])
        self.fc = 5208000.0  # np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = np.array(f["sound_speed"]).item()
        self.time_zero = np.array(f["initial_time"])
        self.ele_pos = np.array(f["probe_geometry"]).T
        self.fdemod = self.fc if dtype == "iq" else 0

        # If the data is RF, use the Hilbert transform to get the imag. component.
        if dtype == "rf":
            iqdata = hilbert(self.idata, axis=-1)
            self.qdata = np.imag(iqdata)

        # Make sure that time_zero is an array of size [nangles]
        if self.time_zero.size == 1:
            self.time_zero = np.ones_like(self.angles) * self.time_zero

        # Validate that all information is properly included
        super().validate()
