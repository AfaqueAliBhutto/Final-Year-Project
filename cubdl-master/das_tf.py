# File:       das_tf.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-04-27
import tensorflow as tf
from tqdm import tqdm
import numpy as np


class DAS_PW(tf.keras.models.Model):
    """ TensorFlow implementation of DAS plane wave beamforming.

    This class implements DAS plane wave beamforming as a neural network via a TensorFlow
    Keras Model. Subclasses derived from this class can choose to make certain parameters
    trainable. All components can be turned into trainable parameters.
    """

    def __init__(
        self, P, grid, ang_list=None, ele_list=None, rxfnum=2, device="/gpu:0",
    ):
        """ Initialization method for DAS_PW.

        All inputs are specified in SI units, and stored in self as Pynumpy arrays.
        INPUTS
        P           A PlaneWaveData object that describes the acquisition
        grid        A [ncols, nrows, 3] numpy array of the reconstruction grid
        ang_list    A list of the angles to use in the reconstruction
        ele_list    A list of the elements to use in the reconstruction
        rxfnum      The f-number to use for receive apodization
        device_str  For GPU0: "/gpu:0". For CPU: "/cpu:0".
        """
        super().__init__()
        # If no angle or element list is provided, delay-and-sum all
        if ang_list is None:
            ang_list = range(P.angles.shape[0])
        elif not hasattr(ang_list, "__getitem__"):
            ang_list = [ang_list]
        if ele_list is None:
            ele_list = range(P.ele_pos.shape[0])
        elif not hasattr(ele_list, "__getitem__"):
            ele_list = [ele_list]

        # Convert plane wave data to tensors
        with tf.device(device):
            self.angles = tf.constant(P.angles, dtype=tf.float32)
            self.ele_pos = tf.constant(P.ele_pos, dtype=tf.float32)
            self.fc = tf.constant(P.fc, dtype=tf.float32)
            self.fs = tf.constant(P.fs, dtype=tf.float32)
            self.fdemod = tf.constant(P.fdemod, dtype=tf.float32)
            self.c = tf.constant(P.c, dtype=tf.float32)
            self.time_zero = tf.constant(P.time_zero, dtype=tf.float32)

            # Convert grid to tensor
            self.grid = tf.constant(grid, dtype=tf.float32)
            self.grid = tf.reshape(self.grid, (-1, 3))
            self.out_shape = grid.shape[:-1]

            # Store other information as well
            self.ang_list = tf.constant(ang_list, dtype=tf.int32)
            self.ele_list = tf.constant(ele_list, dtype=tf.int32)
            self.device_str = device

    def call(self, iqdata):
        """ Forward pass for DAS_PW neural network. """
        # Compute delays in meters
        nangles = len(self.ang_list)
        nelems = len(self.ele_list)
        npixels = self.grid.shape[0]
        xlims = (self.ele_pos[0, 0], self.ele_pos[-1, 0])  # Aperture width
        # Initialize delays, apodizations, output array
        txdel = np.zeros((nangles, npixels), dtype="float")
        rxdel = np.zeros((nelems, npixels), dtype="float")
        txapo = np.ones((nangles, npixels), dtype="float")
        rxapo = np.ones((nelems, npixels), dtype="float")
        # Compute transmit and receive delays and apodizations
        for i, tx in enumerate(self.ang_list):
            txdel[i] = delay_plane(self.grid, self.angles[tx])
            txdel[i] += self.time_zero[tx] * self.c
            txapo[i] = apod_plane(self.grid, self.angles[tx], xlims)
        for j, rx in enumerate(self.ele_list):
            rxdel[j] = delay_focus(self.grid, self.ele_pos[rx])
            rxapo[j] = apod_focus(self.grid, self.ele_pos[rx])
        # Place on device as a tensor
        with tf.device(self.device_str):
            txdel = tf.constant(txdel, dtype=tf.float32)
            rxdel = tf.constant(rxdel, dtype=tf.float32)
            txapo = tf.constant(txapo, dtype=tf.float32)
            rxapo = tf.constant(rxapo, dtype=tf.float32)
        # Convert to samples
        txdel *= self.fs / self.c
        rxdel *= self.fs / self.c

        # Initialize the output array
        with tf.device(self.device_str):
            idas = tf.zeros(npixels, dtype=tf.float32)
            qdas = tf.zeros(npixels, dtype=tf.float32)

        # Loop over angles and elements
        for t, td, ta in tqdm(zip(self.ang_list, txdel, txapo), total=nangles):
            # Grab data from t-th Tx
            iq = tf.constant(iqdata[t], dtype=tf.float32)
            # Convert delays to be used with grid_sample
            delays = td + rxdel
            delays = tf.expand_dims(delays, axis=-1)
            # Apply delays
            ifoc, qfoc = apply_delays(iq, delays)
            # Apply phase-rotation if focusing demodulated data
            if self.fdemod != 0:
                tshift = delays[:, :, 0] / self.fs
                tdemod = tf.expand_dims(self.grid[:, 2], 0) * 2 / self.c
                theta = 2 * np.pi * self.fdemod * (tshift - tdemod)
                ifoc, qfoc = _complex_rotate(ifoc, qfoc, theta)
            # Apply apodization, reshape, and add to running sum
            apods = ta * rxapo
            idas += tf.reduce_sum(ifoc * apods, axis=0, keepdims=False)
            qdas += tf.reduce_sum(qfoc * apods, axis=0, keepdims=False)

        # Finally, restore the original pixel grid shape and convert to numpy array
        idas = tf.reshape(idas, self.out_shape)
        qdas = tf.reshape(qdas, self.out_shape)
        return idas, qdas


@tf.function
def apply_delays(iq, d):
    """ Apply time delays using linear interpolation. """
    # Get lower and upper values around delays dd
    d0 = tf.cast(tf.floor(d), "int32")  # Cast to integer
    d1 = d0 + 1
    # Gather pixel values
    iq0 = tf.gather_nd(iq, d0, batch_dims=1)
    iq1 = tf.gather_nd(iq, d1, batch_dims=1)
    # Compute interpolated pixel value
    d0 = tf.cast(d0, "float32")  # Cast to float
    d1 = tf.cast(d1, "float32")  # Cast to float
    out = (d1 - d) * iq0 + (d - d0) * iq1
    # Grab I and Q components separately
    ifoc, qfoc = out[:, :, 0], out[:, :, 1]
    return ifoc, qfoc


## Simple phase rotation of I and Q component by complex angle theta
@tf.function
def _complex_rotate(i, q, theta):
    ir = i * tf.cos(theta) - q * tf.sin(theta)
    qr = q * tf.cos(theta) + i * tf.sin(theta)
    return ir, qr


## Compute distance to user-defined pixels from elements
# Expects all inputs to be numpy arrays specified in SI units.
# INPUTS
#   grid    Pixel positions in x,y,z    [npixels, 3]
#   ele_pos Element positions in x,y,z  [nelems, 3]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_focus(grid, ele_pos):
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = np.linalg.norm(grid - tf.expand_dims(ele_pos, 0), axis=-1)
    return dist


## Compute distance to user-defined pixels for plane waves
# Expects all inputs to be numpy arrays specified in SI units.
# INPUTS
#   grid    Pixel positions in x,y,z    [npixels, 3]
#   angles  Plane wave angles (radians) [nangles]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_plane(grid, angles):
    # Use broadcasting to simplify computations
    x = np.expand_dims(grid[:, 0], 0)
    z = np.expand_dims(grid[:, 2], 0)
    # For each element, compute distance to pixels
    dist = x * np.sin(angles) + z * np.cos(angles)
    return dist


## Compute rect apodization to user-defined pixels for desired f-number
# Expects all inputs to be numpy arrays specified in SI units.
# INPUTS
#   grid        Pixel positions in x,y,z        [npixels, 3]
#   ele_pos     Element positions in x,y,z      [nelems, 3]
#   fnum        Desired f-number                scalar
#   min_width   Minimum width to retain         scalar
# OUTPUTS
#   apod    Apodization for each pixel to each element  [nelems, npixels]
def apod_focus(grid, ele_pos, fnum=1, min_width=1e-3):
    # Get vector between elements and pixels via broadcasting
    ppos = np.expand_dims(grid, 0)
    epos = np.reshape(ele_pos, (-1, 1, 3))
    v = ppos - epos
    # Select (ele,pix) pairs whose effective fnum is greater than fnum
    mask = np.abs(v[:, :, 2] / (v[:, :, 0] + 1e-30)) > fnum
    mask = mask | (np.abs(v[:, :, 0]) <= min_width)
    # Also account for edges of aperture
    mask = mask | ((v[:, :, 0] >= min_width) & (ppos[:, :, 0] <= epos[0, 0, 0]))
    mask = mask | ((v[:, :, 0] <= -min_width) & (ppos[:, :, 0] >= epos[-1, 0, 0]))
    # Convert to float and normalize across elements (i.e., delay-and-"average")
    apod = np.array(mask, dtype="float32")
    # Output has shape [nelems, npixels]
    return apod


## Compute rect apodization to user-defined pixels for desired f-number
# Retain only pixels that lie within the aperture projected along the transmit angle.
# Expects all inputs to be numpy arrays specified in SI units.
# INPUTS
#   grid    Pixel positions in x,y,z            [npixels, 3]
#   angles  Plane wave angles (radians)         [nangles]
#   xlims   Azimuthal limits of the aperture    [2]
# OUTPUTS
#   apod    Apodization for each angle to each element  [nangles, npixels]
def apod_plane(grid, angles, xlims):
    pix = np.expand_dims(grid, 0)
    ang = np.reshape(angles, (-1, 1, 1))
    # Project pixels back to aperture along the defined angles
    x_proj = pix[:, :, 0] - pix[:, :, 2] * np.tan(ang)
    # Select only pixels whose projection lie within the aperture, with fudge factor
    mask = (x_proj >= xlims[0] * 1.2) & (x_proj <= xlims[1] * 1.2)
    # Convert to float and normalize across angles (i.e., delay-and-"average")
    apod = np.array(mask, dtype="float32")
    # Output has shape [nangles, npixels]
    return apod
