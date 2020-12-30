# File:       metrics.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-04-20
import numpy as np


# Compute contrast ratio
def contrast(img1, img2):
    return img1.mean() / img2.mean()


# Compute contrast-to-noise ratio
def cnr(img1, img2):
    return (img1.mean() - img2.mean()) / np.sqrt(img1.var() + img2.var())


# Compute the generalized contrast-to-noise ratio
def gcnr(img1, img2):
    _, bins = np.histogram(np.stack((img1, img2)), bins=256)
    f, _ = np.histogram(img1, bins=bins, density=True)
    g, _ = np.histogram(img2, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))


def res_FWHM(img):
    # TODO: Write FWHM code
    raise NotImplementedError


def speckle_res(img):
    # TODO: Write speckle edge-spread function resolution code
    raise NotImplementedError


def snr(img):
    return img.mean() / img.std()


## Compute L1 error
def l1loss(img1, img2):
    return np.abs(img1 - img2).mean()


## Compute L2 error
def l2loss(img1, img2):
    return np.sqrt(((img1 - img2) ** 2).mean())


def psnr(img1, img2):
    dynamic_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    return 20 * np.log10(dynamic_range / l2loss(img1, img2))


def ncc(img1, img2):
    return (img1 * img2).sum() / np.sqrt((img1 ** 2).sum() * (img2 ** 2).sum())


if __name__ == "__main__":
    img1 = np.random.rayleigh(2, (80, 50))
    img2 = np.random.rayleigh(1, (80, 50))
    print("Contrast [dB]:  %f" % (20 * np.log10(contrast(img1, img2))))
    print("CNR:            %f" % cnr(img1, img2))
    print("SNR:            %f" % snr(img1))
    print("GCNR:           %f" % gcnr(img1, img2))
    print("L1 Loss:        %f" % l1loss(img1, img2))
    print("L2 Loss:        %f" % l2loss(img1, img2))
    print("PSNR [dB]:      %f" % psnr(img1, img2))
    print("NCC:            %f" % ncc(img1, img2))
