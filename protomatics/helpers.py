import os
import warnings

import numpy as np

from .constants import au_pc, c_kms


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """This normalizes an array between 0, 1"""

    x = np.array(arr)

    return (x - np.min(x)) / np.max(x - np.min(x))


def get_vels_from_freq(hdr, relative: bool = True, syst_chan: int = 0):
    """Gets velocities from a fits header using frequency as units"""

    f0 = hdr["CRVAL3"]
    delta_f = hdr["CDELT3"]
    center = int(hdr["CRPIX3"])
    num_freq = int(hdr["NAXIS3"])
    freqs = [f0 + delta_f * (i - center) for i in range(num_freq)]
    vels = np.array([-c_kms * (f - f0) / f0 for f in freqs])
    if relative:
        if syst_chan > len(vels) - 1:
            warnings.warn("Systemic channel is too high; using middle channel", stacklevel=2)
            syst_chan = len(vels) // 2
        vels -= vels[syst_chan]

    return vels


def get_vels_from_dv(hdu: list) -> np.ndarray:
    """Gets velocities from a fits header using dv as units"""

    vels = []
    for i in range(hdu[0].header["NAXIS3"]):
        vel = (
            hdu[0].header["CDELT3"] * (i + 1 - hdu[0].header["CRPIX3"]) + hdu[0].header["CRVAL3"]
        )
        vels.append(vel)

    return np.array(vels)


def angular_to_physical(angle: float, distance: float = 200, units: str = "au") -> float:
    """Converts angular size (arcseconds) to physical size (distance in pc)"""

    angle /= 3600.0
    angle /= 180.0
    angle *= np.pi

    pc_size = 2.0 * distance * np.tan(angle / 2.0)

    return pc_size * au_pc if units == "au" else pc_size


def check_and_make_dir(path: str) -> None:
    """Makes a directory if it doesn't exist"""

    if not os.path.exists(path):
        os.mkdir(path)


def cartesian_to_cylindrical(x: float, y: float) -> tuple:
    """Converts x, y to r, phi"""

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return r, phi


def cylindrical_to_cartesian(r: float, phi: float) -> tuple:
    """Converts r, phi (radians) to x, y"""

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y
