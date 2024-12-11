import os
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import sarracen as sn

from .constants import au_pc, c, c_kms, jansky


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


def to_mJy_pixel(
    hdu: Optional[list] = None,
    value: float = 1.0,
    wl: Optional[float] = None,
    Hz: Optional[float] = None,
):
    """Watt/m2/pixel to mJy/pixel"""

    ### assuming wl in microns
    if Hz is None:
        if hdu is not None and "wave" in hdu[0].header:
            wl = hdu[0].header["wave"] * 1e-6
        assert wl is not None, "No wavelength or frequency information!"
        Hz = c / wl

    return 1e3 * (jansky**-1.0) * value / Hz


def get_r_bins(
    sdf: pd.DataFrame,
    rmin: Optional[float] = None,
    rmax: Optional[float] = None,
    dr: float = 0.5,
    nr: Optional[int] = None,
    return_rs: bool = False,
):
    """Bins into discrete radial regions"""

    if "r" not in sdf.columns:
        sdf["r"] = np.sqrt(sdf.x**2 + sdf.y**2)
    rmin = np.min(sdf.r) if rmin is None else rmin
    rmax = np.max(sdf.r) if rmax is None else rmax
    rs = np.linspace(rmin, rmax, nr) if nr is not None else np.arange(rmin, rmax, dr)
    dr = dr if dr is not None else np.abs(rs[1] - rs[0])
    # Define the edges of the bins
    bin_edges = np.append(
        rs - dr / 2,
        rs[-1] + dr / 2,
    )
    # Assign each particle to a radial bin
    sdf["r_bin"] = pd.cut(sdf["r"], bins=bin_edges, labels=rs, include_lowest=True)

    if return_rs:
        return sdf, rs
    return sdf


def get_phi_bins(
    sdf: sn.SarracenDataFrame,
    phimin: float = -np.pi,
    phimax: float = np.pi,
    dphi: float = np.pi / 20.0,
    nphi: Optional[int] = None,
    return_phis: bool = False,
):
    """Bins into discrete azimuthal regions"""
    if "phi" not in sdf.columns:
        sdf["phi"] = np.arctan2(sdf.y, sdf.x)
    phis = (
        np.linspace(phimin, phimax, nphi) if nphi is not None else np.arange(phimin, phimax, dphi)
    )
    dphi = dphi if dphi is not None else np.abs(phis[1] - phis[0])
    # Define the edges of the bins
    bin_edges = np.append(
        phis - dphi / 2,
        phis[-1] + dphi / 2,
    )
    # Assign each particle to a radial bin
    sdf["phi_bin"] = pd.cut(sdf["phi"], bins=bin_edges, labels=phis, include_lowest=True)

    if return_phis:
        return sdf, phis
    return sdf


def average_within_bins(df: pd.DataFrame, value_column: str, bin_columns: Union[str, list]):
    """
    Calculate the average of a value within each bin.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        value_column (str): Name of the column to average (e.g., 'cs').
        bin_column (str): Name of the binning column (e.g., 'r_bin').
                   (list): multiple bins e.g., 'r_bin', 'phi_bin'

    Returns:
        pd.Series: Series with the average values indexed by bins.
    """
    return df.groupby(bin_columns)[value_column].mean()


def sum_within_bins(df: pd.DataFrame, value_column: str, bin_columns: Union[str, list]):
    """
    Calculate the sum of a value within each bin.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        value_column (str): Name of the column to average (e.g., 'cs').
        bin_column (str): Name of the binning column (e.g., 'r_bin').
                   (list): multiple bins e.g., 'r_bin', 'phi_bin'

    Returns:
        pd.Series: Series with the average values indexed by bins.
    """
    return df.groupby(bin_columns)[value_column].sum()
