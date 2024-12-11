import os
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

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
    df: pd.DataFrame,
    rmin: Optional[float] = None,
    rmax: Optional[float] = None,
    dr: float = 0.25,
    nr: Optional[int] = None,
    return_rs: bool = False,
):
    """Bins into discrete radial regions"""
    return get_bins(df, value="r", vmin=rmin, vmax=rmax, dval=dr, nval=nr, return_vals=return_rs)


def get_phi_bins(
    df: pd.DataFrame,
    phimin: float = -np.pi,
    phimax: float = np.pi,
    dphi: float = np.pi / 20.0,
    nphi: Optional[int] = None,
    return_phis: bool = False,
):
    """Bins into discrete azimuthal regions"""
    return get_bins(
        df, value="phi", vmin=phimin, vmax=phimax, dval=dphi, nval=nphi, return_vals=return_phis
    )


def get_z_bins(
    df: pd.DataFrame,
    zmin: float = -10,
    zmax: float = 10,
    dz: float = 0.25,
    nz: Optional[int] = None,
    return_zs: bool = False,
):
    """Bins into discrete azimuthal regions"""
    return get_bins(df, value="z", vmin=zmin, vmax=zmax, dval=dz, nval=nz, return_vals=return_zs)


def get_bins(
    df: pd.DataFrame,
    value: str = "r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dval: Optional[float] = None,
    nval: Optional[int] = 100,
    return_vals: bool = False,
):
    """Bins into discrete regions"""
    if value not in df.columns:
        if value == "phi":
            df["phi"] = np.arctan2(df.y, df.x)
        elif value == "r":
            df["r"] = np.sqrt(df.x**2 + df.y**2)
    vals = np.linspace(vmin, vmax, nval) if nval is not None else np.arange(vmin, vmax, dval)
    dval = dval if dval is not None else np.abs(dval[1] - dval[0])
    # Define the edges of the bins
    bin_edges = np.append(
        vals - dval / 2,
        vals[-1] + dval / 2,
    )
    # Assign each particle to a radial bin
    df[f"{value}_bin"] = pd.cut(df[value], bins=bin_edges, labels=vals, include_lowest=True)

    if return_vals:
        return df, vals
    return df


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
