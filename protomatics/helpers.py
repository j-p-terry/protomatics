import os
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from numba import njit
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

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


def gaussian_smoothing(
    df: pd.DataFrame,
    value_col: str,
    x_col: str,
    y_col: str,
    grid_size: int = 512,
    extent: Optional[list] = None,
    average: bool = False,
    smooth_sigma: float = 2.0,
    zlims: Optional[tuple] = None,
):
    """Performs gaussian smoothing of 2D or 3D data. Note that this adds a spatial dimension"""
    if zlims is not None:
        df = df[(df.z > zlims[0]) & (df.z < zlims[1])]
    x = df[x_col]
    y = df[y_col]
    values = df[value_col]

    # Determine grid extent
    if extent is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = extent

    # Create grid
    x_grid = np.linspace(xmin, xmax, grid_size)
    y_grid = np.linspace(ymin, ymax, grid_size)

    # Interpolate onto the grid using SPH-like smoothing
    grid = np.zeros((grid_size, grid_size))
    if average:
        count = np.zeros_like(grid)
    for xi, yi, vi in zip(x, y, values):
        x_idx = np.searchsorted(x_grid, xi) - 1
        y_idx = np.searchsorted(y_grid, yi) - 1
        if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
            grid[y_idx, x_idx] += vi  # Deposit the value into the grid cell
            if average:
                count[y_idx, x_idx] += 1

    if average:
        grid = np.divide(grid, count, out=np.zeros_like(grid), where=(count > 0))
    smoothed_image = gaussian_filter(grid, sigma=smooth_sigma)

    return x_grid, y_grid, smoothed_image


@njit
def dimensionless_w(q):
    """
    Dimensionless cubic spline kernel shape function w(q), without normalization.
    This is the 'w(q)' that appears in W(r,h) = (8/(pi h^3)) * w(q).

    w(q) = 1 - 6q^2 + 6q^3        for 0 <= q <= 0.5
         = 2(1-q)^3              for 0.5 < q <= 1
         = 0                     otherwise
    """
    w_vals = np.zeros_like(q)
    mask1 = (q >= 0) & (q <= 0.5)
    w_vals[mask1] = 1 - 6 * q[mask1] ** 2 + 6 * q[mask1] ** 3
    mask2 = (q > 0.5) & (q <= 1.0)
    w_vals[mask2] = 2 * (1 - q[mask2]) ** 3
    return w_vals


@njit
def trapz_numba(y, x):
    """
    Perform trapezoidal integration over arrays x,y with numba.
    """
    s = 0.0
    for i in range(len(y) - 1):
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return s


@njit
def dimensionless_integrate_F(q_values, zmax=3.0, nz=100):
    r"""
    Compute F(q) = \int w( sqrt(q^2+z'^2) ) dz' from z'=-zmax to z'=zmax.
    This is the dimensionless integral, independent of h.

    zmax chosen so that beyond zmax the kernel contribution is negligible.
    """
    z_prime = np.linspace(-zmax, zmax, nz)
    z_prime[1] - z_prime[0]
    F = np.zeros_like(q_values)
    for i, q in enumerate(q_values):
        r_prime = np.sqrt(q**2 + z_prime**2)
        w_vals = dimensionless_w(r_prime)
        # integrate w over z'
        F[i] = trapz_numba(w_vals, z_prime)
    return F


def precompute_dimensionless_F(q_table=None, zmax=3.0, nz=100):
    """
    Precompute the dimensionless integral F(q) once.
    """
    if q_table is None:
        q_table = np.linspace(0, 1, 200)
    F = dimensionless_integrate_F(q_table, zmax=zmax, nz=nz)
    # Create interpolation for F(q)
    # Outside q=1, F(q)=0, inside q=0..1 use linear interpolation
    F_interp = interp1d(q_table, F, kind="linear", bounds_error=False, fill_value=0.0)
    return q_table, F_interp


def precompute_line_integrated_kernel(h_values, q_table=None, zmax=3.0, nz=100):
    """
    Precompute W_int(R,h) using the dimensionless approach.

    Steps:
    1) Compute F(q) once dimensionlessly.
    2) For each h, W_int(q,h) = (8/(pi h^2)) * F(q).
    We create an interpolation function that applies this scaling.

    This avoids repeated integration for each h.
    """
    # Compute dimensionless F once
    q_table, F_interp = precompute_dimensionless_F(q_table=q_table, zmax=zmax, nz=nz)

    W_int_interp_dict = {}
    for h in h_values:
        # Create a lambda that given q returns scaled W_int
        # W_int(q,h) = (8/(pi h^2))*F(q)
        # We'll wrap it in an interp1d call for consistency:
        # We have F_interp(q), we just multiply the result by (8/(pi h^2))
        # To create a proper interpolation object, we do so at q_table points
        F_vals = F_interp(q_table)
        W_int_vals = (8.0 / (np.pi * h**2)) * F_vals
        interp_func = interp1d(
            q_table, W_int_vals, kind="linear", bounds_error=False, fill_value=0.0
        )
        W_int_interp_dict[h] = interp_func

    return W_int_interp_dict


def sph_smoothing(
    df,
    value,
    x_bounds,
    y_bounds,
    nx,
    ny,
    integrate=True,
    zmax=3.0,
    nz=100,
    smooth_sigma: float = 2.0,
    resmooth: bool = False,
    x_axis: str = "x",
    y_axis: str = "y",
):
    """
    Project or average SPH data onto a 2D grid using an SPH kernel.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: x, y, z, h, value
    x_bounds : (float, float)
        (xmin, xmax)
    y_bounds : (float, float)
        (ymin, ymax)
    nx, ny : int
        Grid resolution in x and y directions
    integrate : bool
        If True, integrate along z (like surface density).
        If False, produce a vertically averaged value.
    zmax : float
        Integration limit in units of h (for line integration)
    nz : int
        Number of steps in z-integration

    Note: integrate = True will add an extra spatial dimension
    Returns:
    --------
    X, Y : 2D arrays
        Meshgrid arrays for x,y coordinates
    out : 2D array
        The smoothed 2D field.
    """

    # Extract particle data
    x = df[x_axis].to_numpy()
    y = df[y_axis].to_numpy()
    h = df["h"].to_numpy()
    v = df[value].to_numpy()

    # Create output grid
    xgrid = np.linspace(x_bounds[0], x_bounds[1], nx)
    ygrid = np.linspace(y_bounds[0], y_bounds[1], ny)
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    X, Y = np.meshgrid(xgrid, ygrid, indexing="xy")
    out = np.zeros((ny, nx))
    weight = np.zeros((ny, nx))

    # Precompute kernel lookups
    # Discretize unique h if desired. For large sets, consider a cache or unique set.
    unique_h = np.unique(h)
    W_int_interp_dict = precompute_line_integrated_kernel(unique_h, zmax=zmax, nz=nz)

    # Group particles by h
    # Sort by h and then group
    order = np.argsort(h)
    x_sorted = x[order]
    y_sorted = y[order]
    h_sorted = h[order]
    v_sorted = v[order]

    # Find unique h groups
    h_vals, h_starts, h_counts = np.unique(h_sorted, return_index=True, return_counts=True)

    # Loop over h groups
    for h_val, start, count in zip(h_vals, h_starts, h_counts):
        x_h = x_sorted[start : start + count]
        y_h = y_sorted[start : start + count]
        v_h = v_sorted[start : start + count]

        W_int_interp_func = W_int_interp_dict[h_val]

        # Process all particles with this h in a loop
        for x_p, y_p, val_p in zip(x_h, y_h, v_h):
            ix_min = max(0, int((x_p - h_val - x_bounds[0]) / dx))
            ix_max = min(nx - 1, int((x_p + h_val - x_bounds[0]) / dx))
            iy_min = max(0, int((y_p - h_val - y_bounds[0]) / dy))
            iy_max = min(ny - 1, int((y_p + h_val - y_bounds[0]) / dy))

            if ix_min > ix_max or iy_min > iy_max:
                continue

            X_sub = X[iy_min : iy_max + 1, ix_min : ix_max + 1]
            Y_sub = Y[iy_min : iy_max + 1, ix_min : ix_max + 1]

            dx_block = X_sub - x_p
            dy_block = Y_sub - y_p
            R_block = np.sqrt(dx_block**2 + dy_block**2)

            Wvals = W_int_interp_func(np.clip(R_block / h_val, 0, 1))

            out[iy_min : iy_max + 1, ix_min : ix_max + 1] += val_p * Wvals
            weight[iy_min : iy_max + 1, ix_min : ix_max + 1] += Wvals

    if not integrate:
        mask = weight > 0
        out[mask] /= weight[mask]

    if resmooth:
        out = gaussian_filter(out, sigma=smooth_sigma)

    return X, Y, out
