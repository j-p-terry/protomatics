from typing import Optional

import numpy as np
import pandas as pd
from numba import njit
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import gaussian_filter

from .plotting import mesh_image


def bin_data(
    df: pd.DataFrame,
    value: str,
    x_axis: str = "x",
    y_axis: str = "y",
    z_axis: str = "z",
    grid_size: int = 512,
    extent: Optional[list] = None,
    average: bool = False,
    smooth_sigma: float = 2.0,
    zlims: Optional[tuple] = None,
    smooth: bool = False,
):
    """Bins data into a 2D array and optionally performs gaussian smoothing"""
    if zlims is not None:
        df = df[(df[z_axis] > zlims[0]) & (df[z_axis] < zlims[1])]
    x = df[x_axis].to_numpy()
    y = df[y_axis].to_numpy()
    values = df[value].to_numpy()

    # Determine grid extent
    if extent is None:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
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
    if smooth:
        grid = gaussian_filter(grid, sigma=smooth_sigma)

    return x_grid, y_grid, grid


@njit
def _dimensionless_w(q):
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
def _trapz_numba(y, x):
    """
    Perform trapezoidal integration over arrays x,y with numba.
    """
    s = 0.0
    for i in range(len(y) - 1):
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return s


@njit
def _dimensionless_integrate_F(q_values, zmax=3.0, nz=100):
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
        w_vals = _dimensionless_w(r_prime)
        # integrate w over z'
        F[i] = _trapz_numba(w_vals, z_prime)
    return F


def _precompute_dimensionless_F(q_table=None, zmax=3.0, nz=100):
    """
    Precompute the dimensionless integral F(q) once.
    """
    if q_table is None:
        q_table = np.linspace(0, 1, 200)
    F = _dimensionless_integrate_F(q_table, zmax=zmax, nz=nz)
    # Create interpolation for F(q)
    # Outside q=1, F(q)=0, inside q=0..1 use linear interpolation
    F_interp = interp1d(q_table, F, kind="linear", bounds_error=False, fill_value=0.0)
    return q_table, F_interp


def _precompute_line_integrated_kernel(h_values, q_table=None, zmax=3.0, nz=100):
    """
    Precompute W_int(R,h) using the dimensionless approach.

    Steps:
    1) Compute F(q) once dimensionlessly.
    2) For each h, W_int(q,h) = (8/(pi h^2)) * F(q).
    We create an interpolation function that applies this scaling.

    This avoids repeated integration for each h.
    """
    # Compute dimensionless F once
    q_table, F_interp = _precompute_dimensionless_F(q_table=q_table, zmax=zmax, nz=nz)

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
    df: pd.DataFrame,
    value: str,
    x_bounds: Optional[tuple] = None,
    y_bounds: Optional[tuple] = None,
    nx: int = 256,
    ny: int = 256,
    integrate: bool = True,
    zmax: float = 10.0,
    nz: int = 100,
    smooth_sigma: float = 2.0,
    resmooth: bool = False,
    x_axis: str = "x",
    y_axis: str = "y",
    z_axis: str = "z",
    zlims: Optional[tuple] = None,
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
    if zlims is not None:
        df = df[(df[z_axis] > zlims[0]) & (df[z_axis] < zlims[1])]
    # Extract particle data
    x = df[x_axis].to_numpy()
    y = df[y_axis].to_numpy()
    df[z_axis].to_numpy()
    h = df["h"].to_numpy()
    v = df[value].to_numpy()

    if x_bounds is None:
        x_bounds = (np.min(x), np.max(x))

    if y_bounds is None:
        y_bounds = (np.min(y), np.max(y))

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
    W_int_interp_dict = _precompute_line_integrated_kernel(unique_h, zmax=zmax, nz=nz)

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


def render_value(
    df: pd.DataFrame,
    value: str,
    interpolate: str = "no",
    x_bounds: Optional[tuple] = None,
    y_bounds: Optional[tuple] = None,
    z_bounds: Optional[tuple] = None,
    x_axis: str = "x",
    y_axis: str = "y",
    z_axis: str = "z",
    average: bool = False,
    nx: int = 256,
    ny: int = 256,
    integrate: bool = True,
    zmax: float = 10.0,
    nz: int = 100,
    smooth_sigma: float = 2.0,
    resmooth: bool = False,
    scale_data: float = 1.0,
    log: bool = False,
    symlog: bool = False,
    linthresh: float = 1e-5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save: bool = False,
    savename: str = "mesh.pdf",
    logx: bool = False,
    logy: bool = False,
    cmap: str = "gist_heat",
    value_label: Optional[str] = None,
    units: str = "",
    x_label: str = "x",
    y_label: str = "y",
    levels: Optional[tuple] = None,
    contour_color: str = "white",
    contour_label: str = "",
    contour_cmap: str = "grays",
    show: bool = False,
    streamlines: Optional[tuple] = None,
    stream_bin_method: str = "linear",
    arrowsize: float = 2.0,
    arrowstyle: str = "->",
    arrowcolor: str = "white",
    arrowwidth: float = 1.0,
    dont_close: bool = False,
    return_image: bool = True,
    **kwargs,
):
    """Renders a value from an SPH simulation with optional smoothing (no, sph, gaussian)
    Note: integrate adds an extra spatial unit
    Streamlines is either None or a 2 entry array with the x_axis and y_axis components of the field
    """

    if interpolate == "sph":
        X, Y, data = sph_smoothing(
            df,
            value,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            nx=nx,
            ny=ny,
            nz=nz,
            integrate=integrate,
            zmax=zmax,
            smooth_sigma=smooth_sigma,
            resmooth=resmooth,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            zlims=z_bounds,
        )
    else:
        if x_bounds is not None and y_bounds is not None:
            extent = (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])
        X, Y, data = bin_data(
            df,
            value,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            grid_size=max(nx, ny),
            extent=extent,
            average=average,
            smooth_sigma=smooth_sigma,
            zlims=z_bounds,
            smooth=interpolate != "no",
        )

    data *= scale_data

    value_label = value if value_label is None else value_label
    value_label = rf"{value_label} [{units}]" if units != "" else rf"{value_label}"

    if streamlines is not None:
        stream_grids = []
        x = df[x_axis].to_numpy()
        y = df[y_axis].to_numpy()
        points = np.column_stack((x, y))
        for val in streamlines:
            vals = df[val].to_numpy()
            stream_grid = griddata(points, vals, (X, Y), method=stream_bin_method)
            stream_grid = np.nan_to_num(stream_grid, nan=0.0)
            stream_grids.append(stream_grid)
    else:
        stream_grids = None

    return mesh_image(
        X,
        Y,
        data,
        log=log,
        symlog=symlog,
        linthresh=linthresh,
        vmin=vmin,
        vmax=vmax,
        save=save,
        savename=savename,
        logx=logx,
        logy=logy,
        cmap=cmap,
        cbar_units=value_label,
        x_label=x_label,
        y_label=y_label,
        levels=levels,
        contour_color=contour_color,
        contour_cmap=contour_cmap,
        contour_label=contour_label,
        show=show,
        streamlines=stream_grids,
        arrowcolor=arrowcolor,
        arrowsize=arrowsize,
        arrowstyle=arrowstyle,
        arrowwidth=arrowwidth,
        dont_close=dont_close,
        return_image=resmooth,
        **kwargs,
    )
