from typing import Optional

import bettermoments as bm
import numpy as np
from astopy.io import fits

from .plotting import plot_wcs_data

##############################################################
##############################################################
##                                                          ##
##          This program contains the necessary functions   ##
##          for various analyses of interest                ##
##                                                          ##
##############################################################
##############################################################


def make_peak_vel_map(
    fits_path: str,
    vel_max: Optional[float] = None,
    vel_min: Optional[float] = None,
    line_index: int = 1,
    sub_cont: bool = True,
    plot: bool = False,
    save: bool = False,
    save_name: str = "",
) -> np.ndarray:
    """Makes a map of the peak velocity at each pixel"""

    data, velax = bm.load_cube(fits_path)
    # get rid of any axes with dim = 1
    data = data.squeeze()
    # get the proper emission line
    if len(data.shape) == 4:
        data = data[line_index, :, :, :]

    if sub_cont:
        # subtract continuum
        data[:] -= 0.5 * (data[0] + data[-1])

    # get channel limits
    first_channel = np.argmin(np.abs(velax - vel_min)) if vel_max is not None else 0
    last_channel = np.argmin(np.abs(velax - vel_max)) if vel_max is not None else len(velax)

    # trim data
    data = data[first_channel:last_channel, :, :]
    velax = velax[first_channel:last_channel]

    # the peak map is the velocity with the most intensity
    peak_map = velax[np.argmax(data, axis=0)]

    if plot:
        hdu = fits.open(fits_path)
        plot_wcs_data(
            hdu,
            fits_path=fits_path,
            plot_data=peak_map,
            plot_cmap="RdBu_r",
            save=save,
            save_name=save_name,
        )

    return peak_map


def calc_azimuthal_average(
    data: np.ndarray,
    r_grid: Optional[np.ndarray] = None,
) -> tuple:
    """Calculates the azimuthal average of data"""

    # use pixels instead of physical distances
    if r_grid is None:
        middle = data.shape[0] // 2
        xs = np.array([i - middle for i in range(data.shape[0])])
        # turn into x and y grids
        gx = np.tile(xs, (data.shape[0], 1))
        gy = np.tile(xs, (data.shape[0], 1)).T

        # turn into r grid
        r_grid = np.sqrt(gx**2 + gy**2)

    # make radii integers in order to offer some finite resolution
    r_grid = r_grid.copy().astype(np.int)

    # Extract unique radii and skip as needed
    rs = np.unique(r_grid)

    az_averages = {}
    # mask the moment where everything isn't at a given radius and take the mean
    for r in rs:
        mask = r_grid == r
        az_averages[r] = np.mean(data[mask]) if np.any(mask) else 0

    # Map the averages to the original shape
    az_avg_map = np.zeros_like(data)
    for r, avg in az_averages.items():
        az_avg_map[r_grid == r] = avg

    return az_averages, az_avg_map
