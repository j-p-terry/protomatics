import bettermoments as bm
import numpy as np
from astopy.io import fits

from .plotting import plot_wcs_data


def make_peak_vel_map(
    fits_path: str,
    vel_max: float | None = None,
    vel_min: float | None = None,
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
