from typing import Optional

import bettermoments as bm
import numpy as np
from astropy.io import fits

from .constants import au_pc
from .helpers import cylindrical_to_cartesian
from .plotting import plot_wcs_data

##############################################################
##############################################################
##                                                          ##
##          This program contains the necessary functions   ##
##          for various analyses of interest                ##
##                                                          ##
##############################################################
##############################################################


def get_image_physical_size(
    hdu: list,
    distance: float = 200.0,
) -> tuple:
    """Takes an hdu and converts the image into physical sizes at a given distance (pc)"""

    # angular size of each pixel
    radian_width = np.pi * abs(hdu[0].header["CDELT1"] * hdu[0].header["NAXIS1"]) / 180.0

    # physocal size of each pixel in au
    image_size = 2.0 * distance * np.tan(radian_width / 2.0) * au_pc

    npix = int(hdu[0].header["NAXIS1"])

    # Calculate the spatial extent (au)
    x_max = 1.0 * (image_size / 2.0)

    return npix, x_max


def make_grids(
    hdu: Optional[list] = None,
    r_min: Optional[float] = 0.0,
    r_max: Optional[float] = 300.0,
    num_r: Optional[int] = None,
    distance: float = 200.0,
):
    """Makes x, y, r, and phi grids for an hdu/r range at a given distance"""

    # in order to calculate the moment to match an hdu's spatial extent
    if hdu is not None:
        num_r, r_max = get_image_physical_size(
            hdu,
            distance=distance,
        )
        r_min = -r_max

    if num_r is None:
        num_r = int(r_max - r_min)

    # make range x range
    xs = np.linspace(r_min, r_max, num_r)

    # turn into x and y grids
    gx = np.tile(xs, (num_r, 1))
    gy = np.tile(xs, (num_r, 1)).T

    # turn into r, phi grid
    gr = np.sqrt(gx**2 + gy**2)
    gphi = np.arctan2(gy, gx)

    return gr, gphi, gx, gy


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

    full_data, velax = bm.load_cube(fits_path)
    # get rid of any axes with dim = 1
    data = full_data.squeeze()
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
    r_grid = r_grid.copy().astype(np.int32)

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


def mask_keplerian_velocity(
    fits_path: str,
    vel_tol: float = 0.5,
    sub_cont: bool = True,
    distance: float = 200.0,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    num_r: Optional[int] = None,
    M_star: float = 1.0,
    inc: float = 20.0,
) -> tuple:
    """
    This function creates two new data cubes: one with the velocities within some tolerance of the keplerian
    velocity at that location and another that is outside of that range (i.e, the keplerian data and non-keplerian data)
    """

    # avoid circular imports
    from .moments import calculate_keplerian_moment1

    # get cube
    data, velax = bm.load_cube(fits_path)

    # subtract continuum
    if sub_cont:
        data[:] -= 0.5 * (data[0] + data[-1])

    # use header to make position grid
    hdu = fits.open(fits_path)

    # get the keplerian moment
    kep_moment1 = calculate_keplerian_moment1(
        hdu=hdu,
        r_min=r_min,
        r_max=r_max,
        num_r=num_r,
        M_star=M_star,
        distance=distance,
        inc=inc,
    )

    # mask the data that's inside the keplerian tolerance
    keplerian_mask = np.abs(velax[:, np.newaxis, np.newaxis] - kep_moment1) < vel_tol
    # get the anti-mask
    non_keplerian_mask = ~keplerian_mask

    # eliminate all non-keplerian data
    kep_data = np.where(keplerian_mask, data, 0)
    # and the same for keplerian data
    non_kep_data = np.where(non_keplerian_mask, data, 0)

    return kep_data, non_kep_data, velax


def get_wiggle_amplitude(
    rs: list,
    phis: list,
    ref_rs: Optional[list] = None,
    ref_phis: Optional[list] = None,
    rmin: Optional[float] = None,
    rmax: Optional[float] = None,
    wiggle_rmax: Optional[float] = None,
    vel_is_zero: bool = True,
    return_diffs: bool = False,
    use_std_as_amp: bool = False,
):
    """
    This gets the amplitude of a curve relative to some reference curve.
    Can be done via integration or simply the standard deviation.
    If vel_is_zero then it simple takes the refence curve to be +- pi/2
    """

    ref_length = 0.0
    diff_length = 0.0
    diffs = []
    used_rs = []

    # signed distances
    dists = rs.copy() * np.sign(phis.copy())

    # make systemic channel minor axis
    if vel_is_zero and ref_rs is None:
        ref_phis = np.sign(dists.copy()) * np.pi / 2.0
        ref_dists = rs.copy() * np.sign(ref_phis.copy())
        ref_rs = rs.copy()
    elif ref_rs is None:
        print("No reference curve! Amplitude is zero!")
        return 0.0, [], 0.0 if return_diffs else 0.0

    ref_dists = ref_rs.copy() * np.sign(ref_phis.copy())

    if wiggle_rmax is None:
        wiggle_rmax = np.max(ref_rs)
    if rmin is None:
        rmin = 1.0
    if rmax is None:
        rmax = np.max(ref_rs)

    # can just use the standard deviation of wiggle
    if use_std_as_amp:
        # select right radial range
        okay = np.where((np.abs(ref_rs) < wiggle_rmax) & (np.abs(ref_rs) > rmin))
        used_phis = phis[okay]
        used_rs = rs[okay]
        used_ref_phis = ref_phis[okay]
        # try to subtract reference curve if possible
        amp = (
            np.std(used_phis)
            if len(used_phis) != len(used_ref_phis)
            else np.std(np.abs(used_phis - used_ref_phis))
        )
        if return_diffs:
            return amp, used_rs, used_phis - used_ref_phis
        return amp

    # otherwise, integrate along curve
    for i, ref_r in enumerate(ref_rs):
        # make sure it's in the right radius
        if (
            abs(ref_r) > wiggle_rmax
            or abs(ref_r) < rmin
            or abs(ref_r) > np.max(np.abs(rs))
            or abs(ref_r) < np.min(np.abs(rs))
        ):
            continue

        # there's no next r after the last one
        if i == len(ref_rs) - 1:
            continue

        ref_phi = ref_phis[i]
        ref_dist = ref_dists[i]

        # find closest radius
        index = np.argmin(np.abs(dists - ref_dist))
        curve_phi = phis[index]

        # convert to cartesian
        ref_x, ref_y = cylindrical_to_cartesian(ref_r, ref_phi)
        next_ref_x, next_ref_y = cylindrical_to_cartesian(ref_rs[i + 1], ref_phis[i + 1])

        # get difference
        this_diff = abs(curve_phi - ref_phi) ** 2.0
        diffs.append(this_diff)
        used_rs.append(np.sign(ref_phi) * ref_r)
        # get differential
        ds = np.sqrt((ref_x - next_ref_x) ** 2 + (ref_y - next_ref_y) ** 2)

        ref_length += ds
        diff_length += this_diff * ds

    coeff = 1

    if return_diffs:
        if ref_length == 0:
            return 0, used_rs, diffs
        return coeff * np.sqrt(diff_length / ref_length), used_rs, diffs

    if ref_length == 0:
        return 0

    return coeff * np.sqrt(diff_length / ref_length)
