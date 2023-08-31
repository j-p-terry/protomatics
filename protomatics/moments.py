import bettermoments as bm
import numpy as np
import scipy
from astopy.io import fits

from .constants import G, Msol_kg, au, au_pc
from .plotting import get_wiggle_from_contour, plot_polar_and_get_contour, plot_wcs_data

##############################################################
##############################################################
##                                                          ##
##          This program contains the necessary functions   ##
##          to calculate and plot moments                   ##
##          and extract channel curves                      ##
##                                                          ##
##############################################################
##############################################################

# bettermoments functions corresponding to their order
moment_functions = {
    0: bm.collapse_zeroth,
    1: bm.collapse_first,
    2: bm.collapse_second,
    8: bm.collapse_eighth,
    9: bm.collapse_ninth,
}

moment_names = {
    0: "zeroth",
    1: "first",
    2: "second",
    8: "eighth",
    9: "ninth",
}

moment_units = {
    0: "Normalized",
    1: r"km/s",
    2: r"(km/s)$^{2}$",
    8: r"(km/s)$^{8}$",
    9: r"(km/s)$^{9}$",
}


def get_image_physical_size(
    hdu: list,
    distance: float = 200.0,
) -> tuple(int, float):
    """Takes an hdu and converts the image into physical sizes at a given distance (pc)"""

    # angular size of each pixel
    radian_width = np.pi * abs(hdu[0].header["CDELT1"] * hdu[0].header["NAXIS1"]) / 180.0

    # physocal size of each pixel in au
    image_size = 2.0 * distance * np.tan(radian_width / 2.0) * au_pc

    npix = hdu[0].header["NAXIS1"]

    # Calculate the spatial extent
    x_max = 1.0 * (image_size / 2.0) * au

    return npix, x_max


def calculate_keplerian_moment1(
    r_min: float,
    r_max: float,
    num_r: float,
    M_star: float = 1.0,
    inc: float = 20.0,
    distance: float = 200.0,
    hdu: list | None = None,
) -> np.ndarray:
    """
    This calculates the moment-1 map of a Keplerian disk with
    a given star mass (solar masses) and inclination (degrees) and distance (pc)
    If an hdu is given, the grid is made using WCS
    Assumes a square image
    """

    # in order to calculate the moment to match an hdu's spatial extent
    if hdu is not None:
        r_max, num_r = get_image_physical_size(
            hdu,
            distance=distance,
        )
        r_min = -r_max

    # make range x range
    xs = np.linspace(r_min, r_max, num_r)

    # turn into x and y grids
    gx = np.tile(xs, (num_r, 1))
    gy = np.tile(xs, (num_r, 1)).T

    # turn into r, phi grid
    gr = np.sqrt(gx**2 + gy**2)
    gphi = np.arctan2(gy, gx)

    # calculate Keplerian moment
    moment1 = (
        np.sqrt(G * M_star * Msol_kg / (gr * au)) * np.cos(gphi) * np.sin(inc * np.pi / 180.0)
    )
    moment1 *= 1e-3  # convert to km/s

    return moment1


def calculate_moments(
    fits_path: str,
    which_moments: tuple = (0, 1),
    vel_min: float | None = None,
    vel_max: float | None = None,
    sub_cont: bool = True,
    save_moments: bool = False,
    save_path: str = "",
) -> tuple(dict, dict):
    """Calculates moments for a given fits file between a give velocity range"""

    data, velax = bm.load_cube(fits_path)

    # subtract continuum
    if sub_cont:
        data[:] -= 0.5 * (data[0] + data[-1])

    # estimate RMS
    rms = bm.estimate_RMS(data=data, N=5)

    # get channel masks
    first_channel = np.argmin(np.abs(velax - vel_min)) if vel_max is not None else 0
    last_channel = np.argmin(np.abs(velax - vel_max)) if vel_max is not None else -1

    channel_mask = bm.get_channel_mask(
        data=data,
        firstchannel=first_channel,
        lastchannel=last_channel,
    )
    masked_data = data * channel_mask

    # calculate all moments, each is returned as a tuple with two entries
    # the first entry is the moment map and the second is the uncertainty map
    calc_moments = {
        i: moment_functions[i](velax=velax, data=masked_data, rms=rms) for i in which_moments
    }

    # get rid of NaNs
    for i in which_moments:
        if np.any(np.isnan(calc_moments[i][0])):
            calc_moments[i][0][np.isnan(calc_moments[i][0])] = 0

    # optionally save
    if save_moments:
        for moment in calc_moments:
            bm.save_to_FITS(
                moments=calc_moments[moment],
                method=moment_names[moment],
                path=fits_path,
            )

    # now we split into moments and uncertainties (save_to_FITS needs both, so we don't split before then)
    calc_uncertainties = {i: calc_moments[i][1] for i in calc_moments}
    calc_moments = {i: calc_moments[i][0] for i in calc_moments}

    return calc_moments, calc_uncertainties


def plot_moments(
    calc_moments: dict | None = None,
    fits_path: str | None = None,
    which_moments: tuple = (0, 1),
    vel_min: float | None = None,
    vel_max: float | None = None,
    sub_cont: bool = True,
    sub_kep_moment: bool = False,
    save: bool = False,
    save_name: str = "",
    plot_zero: bool = False,
    M_star: float = 1.0,
    inc: float = 20.0,
    distance: float = 200.0,
) -> None:
    assert calc_moments is not None or fits_path is not None, "Nothing to plot!"

    # calculate moments if we haven't already
    if calc_moments is None:
        calc_moments, _ = calculate_moments(
            fits_path,
            which_moments=which_moments,
            vel_min=vel_min,
            vel_max=vel_max,
            sub_cont=sub_cont,
        )

    for moment in calc_moments:
        print(f"Plotting moment {moment}")

        # load the fits file to give us WCS
        hdu = fits.open(fits_path)

        if sub_kep_moment and moment == 1:
            # calculate a keplerian moment-1 map to match
            kep_moment = calculate_keplerian_moment1(
                0.0, 0.0, 0.0, M_star=M_star, inc=inc, distance=distance, hdu=hdu
            )

        plot_wcs_data(
            hdu,
            plot_data=calc_moments[moment],
            contour_value=0 if plot_zero else None,
            save=save,
            save_name=save_name,
            subtract_data=kep_moment if sub_kep_moment and moment == 1 else None,
            vmin=vel_min,
            vmax=vel_max,
            plot_cmap="RdBu_r",
            plot_units=moment_units[moment],
        )


def get_pv_curve(
    moment: np.ndarray,
) -> tuple(np.ndarray, np.ndarray):
    """Gets the postion-velocity curve down the middle of a moment-1 map"""

    middlex = moment.shape[1] // 2
    # the p-v wiggle is simply the minor axis
    pv_wiggle = moment[:, middlex]

    # the rs are just the y values since we are on the minor axis
    rs = np.array([i - moment.shape[0] // 2 for i in range(len(pv_wiggle))])

    return rs, pv_wiggle


def split_pv_curve(
    rs: np.ndarray,
    pv_wiggle: np.ndarray,
    pv_rmin: float = 0.0,
) -> tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Splits the positon-velocity curve into the positive and negative curves"""

    pos_okay = np.where(rs > pv_rmin)
    neg_okay = np.where(-rs > pv_rmin)
    okay = np.where(np.abs(rs) > pv_rmin)
    pos_pv_wiggle = pv_wiggle[pos_okay]
    neg_pv_wiggle = pv_wiggle[neg_okay]
    pos_rs = rs[pos_okay]
    neg_rs = rs[neg_okay]

    okay_rs = rs[okay]
    okay_wiggle = pv_wiggle[okay]

    return okay_rs, okay_wiggle, pos_rs, pos_pv_wiggle, neg_rs, neg_pv_wiggle


def extract_wiggle(
    moment1: np.ndarray,
    in_pv_space: bool = False,
    rotation_angle: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    rmin: float | None = None,
    rmax: float | None = None,
) -> tuple(np.ndarray, np.ndarray):
    """
    Extracts the v = 0 curve from a moment-1 map.
    This is done either in position-position space or position-velocity space.
    position-position curves are taken from extracting polar contours of v = 0 and are in polar coordinates
    position-velocity curves are taken by a slice down the middle of the moment-1 map (with an appropriate rotation in degrees)
    """

    if in_pv_space:
        # rotate the moment-1 image to align the minor axis with the center (parallel to y axis)
        if rotation_angle != 0:
            moment1 = scipy.ndimage.rotate(moment1.copy(), rotation_angle)

        return get_pv_curve(moment1)

    contour = plot_polar_and_get_contour(
        moment1, vmin=vmin, vmax=vmax, rmax=rmax, show=False, units=r"km s$^{-1}$"
    )

    return get_wiggle_from_contour(contour, rmin=rmin, rmax=rmax)
