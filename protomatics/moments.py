from typing import Optional

import bettermoments as bm
import numpy as np
import scipy
from astropy.io import fits

from .constants import G, Msol_kg, au
from .helpers import get_vels_from_freq
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


def calculate_keplerian_moment1(
    r_min: Optional[float] = 0.0,
    r_max: Optional[float] = 300.0,
    num_r: Optional[int] = None,
    M_star: float = 1.0,
    inc: float = 20.0,
    distance: float = 200.0,
    hdu: Optional[list] = None,
    rotate: float = 0.0,
) -> np.ndarray:
    """
    This calculates the moment-1 map of a Keplerian disk with
    a given star mass (solar masses) and inclination (degrees) and distance (pc)
    If an hdu is given, the grid is made using WCS
    Assumes a square image
    """

    # avoid circular imports
    from .analysis import make_grids

    gr, gphi, _, _ = make_grids(
        hdu,
        r_min=r_min,
        r_max=r_max,
        num_r=num_r,
        distance=distance,
    )

    # add rotation to azimuth (default = 0)
    gphi += rotate

    # calculate Keplerian moment
    moment1 = (
        np.sqrt(G * M_star * Msol_kg / (gr * au)) * np.cos(gphi) * np.sin(inc * np.pi / 180.0)
    )
    moment1 *= 1e-3  # convert to km/s

    return moment1


def prepare_moment_data(
    fits_path: str,
    vel_min: Optional[float] = None,
    vel_max: Optional[float] = None,
    sub_cont: bool = True,
) -> tuple:
    """Prepares data for making moments"""

    data, velax = bm.load_cube(fits_path)

    # convert to km/s if it's in Hz
    hdu = fits.open(fits_path)
    if "CUNIT3" in hdu[0].header and "Hz" in hdu[0].header["CUNIT3"]:
        velax = get_vels_from_freq(hdu, relative=True, syst_chan=hdu[0].header["CRPIX3"])

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

    return masked_data, velax, rms


def make_moments(
    fits_path: str,
    which_moments: tuple = (0, 1, 2),
    vel_min: Optional[float] = None,
    vel_max: Optional[float] = None,
    sub_cont: bool = True,
    save_moments: bool = False,
    masked_data: Optional[np.ndarray] = None,
    velax: Optional[np.ndarray] = None,
    rms: Optional[np.ndarray] = None,
    outname: Optional[str] = None,
) -> tuple:
    """Calculates moments for a given fits file between a give velocity range"""

    # get data if not provided
    if masked_data is None:
        masked_data, velax, rms = prepare_moment_data(
            fits_path,
            vel_min=vel_min,
            vel_max=vel_max,
            sub_cont=sub_cont,
        )

    # calculate all moments, each is returned as a tuple with two entries
    # the first entry is the moment map and the second is the uncertainty map
    print(f"which moments: {which_moments}")
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
                outname=outname,
            )

    # now we split into moments and uncertainties (save_to_FITS needs both, so we don't split before then)
    calc_uncertainties = {i: calc_moments[i][1] for i in calc_moments}
    calc_moments = {i: calc_moments[i][0] for i in calc_moments}

    return calc_moments, calc_uncertainties


def make_masked_moments(
    fits_path: str,
    which_moments: tuple = (0, 1, 2),
    vel_tol: float = 0.5,
    vel_max: Optional[float] = None,
    vel_min: Optional[float] = None,
    sub_cont: bool = True,
    distance: float = 200.0,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    num_r: Optional[int] = None,
    M_star: float = 1.0,
    inc: float = 20.0,
    save_moments: bool = False,
    rotate: float = 0.0,
) -> tuple:
    """This gets the Keplerian and non-Keplerian components of the data and calculates moments"""

    # avoid circular imports
    from .analysis import mask_keplerian_velocity

    # split the data
    kep_data, non_kep_data, velax = mask_keplerian_velocity(
        fits_path,
        vel_tol=vel_tol,
        distance=distance,
        inc=inc,
        M_star=M_star,
        sub_cont=sub_cont,
        num_r=num_r,
        r_min=r_min,
        r_max=r_max,
        rotate=rotate,
    )

    # estimate RMS
    kep_rms = bm.estimate_RMS(data=kep_data, N=5)
    non_kep_rms = bm.estimate_RMS(data=non_kep_data, N=5)

    # make moments using masked data
    kep_moments, kep_uncertainties = make_moments(
        fits_path,
        which_moments=which_moments,
        save_moments=save_moments,
        masked_data=kep_data,
        velax=velax,
        rms=kep_rms,
        vel_min=vel_min,
        vel_max=vel_max,
        outname="keplerian",
    )
    non_kep_moments, non_kep_uncertainties = make_moments(
        fits_path,
        which_moments=which_moments,
        save_moments=save_moments,
        masked_data=non_kep_data,
        velax=velax,
        rms=non_kep_rms,
        vel_min=vel_min,
        vel_max=vel_max,
        outname="non_keplerian",
    )

    return kep_moments, kep_uncertainties, non_kep_moments, non_kep_uncertainties


def plot_moments(
    calc_moments: Optional[dict] = None,
    fits_path: Optional[str] = None,
    which_moments: tuple = (0, 1),
    vel_min: Optional[float] = None,
    vel_max: Optional[float] = None,
    sub_cont: bool = True,
    sub_kep_moment: bool = False,
    save: bool = False,
    save_name: str = "",
    plot_zero: bool = False,
    M_star: float = 1.0,
    inc: float = 20.0,
    distance: float = 200.0,
    show: bool = True,
    vmaxes: Optional[dict] = None,
    vmins: Optional[dict] = None,
    scale_data: float = 1.0,
    scale_kep_data: float = 1.0,
    mask_values: Optional[dict] = None,
    rotate: float = 0.0,
    return_moments: bool = False,
    **kwargs,
) -> None:
    assert calc_moments is not None or fits_path is not None, "Nothing to plot!"

    # calculate moments if we haven't already
    if calc_moments is None:
        calc_moments, _ = make_moments(
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
                0.0,
                0.0,
                0.0,
                M_star=M_star,
                inc=inc,
                distance=distance,
                hdu=hdu,
                rotate=rotate,
            )

        vmax = (None if moment not in vmaxes else vmaxes[moment]) if vmaxes is not None else None
        vmin = (None if moment not in vmins else vmins[moment]) if vmins is not None else None

        # mask a range of low values to 0 (helpful for delta Keplerian)
        if mask_values is None or (type(mask_values) == dict and moment not in mask_values):
            mask_value = None
        else:
            mask_value = mask_values[moment]

        plot_wcs_data(
            hdu,
            plot_data=calc_moments[moment] * scale_data,
            contour_value=0 if plot_zero else None,
            save=save,
            save_name=save_name,
            subtract_data=kep_moment * scale_kep_data if sub_kep_moment and moment == 1 else None,
            vmin=vmin,
            vmax=vmax,
            plot_cmap="RdBu_r" if moment % 2 == 1 else "magma",
            plot_units=moment_units[moment],
            show=show,
            mask_value=mask_value,
            **kwargs,
        )

    if return_moments:
        return calc_moments
    return None


def get_pv_curve(
    moment: np.ndarray,
) -> tuple:
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
) -> tuple:
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
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    rmin: Optional[float] = None,
    rmax: Optional[float] = None,
) -> tuple:
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
