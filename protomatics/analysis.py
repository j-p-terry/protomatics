from typing import Optional, Union

import bettermoments as bm
import h5py
import numpy as np
import pandas as pd
import sarracen as sn
from astropy.io import fits
from scipy.interpolate import griddata

from .constants import au_pc
from .data import make_hdf5_dataframe
from .helpers import get_azimuthal_average
from .plotting import basic_image_plot, plot_wcs_data
from .rendering import sph_smoothing

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

    # physical size of each pixel in au
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
    r_tol: float = 0.0,
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
        mask = np.abs(r_grid - r) <= r_tol
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
    rotate: float = 0.0,
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
        rotate=rotate,
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
    return_curves: bool = False,
):
    """
    This gets the amplitude of a curve relative to some reference curve.
    Can be done via integration or simply the standard deviation.
    If vel_is_zero then it simple takes the refence curve to be +- pi/2
    """

    # signed distances
    dists = rs.copy() * np.sign(phis.copy())

    # make systemic channel minor axis
    if vel_is_zero and ref_rs is None:
        np.sign(dists.copy()) * np.pi / 2.0
        ref_rs = rs.copy()
    elif ref_rs is None:
        print("No reference curve! Amplitude is zero!")
        return 0.0, [], [] if return_curves else 0.0

    if wiggle_rmax is None:
        wiggle_rmax = np.max(ref_rs)
    if rmin is None:
        rmin = 1.0
    if rmax is None:
        rmax = np.max(ref_rs)

    # can just use the standard deviation of wiggle
    # select right radial range
    okay = np.where((np.abs(rs) < wiggle_rmax) & (np.abs(rs) > rmin))
    used_phis = phis[okay]
    used_rs = rs[okay]
    # extract x-component
    used_xs = used_rs * np.cos(used_phis)
    # try to subtract reference curve if possible
    amp = np.std(used_xs)
    if return_curves:
        return amp, used_rs, used_phis
    return amp


def make_interpolated_grid(
    dataframe: Optional[pd.DataFrame] = None,
    grid_size: int = 600,
    interpolate_value: str = "vphi",
    file_path: Optional[str] = None,
    extra_file_keys: Optional[list] = None,
    return_grids: bool = False,
    xaxis: str = "x",
    yaxis: str = "y",
    interpolation_method: str = "linear",
    zmax: float = 6.0,
    nz: float = 100.0,
) -> Union[np.ndarray, tuple]:
    """Makes an interpolated grid of a given value in a dataframe
    interpolation_method is ["linear", "nearest", or "cubic"]
    """

    assert (
        dataframe is not None or file_path is not None
    ), "No data! Provide dataframe or path to hdf5 file"

    # load dataframe if not already given
    if dataframe is None:
        dataframe = make_hdf5_dataframe(file_path, extra_file_keys=extra_file_keys)

    # make sure it's in there
    assert interpolate_value in dataframe.columns, "Data not in dataframe!"

    rmax = np.max([np.ceil(np.max(dataframe[xaxis])), np.ceil(np.max(dataframe[yaxis]))])
    rmin = np.min([np.ceil(np.min(dataframe[xaxis])), np.ceil(np.min(dataframe[yaxis]))])

    # make grid of disk
    gr, gphi, gx, gy = make_grids(r_min=rmin, r_max=rmax, num_r=grid_size)

    # Interpolate using griddata
    if interpolation_method != "sph":
        interpolated_grid = griddata(
            (dataframe[xaxis].to_numpy(), dataframe[yaxis].to_numpy()),
            dataframe[interpolate_value].to_numpy(),
            (gx, gy),
            method=interpolation_method,
            fill_value=0.0,
        )
    else:
        if xaxis == "x" and yaxis == "y":
            zaxis = "z"
        elif xaxis == "x":
            zaxis = "y"
        else:
            zaxis = "x"

        _, _, interpolated_grid = sph_smoothing(
            dataframe,
            interpolate_value,
            (np.min(gx), np.max(gx)),
            (np.min(gy), np.max(gy)),
            integrate=False,
            nx=grid_size,
            ny=grid_size,
            x_axis=xaxis,
            y_axis=yaxis,
            zmax=zmax,
            nz=nz,
            z_axis=zaxis,
        )
    if not return_grids:
        return interpolated_grid
    return interpolated_grid, (gr, gphi, gx, gy)


def calculate_doppler_flip(
    hdf5_path: str,
    grid_size: int = 600,
    plot: bool = False,
    save_plot: bool = False,
    save_name: str = "",
    xlabel: str = "x [au]",
    ylabel: str = "y [au]",
    cbar_label: str = r"$v_{\phi} - \langle v_{\phi} \rangle$ [km s$^{-1}$]",
    show_plot: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    put_in_kms: bool = True,
    r_tol: float = 0.0,
) -> np.ndarray:
    """
    Calculates the doppler flip of a disk given an HDF5 output
    Returns the doppler flip map, the phi velocity field, and the azimuthally averaged vphi
    """

    vphi, grids = make_interpolated_grid(
        dataframe=None,
        grid_size=grid_size,
        interpolate_value="vphi",
        file_path=hdf5_path,
        return_grids=True,
    )

    _, avg_vphi_map = calc_azimuthal_average(vphi, r_grid=grids[0], r_tol=r_tol)

    # get code units
    if put_in_kms:
        # read in file
        units = get_code_units(hdf5_path)
        utime, udist = units["utime"], units["udist"]
        uvel = udist / utime
        vphi *= uvel  # cm/s
        avg_vphi_map *= uvel
        vphi *= 1e-5  # km/s
        avg_vphi_map *= 1e-5

    doppler_flip = vphi.copy() - avg_vphi_map.copy()

    if plot:
        basic_image_plot(
            doppler_flip,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            save=save_plot,
            save_name=save_name,
            show=show_plot,
            vmin=vmin,
            vmax=vmax,
            plot_cmap="RdBu_r",
        )

    return doppler_flip, vphi, avg_vphi_map


def get_code_units(hdf5_path: str, extra_values: Optional[tuple] = None) -> dict:
    """Gets the code units from a simulation"""

    # read in file
    file = h5py.File(hdf5_path, "r")

    umass = file["header/umass"][()]  ## M_sol in grams
    utime = file["header/utime"][()]  ## time such that G = 1
    udist = file["header/udist"][()]  ## au in cm

    units = {"umass": umass, "udist": udist, "utime": utime}

    if extra_values is None:
        return units

    # can also get other information (like gamma)
    for val in extra_values:
        if val in file["header"]:
            units[val] = file[f"header/{val}"][()]

    return units


def calculate_fourier_amps(
    r_min: float,
    r_max: float,
    modes: tuple = (1, 2, 3),
    hdf5_df: Optional[pd.DataFrame] = None,
    hdf5_path: Optional[str] = None,
) -> dict:
    """Calculates the fourier mode within a radial range according to Eq 12 of Hall (2019)"""

    assert (
        hdf5_df is not None or hdf5_path is not None
    ), "No data! Provide dataframe or path to hdf5 file"

    if hdf5_df is None:
        hdf5_df = make_hdf5_dataframe(hdf5_path)

    # trim to correct radial range
    hdf5_df = hdf5_df[(hdf5_df["r"] < r_max) & (hdf5_df["r"] > r_min)]

    # get number of particles
    N = len(hdf5_df)

    amps = {mode: 0.0 for mode in modes}

    # go over each mode
    for mode in modes:
        # get phase for each particle
        hdf5_df["exp_m_phi"] = np.exp(-1.0j * mode * hdf5_df["phi"])
        coeffs = hdf5_df.exp_m_phi.to_numpy()

        amps[mode] = abs(np.sum(coeffs)) / N

    return amps


def get_annulus(
    sdf: sn.SarracenDataFrame,
    r_annulus: float,
    dr: float = 0.5,
) -> sn.SarracenDataFrame:
    """Returns a dataframe with data between r - 0.5 dr -> r + 0.5 dr"""
    return sdf[(sdf.r < r_annulus + 0.5 * dr) & (sdf.r > r_annulus - 0.5 * dr)]


def get_annulus_Sigma(
    M_annulus: float,
    r_annulus: float,
    dr: float = 0.5,
) -> float:
    """Simga(r) = M_enc_annulus / pi[(r + 0.5 dr)^2 - (r - 0.5dr)^2]"""

    return M_annulus / (np.pi * ((r_annulus + 0.5 * dr) ** 2.0 - (r_annulus - 0.5 * dr) ** 2.0))


def get_cs_sq(sdf: sn.SarracenDataFrame, gamma: float = 5.0 / 3.0) -> np.ndarray:
    """Gets the square of the sound speed"""

    return (
        (2.0 / 3.0) * sdf.u.to_numpy() if gamma == 1 else (gamma - 1.0) * gamma * sdf.u.to_numpy()
    )


def get_annulus_toomre(
    sdf: sn.SarracenDataFrame,
    r_annulus: float,
    dr: float = 0.5,
    mass: float = 1.0,
    G_: float = 1.0,
    gamma: float = 5.0 / 3.0,
    convert: bool = False,
    return_intermediate_values: bool = False,
) -> Union[dict, float]:
    """Gets Q according to
    Q = cs_rms * Omega / pi Sigma
    where Simga(r) = M_enc_annulus / pi[(r + 0.5 dr)^2 - (r - 0.5dr)^2]
    and
    cs_rms^2 = 2/3u (gamma = 1)
           = (gamma - 1) gamma u (gamma != 1)
    """
    sdf["r"] = np.sqrt(sdf.x**2.0 + sdf.y**2.0)

    # find everything inside that annulus
    sdf = get_annulus(sdf, r_annulus, dr=dr)

    # Get enclosed mass
    sdf.create_mass_column()
    M_annulus = np.sum(sdf.m)

    # Get surface density
    Sigma = get_annulus_Sigma(M_annulus, r_annulus, dr=dr)

    # Add rms sound speed
    crms_sq = get_cs_sq(sdf, gamma=gamma)
    if convert:
        mass *= sdf.params["umass"]
        r_annulus *= sdf.params["udist"]
        crms_sq *= (sdf.params["udist"] / sdf.params["utime"]) ** 2.0
        G_ *= (sdf.params["udist"] ** 3.0) / ((sdf.params["utime"] ** 2.0) * sdf.params["umass"])
        Sigma *= sdf.params["umass"] / (sdf.params["udist"] ** 2.0)

    crms = np.sqrt(np.mean(crms_sq))

    # Keplerian frequency
    Omega = np.sqrt(G_ * mass / r_annulus**3.0)

    if return_intermediate_values:
        return {
            "Q": crms * Omega / (np.pi * Sigma),
            "Sigma": Sigma,
            "cs_rms": crms,
            "Omega": Omega,
        }

    return crms * Omega / (np.pi * Sigma)


def compute_local_surface_density(
    sdf: pd.DataFrame,
    dr: float = 0.25,
    dphi: float = np.pi / 18,
    usdense: Optional[float] = None,
    particle_mass: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the local vertically integrated surface density from SPH particle data.
    Recommended to only use a copy of the input data frame because there are some
    sneaky inline operations that can mess things up

    Parameters:
        sdf (: sn.SarracenDataFrame): sn.Dataframe containing SPH particle data with columns:
                           'x', 'y', 'z' with parameters "mass" and "umass" and "udist"
        dr (float): Radial bin width for grouping particles (default: 1.0).
        dphi (float): Azimuthal bin width in radians (default: π/18 or 10 degrees).

    Returns:
        numpy array of surface density in CGS units
    """
    if usdense is None:
        usdense = sdf.params["umass"] / (sdf.params["udist"] ** 2)
    if particle_mass is None:
        particle_mass = sdf.params["mass"]
    cols = sdf.columns
    if "r" not in cols:
        sdf["r"] = np.sqrt(sdf["x"] ** 2.0 + sdf["y"] ** 2.0)
    if "phi" not in cols:
        sdf["phi"] = np.arctan2(sdf["y"], sdf["x"])
    # if "rho" not in cols:
    # sdf.calc_density()

    # Assign particles to radial and azimuthal bins
    sdf["r_bin"] = (sdf["r"] // dr) * dr  # Floor to nearest radial bin
    sdf["phi_bin"] = (sdf["phi"] // dphi) * dphi  # Floor to nearest azimuthal bin
    sdf["mass"] = particle_mass * np.ones_like(sdf["r"])

    # Compute local surface density by summing mass / area for each (R_bin, phi_bin)
    def compute_bin_surface_density(group):
        R_bin = group["r_bin"].iloc[0]
        area = dr * R_bin * dphi  # Area of the bin in polar coordinates
        return group["mass"].sum() / area

    surface_density = (
        sdf.groupby(["r_bin", "phi_bin"])
        .apply(compute_bin_surface_density)
        .reset_index(name="sigma")
    )
    sdf = sdf.copy().merge(surface_density, on=["r_bin", "phi_bin"], how="left")
    return sdf["sigma"].to_numpy() * usdense


def mdot_to_Bphi(
    M: float,
    mdot: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    f: float = 50.0,
    L_factor: float = 6.0,
    use_mdot_abs: bool = True,
) -> Union[float, np.ndarray]:
    """Gets toroidal magnetic field (G)
    If radial transport dominates
    Mdot in Ms/yr
    R in au
    M in Msun
    Weiss et al. (2021) Eq 2
    Optionally use absolute value of mdot (then add sign of mdot)
    """
    if not use_mdot_abs:
        return (
            0.72
            * (M ** (0.25))
            * ((mdot * 1e8) ** (0.5))
            * ((f / L_factor) ** (0.5))
            * (R ** (-11.0 / 8.0))
        )
    return (
        0.72
        * (M ** (0.25))
        * ((np.abs(mdot) * 1e8) ** (0.5))
        * ((f / L_factor) ** (0.5))
        * (R ** (-11.0 / 8.0))
        * np.sign(mdot)
    )


def vr_to_mdot(
    R: Union[float, np.ndarray],
    Sigma: Union[float, np.ndarray],
    vr: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Accretion rate from accretion velocity
    in g/s
    Accretion Power in Astrophysics Eq 5.14
    Same as Eq 15 in Wardle 2007
    """

    return 2.0 * np.pi * R * Sigma * (-vr)


def get_az_avg_Sigma(
    df: pd.DataFrame,
    dr: float = 0.25,
    dphi: float = np.pi / 20.0,
    particle_mass: Optional[float] = None,
    usdense: Optional[float] = None,
    rmin: Optional[float] = None,
    rmax: Optional[float] = None,
) -> pd.DataFrame:
    # Ensure r is present
    if "r" not in df.columns:
        df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)

    # Ensure sigma is present
    if "sigma" not in df.columns:
        # 'compute_local_surface_density' presumably returns an array of values, one per particle
        df["sigma"] = compute_local_surface_density(
            df.copy(), dr=dr, dphi=dphi, particle_mass=particle_mass, usdense=usdense
        )

    return get_azimuthal_average(df, "sigma", dr=dr, rmin=rmin, rmax=rmax)


def get_dSigma_Sigma(
    df: pd.DataFrame,
    dr: float = 0.25,
    dphi: float = np.pi / 20.0,
    particle_mass: Optional[float] = None,
    usdense: Optional[float] = None,
    rmin: Optional[float] = None,
    rmax: Optional[float] = None,
):
    if particle_mass is None and "mass" in df.columns:
        particle_mass = df["mass"].to_numpy()[0]

    df = get_az_avg_Sigma(
        df, dr=dr, dphi=dphi, particle_mass=particle_mass, usdense=usdense, rmin=rmin, rmax=rmax
    )

    df["dsigma_sigma"] = (df.sigma - df.avg_sigma) / df.sigma

    return df
