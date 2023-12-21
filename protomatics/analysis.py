from typing import Optional, Union

import bettermoments as bm
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import griddata

from .constants import au_pc
from .helpers import cylindrical_to_cartesian
from .plotting import basic_image_plot, plot_wcs_data

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


def make_ev_dataframe(file_path: str) -> pd.DataFrame:
    """Reads in a PHANTOM .ev file and returns a pandas dataframe"""

    # load the data
    ev_df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1)

    # get the column names
    with open(file_path) as f:
        line = f.readline()

    # PHANTOM ev files start with # and columns are bracketed with [...]
    header_ = line.split("[")[1:]
    header = []
    for x in header_:
        y = x.split()[1:]
        name = ""
        while len(y) > 0:
            name += y[0]
            name += "_"
            y = y[1:]
        # column ends with ] and there's an extra _
        name = name[:-2]
        header.append(name)

    # assign header to dataframe
    ev_df.columns = header

    return ev_df


def make_hdf5_dataframe(
    file_path: str,
    extra_file_keys: Optional[list] = None,
) -> pd.DataFrame:
    """Reads an HDF5 file and returns a dataframe with the variables in file_keys"""

    # read in file
    file = h5py.File(file_path, "r")

    # basic information that is always loaded
    basic_keys = ["x", "y", "z", "vz", "vy", "vz", "r", "phi", "vr", "vphi"]

    # initialize dataframe
    hdf5_df = pd.DataFrame(columns=basic_keys)

    # make basic information
    xyzs = file["particles/xyz"][:]
    vxyzs = file["particles/vxyz"][:]
    hdf5_df["x"] = xyzs[:, 0]
    hdf5_df["y"] = xyzs[:, 1]
    hdf5_df["z"] = xyzs[:, 2]
    hdf5_df["r"] = np.sqrt(hdf5_df.x**2 + hdf5_df.y**2)
    hdf5_df["phi"] = np.arctan2(hdf5_df.y, hdf5_df.x)
    hdf5_df["vx"] = vxyzs[:, 0]
    hdf5_df["vy"] = vxyzs[:, 1]
    hdf5_df["vz"] = vxyzs[:, 2]
    hdf5_df["vphi"] = -hdf5_df.vx * np.sin(hdf5_df.phi) + hdf5_df.vy * np.cos(hdf5_df.phi)
    hdf5_df["vr"] = hdf5_df.vx * np.cos(hdf5_df.phi) + hdf5_df.vy * np.sin(hdf5_df.phi)

    # add any extra information if you want and can
    if extra_file_keys is not None:
        for key in extra_file_keys:
            # don't get a value we've already used
            if key in hdf5_df.columns:
                continue
            # can also grab sink information
            if key in file["sinks"] and key not in file["particles"].keys():
                for i in range(len(file[f"sinks/{key}"])):
                    # sink values are a scalar, so we repeat over the entire dataframe
                    hdf5_df[f"{key}_{i}"] = np.repeat(file[f"sinks/{key}"][i], hdf5_df.shape[0])
                    continue
            # might be in header
            elif (
                key in file["header"]
                and key not in file["particles"].keys()
                and key not in hdf5_df.columns
            ):
                for i in range(len(file[f"header/{key}"])):
                    # sink values are a scalar, so we repeat over the entire dataframe
                    hdf5_df[f"{key}_{i}"] = np.repeat(file[f"header/{key}"][i], hdf5_df.shape[0])
                    continue
            # value isn't anywhere
            if key not in file["particles"].keys():
                continue
            # only add if each entry is a scalar
            if len(file[f"particles/{key}"][:].shape) == 1:
                hdf5_df[key] = file[f"particles/{key}"][:]
            # if looking for components
            if key == "Bxyz":
                bxyzs = file["particles/Bxyz"][:]
                hdf5_df["Bx"] = bxyzs[:, 0]
                hdf5_df["By"] = bxyzs[:, 1]
                hdf5_df["Bz"] = bxyzs[:, 2]
                hdf5_df["Br"] = hdf5_df.Bx * np.cos(hdf5_df.phi) + hdf5_df.By * np.sin(
                    hdf5_df.phi
                )
                hdf5_df["Bphi"] = -hdf5_df.Bx * np.sin(hdf5_df.phi) + hdf5_df.By * np.cos(
                    hdf5_df.phi
                )

    return hdf5_df


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
    interpolated_grid = griddata(
        (dataframe[xaxis].to_numpy(), dataframe[yaxis].to_numpy()),
        dataframe[interpolate_value].to_numpy(),
        (gx, gy),
        method=interpolation_method,
        fill_value=0.0,
    )
    if not return_grids:
        return interpolated_grid
    return interpolated_grid, (gr, gphi, gx, gy)


def get_Q_toomre(
    dataframe: pd.DataFrame,
    r_annulus: float,
    dr: float = 1.0,
    phi: float = np.pi / 2.0,
    dphi: float = np.pi / 10.0,
    gamma: float = 5.0 / 3.0,
    mass_key: str = "massoftype_0",
    az_avg: bool = False,
    m_star: Optional[float] = 1.0,
    code_units: dict = {},
    G: float = 1.0,
) -> tuple:
    """Gets Toomre Q at a given r, phi. Can optionally do azimuthal average
    Returns Q along with Sigma and the squared sound speeds.
    Uses same method as SPLASH
    """

    # get annulus
    dataframe = dataframe[np.abs(dataframe["r"] - r_annulus) <= dr]

    # get relevant azimuths
    # azimuthal average just gets every angle
    if az_avg:
        dphi = 2.0 * np.pi
        phi_df = dataframe
    else:
        phi_df = dataframe[np.abs(dataframe["phi"] - phi) <= dphi]

    # find mass within annulus
    M = np.sum(phi_df[mass_key])
    # internal energy for each particle
    ui_s = phi_df["u"].to_numpy()
    #  convert to CGS
    if "uenergy" in code_units:
        ui_s *= code_units["uenergy"]
    # get squared sound speed for each particle
    cs_sqs = 2.0 / 3.0 * ui_s.copy() if gamma == 1.0 else (gamma - 1.0) * gamma * ui_s.copy()

    # convert to CGS
    if "umass" in code_units:
        M *= code_units["umass"]

    # convert to CGS
    if "udist" in code_units:
        r_annulus *= code_units["udist"]
        dr *= code_units["udist"]

    # get RMS sound speed
    rms_cs = np.sqrt(np.mean(cs_sqs))

    # get surface density
    sigma = M / ((r_annulus + 0.5 * dr) ** 2 - (r_annulus - 0.5 * dr) ** 2)
    # account for azimuthal area contribution
    sigma /= dphi / 2.0
    # sigma /= np.pi

    # read in the star mass if it's there
    if "m_0" in dataframe.columns:
        m_star = np.unique(dataframe["m_0"].to_numpy())[0]

    # convert to CGS
    if "umass" in code_units:
        m_star *= code_units["umass"]

    # convert to CGS
    if "uG" in code_units:
        G *= code_units["uG"]

    # assume Keplerian frequency
    omega_kep = np.sqrt(G * m_star / (r_annulus**3.0))

    Q = rms_cs * omega_kep / (np.pi * sigma)

    return Q, sigma, cs_sqs


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
