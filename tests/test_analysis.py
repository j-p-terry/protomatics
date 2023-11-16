import sys
from os.path import abspath, dirname

import pytest
from astropy.io import fits

# Add the parent directory to the Python path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from protomatics.analysis import (
    calc_azimuthal_average,
    calculate_doppler_flip,
    calculate_fourier_amps,
    get_code_units,
    get_Q_toomre,
    get_wiggle_amplitude,
    make_ev_dataframe,
    make_grids,
    make_hdf5_dataframe,
    make_interpolated_grid,
    make_peak_vel_map,
)
from protomatics.moments import calculate_keplerian_moment1, extract_wiggle, make_moments


@pytest.mark.parametrize("fits_name", ["test_3d_cube.fits", "test_6d_cube.fits"])
def test_make_grid(fits_name):
    """Tests if a wcs plot can be made with a varying number of input dimensions"""

    print(f"Testing grid making with {fits_name}")

    # plot with preloaded
    path = f"./tests/data/{fits_name}"

    hdu = fits.open(path)

    # test with loaded hdu
    print("Testing grid with HDU")
    _ = make_grids(hdu)

    # test in pixel space
    print("Testing grid without HDU")
    _ = make_grids()

    print("Passed!")


@pytest.mark.parametrize("fits_name", ["test_3d_cube.fits", "test_6d_cube.fits"])
def test_azimuthal_average(fits_name):
    """Tests calculating the azimuthal average of a moment-1 map"""

    print(f"Testing azimuthal average with {fits_name}")

    # plot with preloaded
    path = f"./tests/data/{fits_name}"

    hdu = fits.open(path)

    calc_moments, _ = make_moments(path, which_moments=[1])
    moment1 = calc_moments[1]

    # test without grid
    print("Azimuthal average without grid")
    _ = calc_azimuthal_average(moment1)

    # test with grid
    gr, _, _, _ = make_grids(hdu)
    print("Azimuthal average with grid")
    _ = calc_azimuthal_average(moment1, r_grid=gr, r_tol=2.0)

    print("Passed!")


@pytest.mark.parametrize("fits_name", ["test_3d_cube.fits", "test_6d_cube.fits"])
def test_peak_vel_map(fits_name):
    """Tests making a peak velocity map"""

    print(f"Testing peak velocity map with {fits_name}")

    # plot with preloaded
    path = f"./tests/data/{fits_name}"

    _ = make_peak_vel_map(path)

    print("Passed!")


@pytest.mark.parametrize("fits_name", ["test_3d_cube.fits", "test_6d_cube.fits"])
def test_wiggle_amplitude(fits_name):
    """Tests functions regarding extracting and analyzing wiggle"""

    print(f"Testing wiggle ampltidue with {fits_name}")

    # plot with preloaded
    path = f"./tests/data/{fits_name}"

    print("Calculating moments")
    calc_moments, _ = make_moments(path, which_moments=[1])

    print("Extracting wiggle")
    rs, phis = extract_wiggle(calc_moments[1], in_pv_space=False)

    print("Getting ampltidue without reference curve")
    _ = get_wiggle_amplitude(rs, phis, vel_is_zero=True)

    print("Getting standard deviation without reference curve")
    _ = get_wiggle_amplitude(rs, phis, vel_is_zero=True, use_std_as_amp=True)

    print("Getting ampltidue with identical reference curve")
    _ = get_wiggle_amplitude(rs, phis, ref_rs=rs, ref_phis=phis, vel_is_zero=False)

    print("Getting standard deviation with identical reference curve")
    _ = get_wiggle_amplitude(
        rs, phis, ref_rs=rs, ref_phis=phis, vel_is_zero=False, use_std_as_amp=True
    )

    # calculate with rotation
    kep_moment1 = calculate_keplerian_moment1(hdu=fits.open(path), rotate=0.785)

    # calculate without rotation
    kep_moment1 = calculate_keplerian_moment1(hdu=fits.open(path))

    kep_rs, kep_phis = extract_wiggle(kep_moment1, in_pv_space=False)

    print("Getting ampltidue with keplerian reference curve")
    _ = get_wiggle_amplitude(rs, phis, ref_rs=kep_rs, ref_phis=kep_phis, vel_is_zero=False)

    print("Getting standard deviation with keplerian reference curve")
    _ = get_wiggle_amplitude(
        rs, phis, ref_rs=kep_rs, ref_phis=kep_phis, vel_is_zero=False, use_std_as_amp=True
    )

    print("Passed!")


@pytest.mark.parametrize("hdf5_name", ["test_hdf5.h5"])
def test_hdf5(hdf5_name):
    """Tests calculating the doppler flip"""

    path = f"./tests/data/{hdf5_name}"

    print("Loading dataframe")
    _ = make_hdf5_dataframe(path)
    print("Loading with extra information")
    hdf5_df = make_hdf5_dataframe(
        path,
        extra_file_keys=["h", "dt", "eta_AD", "Bxyz"],
    )
    print("Loading with sink/header information")
    hdf5_df = make_hdf5_dataframe(
        path,
        extra_file_keys=["m", "massoftype", "u"],
    )

    print("Testing Toomre Q")
    code_units = get_code_units(path, extra_values=["gamma"])
    udist = code_units["udist"]
    umass = code_units["umass"]
    utime = code_units["utime"]
    code_units["uG"] = udist**3 / (umass * utime**2)
    code_units["uenergy"] = umass * (udist / utime) ** 2
    r_annulus = 25.0

    print("Testing Q at (r, phi)")
    _ = get_Q_toomre(
        hdf5_df,
        r_annulus,
        gamma=code_units["gamma"],
        code_units=code_units,
    )

    print("Testing azimuthal average Q at (r)")
    _ = get_Q_toomre(
        hdf5_df,
        r_annulus,
        gamma=code_units["gamma"],
        code_units=code_units,
        az_avg=True,
    )

    print("Testing grid with loaded frame")
    _ = make_interpolated_grid(dataframe=hdf5_df, grid_size=200)

    print("Testing grid with no loaded frame and different axis")
    _ = make_interpolated_grid(dataframe=None, file_path=path, grid_size=200, yaxis="z")

    print("Passed!")


@pytest.mark.parametrize("ev_name", ["test.ev"])
def test_ev(ev_name):
    """Tests calculating the doppler flip"""

    path = f"./tests/data/{ev_name}"

    print("Loading ev dataframe")
    _ = make_ev_dataframe(path)

    print("Passed!")


@pytest.mark.parametrize("hdf5_name", ["test_hdf5.h5"])
def test_doppler_flip(hdf5_name):
    """Tests calculating the doppler flip"""

    path = f"./tests/data/{hdf5_name}"

    print("Calculating Doppler flip")
    _ = calculate_doppler_flip(path, grid_size=200)

    print("Calculating Doppler flip and plotting map")
    _ = calculate_doppler_flip(path, plot=True, show_plot=False, save_plot=False, grid_size=200)

    print("Passed!")


@pytest.mark.parametrize("hdf5_name", ["test_hdf5.h5"])
def test_fourier_modes(hdf5_name):
    """Tests calculating the doppler flip"""

    path = f"./tests/data/{hdf5_name}"

    print("Loading dataframe")
    hdf5_df = make_hdf5_dataframe(path)

    r_min = 50.0
    r_max = 75.0

    print("Calculating modes with loaded frame")
    _ = calculate_fourier_amps(r_min, r_max, hdf5_df=hdf5_df)

    print("Calculating modes without loaded frame")
    _ = calculate_fourier_amps(r_min, r_max, hdf5_df=None, hdf5_path=path)

    print("Passed!")
