import sys
from os.path import abspath, dirname

import pytest
from astropy.io import fits

# Add the parent directory to the Python path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from protomatics.analysis import (
    calc_azimuthal_average,
    get_wiggle_amplitude,
    make_grids,
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
    _ = calc_azimuthal_average(moment1, r_grid=gr)

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

    kep_moment1 = calculate_keplerian_moment1(hdu=fits.open(path))

    kep_rs, kep_phis = extract_wiggle(kep_moment1, in_pv_space=False)

    print("Getting ampltidue with keplerian reference curve")
    _ = get_wiggle_amplitude(rs, phis, ref_rs=kep_rs, ref_phis=kep_phis, vel_is_zero=False)

    print("Getting standard deviation with keplerian reference curve")
    _ = get_wiggle_amplitude(
        rs, phis, ref_rs=kep_rs, ref_phis=kep_phis, vel_is_zero=False, use_std_as_amp=True
    )
