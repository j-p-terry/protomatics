import sys
from os.path import abspath, dirname

import pytest
from astropy.io import fits

# Add the parent directory to the Python path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from protomatics.analysis import calc_azimuthal_average, make_grids, make_peak_vel_map
from protomatics.moments import make_moments


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
