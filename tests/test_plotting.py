import sys
from os.path import abspath, dirname

import pytest
from astropy.io import fits

# Add the parent directory to the Python path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from protomatics.moments import make_moments
from protomatics.plotting import (
    get_wiggle_from_contour,
    plot_polar_and_get_contour,
    plot_wcs_data,
    polar_plot,
)


@pytest.mark.parametrize(
    "fits_name",
    [
        "test_6d_cube.fits",
        "test_2d.fits",
        "test_3d_cube.fits",
    ],
)
def test_wcs_plot(fits_name):
    """Tests if a wcs plot can be made with a varying number of input dimensions"""

    print(f"Plotting WCS for {fits_name}")

    # plot with preloaded
    print("Loading")
    hdu = fits.open(f"./tests/data/{fits_name}")

    # basic plot
    print("Default plot")
    plot_wcs_data(hdu, show=False)

    # with additional options
    print("Plotting with subtraction")
    plot_wcs_data(hdu, subtract_channels=[0, -1], show=False)

    # with additional options
    print("Plotting with maximum values")
    plot_wcs_data(hdu, vmax=0.8, vmin=0.2, show=False, interpolation="bicubic")

    # with additional options
    print("Plotting with log norm")
    plot_wcs_data(hdu, vmax=0.8, vmin=0.2, log=True, show=False)

    if "cube" in fits_name:
        overlay_hdu = fits.open("./tests/data/test_2d.fits")
    else:
        overlay_hdu = fits.open("./tests/data/test_3d_cube.fits")

    # with additional options
    print("Plotting with overlay")
    plot_wcs_data(
        hdu,
        subtract_channels=[0, -1],
        overlay_hdu=overlay_hdu,
        overlay_channels=[3],
        subtract_overlay_channels=[0, -1],
        show=False,
    )
    print("Passed!")


@pytest.mark.parametrize("fits_name", ["test_3d_cube.fits", "test_6d_cube.fits"])
def test_wiggle_plots(fits_name):
    """Tests wiggle extraction and plotting"""

    print(f"Testing wiggle plots with {fits_name}")
    # plot with preloaded
    path = f"./tests/data/{fits_name}"

    print("Calculating moments")
    calc_moments, _ = make_moments(path)
    moment1 = calc_moments[1]

    print("Extracting contour")
    contour = plot_polar_and_get_contour(moment1, show=False)

    print("Extracting wiggle")
    rs, phis = get_wiggle_from_contour(contour)

    print("Scattering wiggle")
    polar_plot(rs, phis, show=False)

    print("Plotting wiggle")
    polar_plot(rs, phis, show=False, scatter=False)

    print("Passed!")
