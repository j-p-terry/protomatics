import sys
from os.path import abspath, dirname

import numpy as np
import pytest
from astropy.io import fits

# Add the parent directory to the Python path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from protomatics.moments import make_moments
from protomatics.plotting import (
    get_wiggle_from_contour,
    plot_polar_and_get_contour,
    plot_series,
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

    # with log cmap
    print("Plotting with log norm")
    plot_wcs_data(hdu, vmax=0.8, vmin=0.2, log=True, show=False)

    # with symlog cmap
    print("Plotting with symlog norm")
    plot_wcs_data(hdu, vmax=0.8, vmin=0.2, symlog=True, show=False)

    # with text
    print("Plotting with text")
    plot_wcs_data(
        hdu,
        vmax=0.8,
        vmin=0.2,
        log=True,
        show=False,
        plot_text={"foo": (1, 2), "bar": (3, 4)},
        plot_text_color="red",
    )

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


@pytest.mark.parametrize("scatter", [False, True])
def test_series_plots(scatter):
    """Tests wiggle extraction and plotting"""

    print(f"Testing series plots with scattering: {scatter}")

    data_len = 10
    num_vars = 3
    xs = {i: np.random.randn(data_len) for i in range(num_vars)}
    ys = {i: np.random.randn(data_len) for i in range(num_vars)}
    plot_labels = {i: f"{i}" for i in range(num_vars)}
    vlines = {i: np.random.randint(0, high=10, size=1) for i in range(num_vars)}
    hlines = {i: np.random.randint(0, high=10, size=1) for i in range(num_vars)}

    if scatter:
        cs = {i: np.random.randn(data_len) for i in range(num_vars)}
        cbar_label = "foo"

    print("Plotting")
    plot_series(
        xs,
        ys,
        scatter=scatter,
        scatter_colors={} if not scatter else cs,
        cbar_label="" if not scatter else cbar_label,
        show=False,
        vlines=vlines,
        hlines=hlines,
        plot_labels=plot_labels,
    )

    if scatter:
        print("Plotting scatter without coloring")
        plot_series(
            xs,
            ys,
            scatter=scatter,
            scatter_colors={},
            show=False,
            vlines=vlines,
            hlines=hlines,
            plot_labels=plot_labels,
        )

    print("Passed!")
