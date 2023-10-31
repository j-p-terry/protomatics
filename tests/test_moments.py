import sys
from os.path import abspath, dirname

import pytest

# Add the parent directory to the Python path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from protomatics.moments import (
    extract_wiggle,
    make_masked_moments,
    make_moments,
    plot_moments,
    split_pv_curve,
)


@pytest.mark.parametrize("fits_name", ["test_3d_cube.fits", "test_6d_cube.fits"])
def test_moment_and_wiggle(fits_name):
    """Makes moments and extracts wiggle in p-v and p-p space"""

    print(f"Testing moment calculations and wiggle extraction with {fits_name}")

    # plot with preloaded
    path = f"./tests/data/{fits_name}"

    print("Calculating moments")
    calc_moments, _ = make_moments(path)

    # test with precalculated moments
    print("Plotting moments with precalculation")
    plot_moments(calc_moments=calc_moments, fits_path=path, show=False)

    # test keplerian subtraction
    print("Plotting moments with keplerian subtraction")
    plot_moments(calc_moments=calc_moments, fits_path=path, show=False, sub_kep_moment=True)

    # test keplerian subtraction with rotation
    print("Plotting moments with keplerian subtraction")
    plot_moments(
        calc_moments=calc_moments, fits_path=path, show=False, sub_kep_moment=True, rotate=0.785
    )

    # test with loading
    print("Plotting moments with no data loaded")
    plot_moments(fits_path=path, show=False)

    # test with no loading
    print("Plotting moments with no data loaded")
    plot_moments(
        fits_path=path,
        show=False,
        vmins={0: 0.0, 1: -3.0},
        vmaxes={0: 1.0, 1: 3.0},
    )

    # get pv curve
    print("Getting P-V wiggle")
    rs, phis = extract_wiggle(calc_moments[1], in_pv_space=True)

    print("Splitting P-V wiggle")
    _ = split_pv_curve(rs, phis)

    # get pp curve
    print("Getting P-P wiggle")
    _ = extract_wiggle(calc_moments[1], in_pv_space=False)

    print("Passed!")


@pytest.mark.parametrize("fits_name", ["test_3d_cube.fits", "test_6d_cube.fits"])
def test_masked_moments(fits_name):
    """Tests masking keplerian data"""

    print(f"Testing masked moments with {fits_name}")

    # plot with preloaded
    path = f"./tests/data/{fits_name}"

    _ = make_masked_moments(path)

    print("Passed!")
