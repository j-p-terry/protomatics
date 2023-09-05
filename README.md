# ProtoMatics

<h1 align="center">
  <!-- <a href="https://app.circleci.com/pipelines/github/j-p-terry/non_keplerian_anomaly_detection"><img alt="Build" src="https://shields.api-test.nl/circleci/build/github/j-p-terry/non_keplerian_anomaly_detection?style=for-the-badge&token=4bae0fb820e3e7d4ec2352639e35d499c673d78c"></a> -->
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
  <!-- <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a> -->
</h1>


ProtoMatics is a python package to analyze and visualize protoplanetary disk data, particularly in the context of kinematics. It can calculate moment maps, extract velocity channels, visualize fits files (e.g. line emission data), and calculate other quantities of interest. It is mainly designed to be used as a helper in larger analysis scripts.

It can be installed via

```bash
pip install protomatics
```

and imported into any python script with

```python
import protomatics as pm
```

### Basic Examples

(Note: all dashes given in the text variable names should be considered underscores.)

#### Plotting

Several different types of plots can be made. One generally useful function plots data in WCS. Among other functionality, contours of other data can be overlaid, other data can be subtracted, and beams can be plotted.

```python
pm.plot_wcs_data(
    hdu,
    fits_path,
    plot_data,
    channel=0.0,
    line_index=0.0,
    contour_value=None,
    save=False,
    save_name="./plot.pdf",
    trim=(None, None),
    vmin=None,
    vmax=None,
    overlay_hdu=None,
    overlay_pmin=0.0,
    overlay_channels=None,
    subtract_data=None,
    subtract_channels=None,
    subtract_overlay_channels=None,
    num_ticks=5,
    log=False,
    scale_data=1.0,
    overlay_data_scale=1.0,
    plot_cmap="magma",
    plot_units="",
    beam_position=None,
    overlay_beam_position=None,
    beam_color="white",
    overlay_beam_color="limegreen",
    plot_beam=False,
    plot_overlay_beam=False,
    show=True,
    **kwargs,
)
```

where $\mathrm{\texttt{**kwargs}}$ includes options to override default plot values. $\mathrm{\texttt{label-font}}$ sets the axis label font size. $\mathrm{\texttt{tick-font}}$ sets the tick label size. $\mathrm{\texttt{legend-font}}$ sets the legend font size. $\mathrm{\texttt{figsize}}$ sets the figure size as a tuple (H, W). $\mathrm{\texttt{overlay-cmap}}$ sets the default colormap for the overlay contours. $\mathrm{\texttt{overlay-color-list}}$ explicitly passes the colors for the overlay contours. All arguments above are shown as their default values.

Polar plots can be made with

```python
pm.plot_polar_and_get_contour(data, contour_value)
```
which creates a heatmap in polar coordinates and extracts a contour at the given value (optional).

```python
pm.polar_plot(rs, phis, scatter=whether_to_use_scatter_or_line)
```

will make a scatter or line plot in polar coordinates depending on the value of $\mathrm{\texttt{scatter}}$.


#### Moment calculation and analysis

ProtoMatics can calculate moment-0, 1, 2, 8, and 9 maps using [bettermoments](https://github.com/richteague/bettermoments). To caluculate moments, use

```python
moments, uncertainties = pm.make_moments(
    path_to_fits,
    which_moments=[moments_you_want_to_plot],
    vel_min=minimum_velocity_to_use,
    vel_max=maximum_velocity_to_use,
    sub_cont=whether_to_subtract_average_of_first_and_last_channels,
    masked_data=array_with_mask_to_use,
    velax=precalculated_velocity_list,
    rms=precalculated_rms_of_data,
    save_moments=whether_to_save_results,
    outname=prefix_of_saved_file,
)
```

where $\mathrm{\texttt{moments}}$ and $\mathrm{\texttt{uncertainties}}$ are dictionaries with keys corresponding to the moment order. All arguments except for $\mathrm{\texttt{path-to-fits}}$ are optional. If only $\mathrm{\texttt{path-to-fits}}$ is provided, the moments will be loaded and calculated without any additional options.

The moments can be plotted with

```python
pm.plot_moments(moment_dictionary, fits_path=path_to_fits)
```

This has no required arguments. Previously calculated moments ($\mathrm{\texttt{calc-moments}}$) can be passed through or $\mathrm{\texttt{fits-path}}$ can be used to direct the calculation of moments for a given fits file. One of these two things must be put into the function or else there is nothing to plot. The precalcualted moments get priority if both are used. Keplerian moments are calculated if $\mathrm{\texttt{sub-kep-moment}}$ = True. Keplerian moments are calculated using $\mathrm{\texttt{M-star}}$, $\mathrm{\texttt{inc}}$, and $\mathrm{\texttt{distance}}$. They are matched in position space using the fits provided in $\mathrm{\texttt{fits-path}}$.
$\mathrm{\texttt{vmaxes}}$ and $\mathrm{\texttt{vmins}}$ are dictionaries with the maximum and minimum values to plot, respectively.

Moments can also be masked into their Keplerian and non-Keplerian components. Masks are calculated by determining if a given region is within some tolerance ($\mathrm{\texttt{vel-tol}}$) of the Keplerian velocity at that location.

```python
pm.make_masked_moments(path_to_fits)
```
with similar key word arguments to pm.make_moments()

Wiggles can be extracted in either position-position space (where moment-1 = 0) of position-velocity space (velocity along minor axis).

```python
wiggle_rs, wiggle_y = pm.extrac_wiggle(
    moment1_map,
    in_pv_space=whether_to_get_positon_velocity_wiggle,
    rotation_angle=minor_axis_offset_in_degrees,
)
```
$\mathrm{\texttt{wiggle-y}}$ is in radians for (i.e. azimuthal angle of wiggle) position-positon curve and in km/s for position-velocity curve.

The amplitude of the wiggle can be calculated using either integration along the curve or by simple standard deviation.

```python
amplitude = pm.get_wiggle_amplitude(
    rs,
    phis,
    ref_rs=list_of_reference_curves_rs,
    ref_phis=list_of_reference_curves_phis,
    vel_is_zero=whether_the_systemic_channel_is_used,
    use_std_as_amp=whether_to_get_amplitude_as_standard_deviation,
)
```
Only $\mathrm{\texttt{rs}}$ and $\mathrm{\texttt{phis}}$ are required. If $\mathrm{\texttt{vel-is-zero}}$ = True, the $\mathrm{\texttt{reference-curve}}$ is simply taken as the minor axis (i.e, $\phi = \pm \pi / 2$).

One can also calculate the azimuthal average of an array of data using
```python
average_by_r, average_map = pm.calc_azimuthal_average(
    data,
    r_grid=grid_of_radius_at_each_point_in_physical_space,
)
```
$\mathrm{\texttt{data}}$ is mandatory, but the grid is not. If no grid is provided, the radii will be calculated in terms of pixels instead of the physical space defined by $\mathrm{\texttt{r-grid}}$.

This method is conveneint for calculating Doppler flip plots if $\mathrm{\texttt{data}}$ = moment1\_map.

Other functionality is quickly being added. Please report any bugs. More substantial documentation is coming soon.
