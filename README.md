# ProtoMatics

<h1 align="center">
  <!-- <a href="https://app.circleci.com/pipelines/github/j-p-terry/non_keplerian_anomaly_detection"><img alt="Build" src="https://shields.api-test.nl/circleci/build/github/j-p-terry/non_keplerian_anomaly_detection?style=for-the-badge&token=4bae0fb820e3e7d4ec2352639e35d499c673d78c"></a> -->
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
  <!-- <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a> -->
  <!-- <a href="https://doi.org/10.5281/zenodo.8410680"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8410680.svg" alt="DOI"></a> -->
  <a href="https://zenodo.org/record/8410681"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.8410680-blue?style=for-the-badge&logo=Zenodo&logoColor=white" alt="DOI - 10.5281/zenodo.8410680"></a>
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

Basic image plots can be made with
```python
pm.basic_image_plot(
    data,
    xlabel="",
    ylabel="",
    cbar_label="",
    plot_cmap="magma",
    vmin=None,
    vmax=None,
    save=False,
    show=True,
    save_name="plot.pdf",
    log=False,
)
```



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
wiggle_rs, wiggle_y = pm.extract_wiggle(
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

This method is conveneint for calculating Doppler flip plots if $\mathrm{\texttt{data}}$ = azimuthal velocity field

#### HDF5 Analysis

Adding the ability to analyze HDF5 files (e.g. created from PHANTOM outputs) is ongoing. As of now, HDF5 files can be loaded as Pandas dataframes, which can be used to do things such as make an interpolated map of a given value or calculate values such as Fourier amplitudes. Loading a dataframe can be done with

```python
pm.make_hdf5_dataframe(path_to_hdf5_file, extra_file_keys=None)
```
This will return a dataframe with the x, y, z, r, and $\phi$ positions of each particle as well as the x, y, z, r, and $\phi$ components of their velocity. $\mathrm{\texttt{extra-file-keys}}$ is a list of additional data to load. This can be done for any scalar value with a corresponding key in the HDF5 particle dataset. If one wants to load the magnetic field, the key "Bxyz" should be used. This will add the x, y, z, r, and $\phi$ components of the magnetic field to the dataframe.

A 2D grid of interpolated data can be made using this dataframe (or any dataframe). Any value in the dataframe can be made into a grid. This is done with

```python
pm.make_interpolated_grid(
    dataframe,
    grid_size=height_of_grid_in_pixels,
    interpolate_value=value_to_use,
    file_path=path_to_hd5f_file,
    extra_file_keys=any_extra_info,
    return_grids=return_grids_that_were_made,
)
```
$\mathrm{\texttt{dataframe}}$ can be None, but then $\mathrm{\texttt{file-path}}$ needs to be the path to the HDF5 file to be used.

This is all put together to calculate the Doppler flip with

```python
pm.calculate_doppler_flip(
    hdf5_path,
    grid_size=height_of_grid_in_pixels,
    interpolate_value=value_to_use,
    file_path=path_to_hd5f_file,
    extra_file_keys=None,
    return_grids=False,
    plot=whether_to_plot,
    save_plot=whether_to_save_plot,
    show_plot=whether_to_show_plot,
    xlabel=plot_xlabel,
    ylabel=plot_ylabel,
    vmin=plot_minimum_value,
    vmax=plot_maximum_value,
)
```

This will return a map of the doppler flip, a map of the azimuthal velocity, and a map of the azimuthally averaged azimuthal velocity.

HDF5 files can also be used to calculate Fourier amplitudes for given modes within an annulus between a set radial range. This is done with

```python
pm.calculate_fourier_amps(
    r_min,
    r_max,
    modes=modes_to_use,
    hdf5_df=loaded_hdf5_dataframe,
    hdf5_path=path_to_hdf5_file_if_not_loaded,
)
```


Other functionality is quickly being added. Please report any bugs. More substantial documentation is coming soon.


## Citing ProtoMatics

### To Cite a Specific Version:
Please cite the DOI corresponding to the version of $\mathrm{\texttt{ProtoMatics}}$ you used, ensuring that future researchers can access the exact version you utilized. The DOIs for each version can be found in the [Zenodo archive](https://zenodo.org/record/8410680).

### To Cite ProtoMatics Generally:
If you are discussing $\mathrm{\texttt{ProtoMatics}}$ broadly and not referring to a specific version in your work, please use the persistent DOI: [10.5281/zenodo.8410680](https://doi.org/10.5281/zenodo.8410680).

### Example citation using version DOI
```bibtex
@software{protomatics,
  author       = {Jason P. Terry},
  title        = {ProtoMatics: v0.3.2},
  month        = oct,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.3.2},
  doi          = {10.5281/zenodo.8410681},
  url          = {https://doi.org/10.5281/zenodo.8410681}
}
```

### Example citation using persistent DOI
```bibtex
@software{protomatics,
  author       = {Jason P. Terry},
  title        = {ProtoMatics},
  month        = sep,
  year         = 2023,
  publisher    = {PyPi},
  doi          = {10.5281/zenodo.8410680},
  url          = {https://pypi.org/project/protomatics/}
}
```
