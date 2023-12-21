import warnings
from typing import Optional

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS
from matplotlib import colormaps as mplcm
from matplotlib import rc as mplrc
from matplotlib.colors import ListedColormap, LogNorm, SymLogNorm
from matplotlib.patches import Ellipse

##############################################################
##############################################################
##                                                          ##
##          This program contains the necessary functions   ##
##          to make various plots                           ##
##                                                          ##
##############################################################
##############################################################


### plot settings ###
## font sizes
labels = 1.25 * 18  ## x/ylabel
legends = 1.25 * 16  ## legend
ticks = 1.25 * 14  ## x/yticks
titles = 1.25 * 18  # title
lw = 3  # line width
s = 50  # scatter point size (ps)
text = 26  # text size

# colors
cmap = "magma"
diverging_cmap = "RdBu_r"
categorical_cmap = "Set2"
# custom colormap
colors = [
    "firebrick",
    "steelblue",
    "darkorange",
    "darkviolet",
    "cyan",
    "magenta",
    "darkgreen",
    "deeppink",
]
# colors for overlaid contours
overlay_colors = ["cyan", "lime", "magenta", "lightsalmon"]

# marker and lines
markers = ["x", "o", "+", ">", "*", "D", "4"]
linestyles = ["-", "--", ":", "-."]


def prepare_plot_data(
    data: np.ndarray,
    scale_data: float = 1.0,
    line_index: Optional[int] = 1,
    channel: Optional[int] = 0,
    subtract_channels: Optional[list] = None,
) -> np.ndarray:
    """Takes in data and prepares it to be plotted using imshow"""
    # get rid of any axes with dim = 1
    data *= scale_data
    plot_data = data.squeeze()

    # choose transition line if option
    if len(plot_data.shape) == 4:
        plot_data = plot_data[line_index, :, :, :]

    # subtract some channels (e.g. continuum subtraction)
    if subtract_channels is not None and len(plot_data.shape) == 3:
        for i in range(len(subtract_channels)):
            plot_data[channel, :, :] -= plot_data[subtract_channels[i], :, :] / len(
                subtract_channels
            )
        # make sure nothing is negative
        plot_data[plot_data < 0.0] = 0.0

    # choose a channel if it's a cube
    if len(plot_data.shape) == 3:
        plot_data = (
            plot_data[0, :, :] if channel > plot_data.shape[0] - 1 else plot_data[channel, :, :]
        )

    return plot_data


def plot_wcs_data(
    hdu: Optional[list],
    fits_path: Optional[str] = None,
    plot_data: Optional[np.ndarray] = None,
    channel: Optional[int] = 0,
    line_index: Optional[int] = 0,
    contour_value: Optional[float] = None,
    save: bool = False,
    save_name: Optional[str] = None,
    trim: tuple = (None, None),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    overlay_data: Optional[np.ndarray] = None,
    overlay_hdu: Optional[list] = None,
    overlay_pmin: Optional[float] = None,
    overlay_channels: Optional[list] = None,
    subtract_data: Optional[np.ndarray] = None,
    subtract_channels: Optional[list] = None,
    subtract_overlay_channels: Optional[list] = None,
    mask_value: Optional[float] = None,
    num_ticks: int = 5,
    log: bool = False,
    symlog: bool = False,
    symlog_linthresh: float = -0.1,
    scale_data: float = 1.0,
    overlay_data_scale: float = 1.0,
    plot_cmap: str = "magma",
    plot_units: str = "",
    beam_position: Optional[list] = None,
    overlay_beam_position: Optional[list] = None,
    beam_color: str = "white",
    overlay_beam_color: str = "limegreen",
    plot_beam: bool = False,
    plot_overlay_beam: bool = False,
    show: bool = True,
    num_levels: Optional[int] = None,
    interpolation: str = "none",
    plot_text: Optional[dict] = None,
    plot_text_color: str = "white",
    **kwargs,
) -> None:
    """
    This plots a fits file in WCS with the option of overlaying another fits file as contours
    A value (contour_value) from the original data can also be plotted
    A numpy array of the same dimension of the original data can be subtracted (subtract_data)
    kwargs[label_font, tick_font, and legend_font] can be used to override default font sizes
    kwargs[figsize] overrides the default figsize
    kwargs[overlay_cmap, overlay_color_list] overrides the default colors of the overlaid contours
    kwargs[lines] overrides the default linestyle list
    kwargs[linewidth] overrides the default linewidth
    """

    # get font information if given
    label_font = kwargs.get("label_font", labels)
    tick_font = kwargs.get("tick_font", ticks)
    legend_font = kwargs.get("legend_font", legends)
    text_font = kwargs.get("text_font", labels)
    # override default figure size
    figsize = kwargs.get("figsize", (14.0, 10.5))
    # override default colormaps
    overlay_cmap = kwargs.get("overlay_cmap", categorical_cmap)
    overlay_color_list = kwargs.get("overlay_color_list", colors)
    overlay_alpha = kwargs.get("overlay_alpha", 1.0)

    if fits_path is not None:
        hdu = fits.open(fits_path)

    mplrc("xtick", labelsize=tick_font)
    mplrc("ytick", labelsize=tick_font)

    fig = plt.figure(figsize=figsize)

    # set middle to 0 in order to just get angular size (don't care about position)
    hdu[0].header["CRVAL1"] = 0.0
    hdu[0].header["CRVAL2"] = 0.0

    # add WCS to axis
    wcs = WCS(hdu[0].header, naxis=2)
    ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs)
    fig.add_axes(ax)

    RA = ax.coords[0]
    DEC = ax.coords[1]

    RA.set_ticks(number=num_ticks)
    DEC.set_ticks(number=num_ticks)
    RA.set_ticklabel(exclude_overlapping=True)
    DEC.set_ticklabel(exclude_overlapping=True)

    # prepare the data for plotting if none is given
    if plot_data is None:
        plot_data = hdu[0].data.copy()

        plot_data = prepare_plot_data(
            plot_data,
            scale_data=scale_data,
            line_index=line_index,
            channel=channel,
            subtract_channels=subtract_channels,
        )
    else:
        plot_data *= scale_data

    if subtract_data is not None:
        plot_data = plot_data.copy()
        plot_data -= subtract_data

    # masks an inner region to 0 (helpful for delta Keplerian)
    if mask_value is not None:
        plot_data[np.abs(plot_data) <= mask_value] = 0.0

    # make a log normalizer
    norm = LogNorm(vmin, vmax) if log else None
    # do a symmetric log normalizer with a linear threshold as a fraction (if < 0) or given value (> 0)
    norm = (
        norm
        if not symlog
        else SymLogNorm(
            abs(symlog_linthresh) * vmax if symlog_linthresh < 0 else symlog_linthresh,
            vmin=vmin,
            vmax=vmax,
        )
    )

    # plot
    if log:
        plt.imshow(
            plot_data, origin="lower", cmap=plot_cmap, norm=norm, interpolation=interpolation
        )
    else:
        plt.imshow(
            plot_data,
            origin="lower",
            cmap=plot_cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )

    cbar = plt.colorbar(fraction=0.045, pad=0.005)
    cbar.ax.set_ylabel(plot_units, rotation=270, fontsize=legend_font)
    cbar.ax.tick_params(labelsize=tick_font)
    cbar.ax.get_yaxis().labelpad = 40

    # overlay a contour of the original data
    if contour_value is not None:
        ax.contour(plot_data, levels=[contour_value], colors="k")

    # overlay contours from other data
    if overlay_hdu is not None:
        overlay_hdu[0].header["CRVAL1"] = 0.0
        overlay_hdu[0].header["CRVAL2"] = 0.0
        overlay_hdu[0].header["CRPIX1"] = overlay_hdu[0].header["NAXIS1"] // 2
        overlay_hdu[0].header["CRPIX2"] = overlay_hdu[0].header["NAXIS2"] // 2
        overlay_wcs = WCS(overlay_hdu[0].header, naxis=2)

        if overlay_data is None:
            overlay_data = overlay_hdu[0].data.copy().squeeze()

        # make sure we have a channel axis to iterate over (e.g. for continuum)
        if len(overlay_data.shape) == 2:
            overlay_data = overlay_data[np.newaxis, :, :]
            overlay_channels = [0]

        # just make sure that there are actually channels (e.g. for continuum)
        if overlay_channels is None:
            overlay_channels = [0]

        # ensure there are enough colors for all overlays
        if len(overlay_channels) > len(overlay_colors):
            overlay_cmap = overlay_cmap
            overlay_cmap = mplcm.get_cmap(overlay_cmap).colors
            overlay_cmap = ListedColormap(overlay_cmap[: len(overlay_channels)])
        else:
            overlay_cmap = overlay_color_list[:]

        # get each channel to overlay
        for i, overlay_channel in enumerate(overlay_channels):
            this_overlay_data = prepare_plot_data(
                overlay_data,
                scale_data=overlay_data_scale,
                line_index=line_index,
                channel=overlay_channel,
                subtract_channels=subtract_overlay_channels,
            )
            # cut some values
            if overlay_pmin is not None:
                this_overlay_data[
                    this_overlay_data < overlay_pmin * np.max(this_overlay_data)
                ] = 0.0

            # plot the contour
            ax.contour(
                this_overlay_data,
                transform=ax.get_transform(overlay_wcs),
                colors=overlay_cmap[i],
                levels=num_levels,
                alpha=overlay_alpha,
            )

    # add an overlaying text (dictionary [text] -> (x, y))
    if plot_text is not None:
        for text in plot_text:
            plt.text(
                plot_text[text][0],
                plot_text[text][1],
                text,
                color=plot_text_color,
                fontsize=text_font,
            )

    # set axis labels
    y_label = r"$\Delta$ DEC"
    x_label = r"$\Delta$ RA"

    plt.xlabel(x_label, fontsize=label_font)
    plt.ylabel(y_label, fontsize=label_font)

    # set plot limits
    x_size = plot_data.shape[1]
    y_size = plot_data.shape[0]
    if trim[1] is not None:
        plt.ylim(trim[1], y_size - trim[1])
    if trim[0] is not None:
        plt.xlim(trim[0], x_size - trim[0])

    # optionally plot beams
    if plot_beam:
        if "BMIN" not in hdu[0].header.keys() or "BMAJ" not in hdu[0].header.keys():
            pass
        c = Ellipse(
            beam_position,
            width=hdu[0].header["BMIN"],
            height=hdu[0].header["BMAJ"],
            edgecolor=beam_color,
            facecolor=beam_color,
            angle=hdu[0].header.get("BPA", 0),
            transform=ax.get_transform("fk5"),
        )
        ax.add_patch(c)
    if plot_overlay_beam and overlay_hdu is not None:
        if (
            "BMIN" not in overlay_hdu[0].header.keys()
            or "BMAJ" not in overlay_hdu[0].header.keys()
        ):
            pass
        c = Ellipse(
            overlay_beam_position,
            width=overlay_hdu[0].header["BMIN"],
            height=overlay_hdu[0].header["BMAJ"],
            edgecolor=overlay_beam_color,
            facecolor=overlay_beam_color,
            angle=overlay_hdu[0].header.get("BPA", 0),
            transform=ax.get_transform("fk5"),
        )
        ax.add_patch(c)

    if save:
        plt.savefig(save_name)
    if show:
        plt.show()
    else:
        plt.close()


def plot_polar_and_get_contour(
    data: np.ndarray,
    contour_value: float = 0.0,
    middlex: Optional[int] = None,
    middley: Optional[int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    rmax: Optional[float] = None,
    units: str = "",
    show: bool = True,
    rlabel_position: float = 300.0,
    plot_cmap: str = "RdBu_r",
    save: bool = False,
    save_name: str = "",
    **kwargs,
) -> matplotlib.contour.QuadContourSet:
    """Makes a polar plot and extracts the contour with a given contour_value
    kwargs[tick_font, and legend_font] can be used to override default font sizes
    kwargs[figsize] overrides the default figsize
    """

    # get font information if given
    tick_font = kwargs.get("tick_font", ticks)
    legend_font = kwargs.get("legend_font", legends)
    figsize = kwargs.get("figsize", (14.0, 10.5))

    middlex = middlex if middlex is not None else data.shape[1] // 2
    middley = middley if middley is not None else data.shape[0] // 2

    # make range x range
    xs = np.linspace(-middlex, middlex, data.shape[1])
    ys = np.linspace(-middley, middley, data.shape[0])

    # turn into x and y grids
    gx = np.tile(xs, (data.shape[0], 1))
    gy = np.tile(ys, (data.shape[1], 1)).T

    # turn into polar coordinates
    rs = np.sqrt(gx**2 + gy**2)
    phis = np.arctan2(gy, gx)

    mplrc("xtick", labelsize=tick_font)
    mplrc("ytick", labelsize=tick_font)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    plt.grid(False)
    # make a mesh of the data (grid makes warnings, but whatever)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = ax.pcolormesh(phis, rs, data, cmap=plot_cmap)
    # extract the contour for the contour_value
    contour = ax.contour(
        phis,
        rs,
        data,
        levels=[contour_value],
        colors="k",
    )

    ax.tick_params(pad=20)

    im.set_clim(vmin, vmax)

    cbar = fig.colorbar(im, fraction=0.045, pad=0.025, extend="both")
    cbar.ax.set_ylabel(units, rotation=270, fontsize=legend_font, labelpad=0.05)
    cbar.ax.tick_params(labelsize=tick_font)
    cbar.ax.get_yaxis().labelpad = 40

    if rmax is None:
        ax.set_rlabel_position(rlabel_position)

    else:
        ax.set_rlabel_position(0.95 * rmax)
        ax.set_rlim(0, rmax)

    if save:
        plt.savefig(save_name)

    if show:
        plt.show()
    else:
        plt.close()

    return contour


def get_wiggle_from_contour(
    contour: matplotlib.contour.QuadContourSet,
    rmin: Optional[float] = None,
    rmax: Optional[float] = None,
) -> tuple:
    """Goes through a polar contour (extracted with pyplot) and finds the curve with the most entries"""

    # iterates through each contour and finds the longest one
    max_len = 0
    for index in contour.collections[0].get_paths():
        if len(index) > max_len:
            max_path = index
            max_len = len(index)

    # get the vertices of that contour
    v = max_path.vertices.copy()
    phis = np.array(v[:, 0])
    rs = np.array(v[:, 1])

    # trim to fit radial range
    rmin = rmin if rmin is not None else np.min(rs)
    rmax = rmax if rmax is not None else np.max(rs)

    good = np.where((rs >= rmin) & (rs <= rmax))
    rs = rs[good]
    phis = phis[good]

    return rs, phis


def polar_plot(
    rs: np.ndarray,
    phis: np.ndarray,
    rmax: Optional[float] = None,
    scatter: bool = True,
    rlabel_position: float = 300.0,
    show: bool = True,
    save: bool = False,
    save_name: str = "",
    **kwargs,
) -> None:
    """Makes a polar scatter/line plot
    kwargs[tick_font] can be used to override default font sizes
    kwargs[figsize] overrides the default figsize
    kwargs[lines] overrides the default linestyle list
    kwargs[linewidth] overrides the default linewidth
    kwargs[color_list] overrides the default color list
    kwargs[os] overrides the default scatter point size
    """

    # get font information if given
    tick_font = kwargs.get("tick_font", ticks)
    figsize = kwargs.get("figsize", (12.0, 12.0))
    # line attributes
    lines = kwargs.get("lines", linestyles)
    linewidth = kwargs.get("linewidth", lw)
    color_list = kwargs.get("color_list", colors)
    # scatter attributes
    ps = kwargs.get("ps", s)

    mplrc("xtick", labelsize=tick_font)
    mplrc("ytick", labelsize=tick_font)

    _, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

    if scatter:
        plt.scatter(
            phis,
            rs,
            s=ps / 50.0,
            color=colors[0],
            marker=markers[0],
            alpha=0.75,
        )
    else:
        # split into where the curves are above and below the major axis
        negative = np.where(phis < 0)
        positive = np.where(phis > 0)
        plt.plot(phis[negative], rs[negative], lw=linewidth / 2, c=color_list[0], ls=lines[0])
        plt.plot(phis[positive], rs[positive], lw=linewidth / 2, c=color_list[0], ls=lines[0])

    if rmax is None:
        ax.set_rlabel_position(rlabel_position)

    else:
        ax.set_rlabel_position(0.95 * rmax)
        ax.set_rlim(0, rmax)

    if save:
        plt.savefig(save_name)

    if show:
        plt.show()
    else:
        plt.close()


def basic_image_plot(
    data: np.ndarray,
    xlabel: str = "",
    ylabel: str = "",
    cbar_label: str = "",
    plot_cmap: str = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save: bool = False,
    show: bool = True,
    save_name: str = "plot.pdf",
    log: bool = False,
    trimx: int = 0,
    trimy: int = 0,
    **kwargs,
) -> None:
    """Plots a basic image"""

    # get font information if given
    label_font = kwargs.get("label_font", labels)
    tick_font = kwargs.get("tick_font", ticks)
    legend_font = kwargs.get("legend_font", legends)
    # override default figure size
    figsize = kwargs.get("figsize", (14.0, 10.5))

    fig = plt.figure(figsize=figsize)

    # make a log normalizer
    norm = LogNorm(vmin, vmax) if log else None

    # plot
    if log:
        im = plt.imshow(data, origin="lower", cmap=plot_cmap, norm=norm)
    else:
        im = plt.imshow(data, origin="lower", cmap=plot_cmap, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, fraction=0.045, pad=0.025, extend="both")
    cbar.ax.set_ylabel(cbar_label, rotation=270, fontsize=legend_font, labelpad=0.05)
    cbar.ax.tick_params(labelsize=tick_font)
    cbar.ax.get_yaxis().labelpad = 40

    plt.xlabel(xlabel, fontsize=label_font)
    plt.ylabel(ylabel, fontsize=label_font)

    plt.xticks(fontsize=tick_font)
    plt.yticks(fontsize=tick_font)

    if trimx > 0:
        plt.xlim(trimx, data.shape[1] - trimx)
    if trimy > 0:
        plt.ylim(trimy, data.shape[0] - trimy)

    if save:
        plt.savefig(save_name)

    if show:
        plt.show()
    else:
        plt.close()


def plot_series(
    xs: dict,
    ys: dict,
    plot_labels: dict = {},
    x_label: str = "",
    y_label: str = "",
    scatter: bool = False,
    scatter_colors: dict = {},
    scatter_cbar_label: str = "",
    save: bool = False,
    savename: str = "",
    show: bool = True,
    logx: bool = False,
    logy: bool = False,
    vlines: dict = {},
    hlines: dict = {},
    ncols: int = 1,
    **kwargs,
) -> None:
    """
    Plots a series of lines; x, y are dictionaries with corresponding entries in plot_labels (not present = no labels)
    kwargs[color_list] overrides the default color scheme
    kwargs[scatter_color_list] overrides the default colors and makes the scatter points colored by that value
    other kwargs can override the default fonts
    """

    # get font information if given
    label_font = kwargs.get("label_font", labels)
    tick_font = kwargs.get("tick_font", ticks)
    legend_font = kwargs.get("legend_font", legends)
    # override default figure size
    figsize = kwargs.get("figsize", (14.0, 10.5))
    # override default colormaps
    color_list = kwargs.get("color_list", colors)
    scatter_cmap = kwargs.get("scatter_cmap", cmap)

    plt.figure(figsize=figsize)

    i = 0
    min_value = 1e30
    max_value = -1e30
    for var in xs:
        # adds a random color if we haven't given enough
        if i > len(color_list):
            other_colors = mcolors.CSS4_COLORS
            rand_color_index = np.random.randint(0, high=len(other_colors))
            this_color = list(mcolors.CSS4_COLORS.values())[rand_color_index]
            color_list.append(this_color)

        if not scatter:
            plt.plot(
                xs[var],
                ys[var],
                lw=lw,
                color=color_list[i],
                label=None if var not in plot_labels else plot_labels[var],
            )
        else:
            if len(scatter_colors) == 0:
                plt.scatter(
                    xs[var],
                    ys[var],
                    s=s,
                    label=None if var not in plot_labels else plot_labels[var],
                    color=color_list[i],
                )
            else:
                plt.scatter(
                    xs[var],
                    ys[var],
                    s=s,
                    label=None if var not in plot_labels else plot_labels[var],
                    c=scatter_colors[var] if var in scatter_colors else color_list[i],
                    cmap=scatter_cmap,
                )

                if var in scatter_colors:
                    if np.min(scatter_colors[var]) < min_value:
                        min_value = np.min(scatter_colors[var])

                    if np.max(scatter_colors[var]) > max_value:
                        max_value = np.max(scatter_colors[var])

        i += 1

    if len(vlines) > 0:
        for name in vlines:
            plt.axvline(vlines[name], lw=lw, ls="--", c="gray", label=name)

    if len(hlines) > 0:
        for name in hlines:
            plt.axhline(hlines[name], lw=lw, ls=":", c="k", label=name)

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    if len(plot_labels) > 0:
        plt.legend(loc="best", fontsize=legend_font, ncols=ncols)

    if scatter and len(scatter_colors) > 0:
        norm = plt.Normalize(min_value, max_value)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=scatter_cmap)
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.005)
        cbar.ax.set_ylabel(scatter_cbar_label, rotation=270, fontsize=legend_font)
        cbar.ax.tick_params(labelsize=tick_font)
        cbar.ax.get_yaxis().labelpad = 40

    plt.xticks(fontsize=tick_font)
    plt.yticks(fontsize=tick_font)

    plt.xlabel(x_label, fontsize=label_font)
    plt.ylabel(y_label, fontsize=label_font)

    if save:
        plt.savefig(savename)
    if show:
        plt.show()
    else:
        plt.close()
