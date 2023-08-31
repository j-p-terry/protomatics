import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS
from matplotlib import colormaps as mplcm
from matplotlib import rc as mplrc
from matplotlib.colors import ListedColormap, LogNorm
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
    plot_data: np.ndarray,
    scale_data: float = 1.0,
    line_index: int | None = None,
    channel: int | None = None,
    subtract_channels: list | None = None,
) -> np.ndarray:
    """Takes in data and prepares it to be plotted using imshow"""
    # get rid of any axes with dim = 1
    plot_data = scale_data * plot_data.copy().squeeze()

    # choose transition line if option
    if len(plot_data.shape) == 4:
        plot_data = plot_data[line_index, :, :, :]

    # subtract some channels (e.g. continuum subtraction)
    if subtract_channels is not None and len(plot_data.shape) == 3:
        for i in range(len(subtract_channels)):
            plot_data[channel, :, :] -= plot_data[subtract_channels[i], :, :] / len(
                subtract_channels
            )

    # choose a channel if it's a cube
    if len(plot_data.shape) == 3:
        plot_data = (
            plot_data[0, :, :] if channel > plot_data.shape[0] - 1 else plot_data[channel, :, :]
        )

    return plot_data


def plot_wcs_data(
    hdu: list | None,
    fits_path: str | None = None,
    plot_data: np.ndarray | None = None,
    channel: int | None = None,
    line_index: int | None = None,
    contour_value: float | None = None,
    save: bool = False,
    save_name: str | None = None,
    trim: tuple = (None, None),
    vmin: float | None = None,
    vmax: float | None = None,
    overlay_hdu: list | None = None,
    overlay_pmin: float | None = None,
    overlay_channels: list | None = None,
    subtract_data: np.ndarray | None = None,
    num_ticks: int = 5,
    log: bool = False,
    scale_data: float = 1.0,
    overlay_data_scale: float = 1.0,
    plot_cmap: str = "magma",
    plot_units: str = "",
    beam_position: list | None = None,
    overlay_beam_position: list | None = None,
    beam_color: str = "white",
    overlay_beam_color: str = "limegreen",
    plot_beam: bool = False,
    plot_overlay_beam: bool = False,
    show: bool = True,
) -> None:
    """
    This plots a fits file in WCS with the option of overlaying another fits file as contours
    A value (contour_value) from the original data can also be plotted
    A numpy array of the same dimension of the original data can be subtracted (subtract_data)
    """

    if fits_path is not None:
        hdu = fits.open(fits_path)

    mplrc("xtick", labelsize=ticks)
    mplrc("ytick", labelsize=ticks)

    fig = plt.figure(figsize=(14.0, 10.5))

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
        )

        if subtract_data is not None:
            plot_data -= subtract_data

    # get minimum and maximum values
    vmin = vmin if vmin is not None else np.min(plot_data)
    vmax = vmin if vmin is not None else np.min(plot_data)

    # make a log normalizer
    norm = LogNorm(vmin, vmax) if log else None

    # plot
    if log:
        plt.imshow(plot_data, origin="lower", cmap=plot_cmap, vmin=vmin, vmax=vmax, norm=norm)
    else:
        plt.imshow(plot_data, origin="lower", cmap=plot_cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(fraction=0.045, pad=0.005)
    cbar.ax.set_ylabel(plot_units, rotation=270, fontsize=titles)
    cbar.ax.tick_params(labelsize=ticks)
    cbar.ax.get_yaxis().labelpad = 40

    # overlay a contour of the original data
    if contour_value is not None:
        ax.contour(plot_data, levels=[contour_value], colors="k")

    # overlay contours from other data
    if overlay_hdu is not None:
        overlay_hdu[0].header["CRVAL1"] = 0.0
        overlay_hdu[0].header["CRVAL2"] = 0.0
        overlay_wcs = WCS(overlay_hdu[0].header, naxis=2)

        # ensure there are enough colors for all overlays
        if len(overlay_channels) > len(overlay_colors):
            overlay_cmap = categorical_cmap
            overlay_cmap = mplcm.get_cmap(overlay_cmap).colors
            overlay_cmap = ListedColormap(overlay_cmap[: len(overlay_channels)])
        else:
            overlay_cmap = overlay_colors[:]

        overlay_data = overlay_hdu[0].data.copy()

        # get each channel to overlay
        for i, overlay_channel in enumerate(overlay_channels):
            this_overlay_data = prepare_plot_data(
                overlay_data,
                scale_data=overlay_data_scale,
                line_index=line_index,
                channel=overlay_channel,
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
            )

    # set axis labels
    y_label = r"$\Delta$ DEC"
    x_label = r"$\Delta$ RA"

    plt.xlabel(x_label, fontsize=labels)
    plt.ylabel(y_label, fontsize=labels)

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
    middlex: int | None = None,
    middley: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    rmax: float | None = None,
    units: str = "",
    show: bool = True,
) -> matplotlib.contour.QuadContourSet:
    """Makes a polar plot and extracts the contour with a given contour_value"""

    middlex = middlex if middlex is not None else data.shape[1] // 2
    middley = middley if middley is not None else data.shape[0] // 2

    # make range x range
    xs = np.linspace(-middlex, middlex, data.shape[1])
    ys = np.linspace(-middley, middley, data.shape[0])

    # turn into x and y grids
    gx = np.tile(xs, (data.shape[0], 1))
    gy = np.tile(ys, (data.shape[1], 1)).T

    rs = np.sqrt(gx**2 + gy**2)
    phis = np.arctan2(gx, gy)

    mplrc("xtick", labelsize=ticks)
    mplrc("ytick", labelsize=ticks)

    fig, ax = plt.subplots(figsize=(14.0, 10.5), subplot_kw={"projection": "polar"})
    plt.grid(False)
    im = ax.pcolormesh(phis, rs, data, cmap="RdBu_r")
    contour = ax.contour(phis, rs, data, levels=[contour_value], colors="k")

    ax.tick_params(pad=20)

    im.set_clim(vmin, vmax)

    cbar = fig.colorbar(im, fraction=0.045, pad=0.025, extend="both")
    cbar.ax.set_ylabel(units, rotation=270, fontsize=titles, labelpad=0.05)
    cbar.ax.tick_params(labelsize=ticks)
    cbar.ax.get_yaxis().labelpad = 40

    if rmax is None:
        ax.set_rlabel_position(300)

    else:
        ax.set_rlabel_position(0.95 * rmax)
        ax.set_rlim(0, rmax)

    if show:
        plt.show()
    else:
        plt.close()

    return contour


def get_wiggle_from_contour(
    contour: matplotlib.contour.QuadContourSet,
    rmin: float | None = None,
    rmax: float | None = None,
) -> tuple(np.ndarray, np.ndarray):
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


def polar_plot(rs: np.ndarray, phis: np.ndarray, rmax: float | None = None, scatter: bool = True):
    """Makes a polar scatter/line plot"""

    mplrc("xtick", labelsize=ticks)
    mplrc("ytick", labelsize=ticks)

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})

    if scatter:
        plt.scatter(
            phis,
            rs,
            ps=s / 50.0,
            color=colors[0],
            marker=markers[0],
            alpha=0.75,
        )
    else:
        negative = np.where(phis < 0)
        positive = np.where(phis > 0)
        plt.plot(phis[negative], rs[negative], lw=lw / 2, c=colors[0], ls=linestyles[0])
        plt.plot(phis[positive], rs[positive], lw=lw / 2, c=colors[0], ls=linestyles[0])

    if rmax is None:
        ax.set_rlabel_position(300)

    else:
        ax.set_rlabel_position(0.95 * rmax)
        ax.set_rlim(0, rmax)

    plt.show()
