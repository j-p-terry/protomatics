from typing import Optional

import h5py
import numpy as np
import pandas as pd


def make_ev_dataframe(file_path: str) -> pd.DataFrame:
    """Reads in a PHANTOM .ev file and returns a pandas dataframe"""

    # load the data
    ev_df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1)

    # get the column names
    with open(file_path) as f:
        line = f.readline()

    # PHANTOM ev files start with # and columns are bracketed with [...]
    header_ = line.split("[")[1:]
    header = []
    for x in header_:
        y = x.split()[1:]
        name = ""
        while len(y) > 0:
            name += y[0]
            name += "_"
            y = y[1:]
        # column ends with ] and there's an extra _
        name = name[:-2]
        header.append(name)

    # assign header to dataframe
    ev_df.columns = header

    return ev_df


def make_hdf5_dataframe(
    file_path: str, extra_file_keys: Optional[list] = None, return_file: bool = False
) -> pd.DataFrame:
    """Reads an HDF5 file and returns a dataframe with the variables in file_keys"""

    # read in file
    file = h5py.File(file_path, "r")

    # basic information that is always loaded
    basic_keys = [
        "x",
        "y",
        "z",
        "vz",
        "vy",
        "vz",
        "r",
        "phi",
        "vr",
        "vphi",
        "h",
    ]

    # initialize dataframe
    hdf5_df = pd.DataFrame(columns=basic_keys)

    # make basic information
    xyzs = file["particles/xyz"][:]
    vxyzs = file["particles/vxyz"][:]
    hdf5_df["x"] = xyzs[:, 0]
    hdf5_df["y"] = xyzs[:, 1]
    hdf5_df["z"] = xyzs[:, 2]
    hdf5_df["h"] = file["particles/h"][:]
    hdf5_df["r"] = np.sqrt(hdf5_df.x**2 + hdf5_df.y**2)
    hdf5_df["phi"] = np.arctan2(hdf5_df.y, hdf5_df.x)
    hdf5_df["vx"] = vxyzs[:, 0]
    hdf5_df["vy"] = vxyzs[:, 1]
    hdf5_df["vz"] = vxyzs[:, 2]
    hdf5_df["vphi"] = -hdf5_df.vx * np.sin(hdf5_df.phi) + hdf5_df.vy * np.cos(hdf5_df.phi)
    hdf5_df["vr"] = hdf5_df.vx * np.cos(hdf5_df.phi) + hdf5_df.vy * np.sin(hdf5_df.phi)

    # add any extra information if you want and can
    if extra_file_keys is not None:
        for key in extra_file_keys:
            # don't get a value we've already used
            if key in hdf5_df.columns:
                continue
            # can also grab sink information
            if key in file["sinks"] and key not in file["particles"].keys():
                for i in range(len(file[f"sinks/{key}"])):
                    # sink values are a scalar, so we repeat over the entire dataframe
                    hdf5_df[f"{key}_{i}"] = np.repeat(file[f"sinks/{key}"][i], hdf5_df.shape[0])
                    continue
            # might be in header
            elif (
                key in file["header"]
                and key not in file["particles"].keys()
                and key not in hdf5_df.columns
            ):
                for i in range(len(file[f"header/{key}"])):
                    # sink values are a scalar, so we repeat over the entire dataframe
                    hdf5_df[f"{key}_{i}"] = np.repeat(file[f"header/{key}"][i], hdf5_df.shape[0])
                    continue
            # value isn't anywhere
            if key not in file["particles"].keys():
                continue
            # only add if each entry is a scalar
            if len(file[f"particles/{key}"][:].shape) == 1:
                hdf5_df[key] = file[f"particles/{key}"][:]
            # if looking for components
            if key == "Bxyz":
                bxyzs = file["particles/Bxyz"][:]
                hdf5_df["Bx"] = bxyzs[:, 0]
                hdf5_df["By"] = bxyzs[:, 1]
                hdf5_df["Bz"] = bxyzs[:, 2]
                hdf5_df["Br"] = hdf5_df.Bx * np.cos(hdf5_df.phi) + hdf5_df.By * np.sin(
                    hdf5_df.phi
                )
                hdf5_df["Bphi"] = -hdf5_df.Bx * np.sin(hdf5_df.phi) + hdf5_df.By * np.cos(
                    hdf5_df.phi
                )
    if not return_file:
        return hdf5_df

    return hdf5_df, file


class SPHData:

    """A class that includes data read from an HDF5 output (designed for PHANTOM at this point)"""

    def __init__(self, file_path: str, extra_file_keys: Optional[list] = None):
        self.data, file = make_hdf5_dataframe(
            file_path, extra_file_keys=extra_file_keys, return_file=True
        )
        self.params = {}
        self.params["nsink"] = len(file["sinks"]["m"][()])
        for key in list(file["header"].keys()):
            self.params[key] = file["header"][key][()]

        sinks = {}
        sinks["m"] = file["sinks"]["m"][()]
        sinks["maccr"] = file["sinks"]["maccreted"][()]
        sinks["x"] = file["sinks"]["xyz"][()][:, 0]
        sinks["y"] = file["sinks"]["xyz"][()][:, 1]
        sinks["z"] = file["sinks"]["xyz"][()][:, 2]
        sinks["vx"] = file["sinks"]["vxyz"][()][:, 0]
        sinks["vy"] = file["sinks"]["vxyz"][()][:, 1]
        sinks["vz"] = file["sinks"]["vxyz"][()][:, 2]

        self.sink_data = pd.DataFrame(sinks)
