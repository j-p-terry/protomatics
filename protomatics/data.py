from typing import Optional, Union

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
        "iorig",
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
    hdf5_df["iorig"] = file["particles/iorig"][:]
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

    mass = file["header/massoftype"][:]
    if type(mass) == np.ndarray:
        mass = mass[0]
    hdf5_df["m"] = mass * np.ones_like(hdf5_df.x.to_numpy())

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
                print(f"{key} not in file!")
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


def make_sink_dataframe(file_path: str, file: h5py._hl.files.File = None):
    """Reads in an .h5 output and gets sink data"""

    if file is None:
        file = h5py.File(file_path, "r")

    sinks = {}
    sinks["m"] = file["sinks"]["m"][()]
    sinks["maccr"] = file["sinks"]["maccreted"][()]
    sinks["x"] = file["sinks"]["xyz"][()][:, 0]
    sinks["y"] = file["sinks"]["xyz"][()][:, 1]
    sinks["z"] = file["sinks"]["xyz"][()][:, 2]
    sinks["vx"] = file["sinks"]["vxyz"][()][:, 0]
    sinks["vy"] = file["sinks"]["vxyz"][()][:, 1]
    sinks["vz"] = file["sinks"]["vxyz"][()][:, 2]

    return pd.DataFrame(sinks)


def get_run_params(file_path: str, file: h5py._hl.files.File = None):
    """Reads in an .h5 output and gets parameter data"""

    if file is None:
        file = h5py.File(file_path, "r")
    params = {}
    try:
        params["nsink"] = len(file["sinks"]["m"][()])
    except KeyError:
        params["nsink"] = 0
    for key in list(file["header"].keys()):
        params[key] = file["header"][key][()]

    return params


class PhantomFileReader:
    """
    Reads in phantom binary data dump
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.fp = None
        self.int_dtype = None
        self.real_dtype = None
        self.global_params = {}
        self.file_identifier = ""
        self.particle_df = pd.DataFrame()
        self.sinks_df = pd.DataFrame()

    def _open_file(self):
        self.fp = open(self.filename, "rb")

    def _close_file(self):
        if self.fp is not None:
            self.fp.close()
            self.fp = None

    def __enter__(self):
        self._open_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_file()

    def _read_block(self, size: int) -> bytes:
        start_tag = self.fp.read(4)
        data = self.fp.read(size)
        end_tag = self.fp.read(4)
        if start_tag != end_tag:
            raise RuntimeError("Fortran block size tags do not match.")
        return data

    def _identify_precision(self):
        candidates = [
            (np.int32, np.float64),
            (np.int32, np.float32),
            (np.int64, np.float64),
            (np.int64, np.float32),
        ]

        initial_pos = self.fp.tell()

        for int_type, float_type in candidates:
            self.fp.seek(initial_pos)
            start_tag = self.fp.read(4)
            try:
                i1 = np.frombuffer(
                    self.fp.read(np.dtype(int_type).itemsize), dtype=int_type, count=1
                )[0]
                r1 = np.frombuffer(
                    self.fp.read(np.dtype(float_type).itemsize), dtype=float_type, count=1
                )[0]
                i2 = np.frombuffer(
                    self.fp.read(np.dtype(int_type).itemsize), dtype=int_type, count=1
                )[0]
                np.frombuffer(self.fp.read(np.dtype(int_type).itemsize), dtype=int_type, count=1)[
                    0
                ]
                i3 = np.frombuffer(
                    self.fp.read(np.dtype(int_type).itemsize), dtype=int_type, count=1
                )[0]
                end_tag = self.fp.read(4)
            except Exception:
                continue

            if (
                i1 == int_type(60769)
                and i2 == int_type(60878)
                and float_type(i2) == r1
                and i3 == int_type(690706)
                and start_tag == end_tag
            ):
                self.int_dtype = int_type
                self.real_dtype = float_type
                return

        raise ValueError("Failed to detect int/float precision. Not a valid Phantom file?")

    def _read_file_id(self):
        block = self._read_block(100)
        self.file_identifier = block.decode("ascii").strip()

    def _read_global_parameters(self):
        candidate_types = [
            self.int_dtype,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            self.real_dtype,
            np.float32,
            np.float64,
        ]

        names_combined = []
        values_combined = []
        for dt in candidate_types:
            try:
                nvars_data = self._read_block(4)
            except RuntimeError:
                # no more data
                break
            nvars = np.frombuffer(nvars_data, dtype=np.int32, count=1)[0]
            if nvars == 0:
                continue

            name_block = self._read_block(16 * nvars).decode("ascii")
            var_names = [name_block[i : i + 16].strip() for i in range(0, 16 * nvars, 16)]

            var_data = self._read_block(np.dtype(dt).itemsize * nvars)
            var_values = np.frombuffer(var_data, dtype=dt, count=nvars)

            names_combined.extend(var_names)
            values_combined.extend(var_values)

        seen = {}
        for i, n in enumerate(names_combined):
            if n in seen:
                seen[n] += 1
                names_combined[i] = f"{n}_{seen[n]}"
            else:
                seen[n] = 1

        for k, v in zip(names_combined, values_combined):
            self.global_params[k] = v

        self.global_params["file_identifier"] = self.file_identifier

    def _read_particle_blocks(self):
        nblocks = np.frombuffer(self._read_block(4), dtype=np.int32, count=1)[0]
        block_info = []
        for _ in range(nblocks):
            start_tag = self.fp.read(4)
            n = np.frombuffer(self.fp.read(8), dtype=np.int64, count=1)[0]
            nums = np.frombuffer(self.fp.read(32), dtype=np.int32, count=8)
            end_tag = self.fp.read(4)
            if start_tag != end_tag:
                raise RuntimeError("Block header tags mismatch.")
            block_info.append((n, nums))

        arr_types = [
            self.int_dtype,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            self.real_dtype,
            np.float32,
            np.float64,
        ]

        def read_block_data(num_particles, nums_per_type):
            df_block = pd.DataFrame()
            for dt, count_dt in zip(arr_types, nums_per_type):
                for _ in range(count_dt):
                    name_raw = self._read_block(16).decode("ascii").strip()
                    data_block = self._read_block(num_particles * np.dtype(dt).itemsize)
                    data_arr = np.frombuffer(data_block, dtype=dt, count=num_particles)
                    base_name = name_raw
                    col_name = base_name
                    ccount = 1
                    while col_name in df_block.columns:
                        ccount += 1
                        col_name = f"{base_name}_{ccount}"
                    df_block[col_name] = data_arr
            return df_block

        main_df = pd.DataFrame()
        sink_df = pd.DataFrame()
        for i, (n, nums) in enumerate(block_info):
            block_df = read_block_data(n, nums)
            if i == 1:
                sink_df = block_df
            else:
                main_df = pd.concat([main_df, block_df], ignore_index=True)

        self.particle_df = main_df
        self.sinks_df = sink_df

    def _assign_masses(self):
        if "itype" in self.particle_df.columns:
            if self.particle_df["itype"].nunique() > 1:
                self.particle_df["mass"] = self.global_params.get("massoftype", np.nan)
                for t in self.particle_df["itype"].unique():
                    if t > 1:
                        key = f"massoftype_{t}"
                        if key in self.global_params:
                            self.particle_df.loc[
                                self.particle_df["itype"] == t, "mass"
                            ] = self.global_params[key]
            else:
                self.particle_df["mass"] = self.global_params.get("massoftype", np.nan)
        else:
            self.particle_df["mass"] = self.global_params.get(
                "massoftype", self.global_params.get("mass", np.nan)
            )

    def read(
        self,
        separate_types: str = "sinks",
        ignore_inactive: bool = True,
        return_params: bool = False,
    ) -> Union[
        pd.DataFrame, list[pd.DataFrame], tuple[Union[pd.DataFrame, list[pd.DataFrame]], dict]
    ]:
        self._open_file()
        try:
            self._identify_precision()
            self._read_file_id()
            self._read_global_parameters()
            self._read_particle_blocks()
        finally:
            self._close_file()

        if ignore_inactive and "h" in self.particle_df.columns:
            self.particle_df = self.particle_df[self.particle_df["h"] > 0]

        self._assign_masses()

        # Combine or separate dataframes according to separate_types
        if separate_types is None:
            combined = pd.concat([self.particle_df, self.sinks_df], ignore_index=True)
            result = combined
        elif separate_types == "sinks":
            if not self.sinks_df.empty:
                # Assign mass to sinks if needed
                if "itype" in self.sinks_df and self.sinks_df["itype"].nunique() > 1:
                    # handle multiple sink species if needed
                    self.sinks_df["mass"] = self.global_params.get("massoftype", np.nan)
                else:
                    self.sinks_df["mass"] = self.global_params.get("massoftype", np.nan)
                result = [self.particle_df, self.sinks_df]
            else:
                result = self.particle_df
        else:  # separate_types == "all"
            result_dfs = []
            if "itype" in self.particle_df.columns and self.particle_df["itype"].nunique() > 1:
                for _, group in self.particle_df.groupby("itype"):
                    group_ = group.copy()
                    result_dfs.append(group_)
            else:
                result_dfs.append(self.particle_df)

            if not self.sinks_df.empty:
                if "itype" in self.sinks_df.columns and self.sinks_df["itype"].nunique() > 1:
                    self.sinks_df["mass"] = self.global_params.get("massoftype", np.nan)
                else:
                    self.sinks_df["mass"] = self.global_params.get("massoftype", np.nan)
                result_dfs.append(self.sinks_df)

            result = result_dfs[0] if len(result_dfs) == 1 else result_dfs

        if return_params:
            return result, self.global_params
        return result


def read_phantom(
    filename: str,
    separate_types: str = "sinks",
    ignore_inactive: bool = True,
    return_params: bool = True,
):
    """
    Convenience function to read a Phantom dump file.

    Parameters
    ----------
    filename : str
        The Phantom binary file to read.
    separate_types : {"sinks", "all", None}
        How to separate particle data into multiple DataFrames.
    ignore_inactive : bool
        If True, removes particles with non-positive smoothing length.
    return_params : bool
        If True, return the global parameters dictionary along with the particle data.

    Returns
    -------
    data : pd.DataFrame or list of pd.DataFrame
        Particle data as a single DataFrame or multiple DataFrames depending on separate_types.
    params : dict (only if return_params=True)
        Dictionary of global parameters from the file header.
    """
    with PhantomFileReader(filename) as reader:
        return reader.read(
            separate_types=separate_types,
            ignore_inactive=ignore_inactive,
            return_params=return_params,
        )


class SPHData:

    """A class that includes data read from a dumpfile in binary or HDF5 (designed for PHANTOM at this point)"""

    def __init__(
        self,
        file_path: str,
        extra_file_keys: Optional[list] = None,
        ignore_inactive: bool = True,
        separate: str = "sinks",
    ):
        self.file_path = file_path

        if ".h5" in file_path:
            self.data, file = make_hdf5_dataframe(
                file_path,
                extra_file_keys=extra_file_keys,
                return_file=True,
            )

            self.sink_data = make_sink_dataframe(None, file)
            self.params = get_run_params(None, file)

            self.params["usdensity"] = self.params["umass"] / (self.params["udist"] ** 2)
            self.params["udensity"] = self.params["umass"] / (self.params["udist"] ** 3)
            self.params["uvol"] = self.params["udist"] ** 3.0
            self.params["uarea"] = self.params["udist"] ** 2.0
            self.params["uvel"] = self.params["udist"] / self.params["utime"]

            if type(self.params["massoftype"]) == np.ndarray:
                self.params["mass"] = self.params["massoftype"][0]
            else:
                self.params["mass"] = self.params["massoftype"]
        else:
            (self.data, self.sink_data), self.params = read_phantom(
                file_path,
                ignore_inactive=ignore_inactive,
                separate_types=separate,
            )

    def add_surface_density(
        self,
        dr: float = 0.1,
        dphi: float = np.pi / 20,
    ):
        from .analysis import compute_local_surface_density

        """Compuates surface density in r, phi bins and converts to cgs"""
        sigma = compute_local_surface_density(
            self.data.copy(),
            dr=dr,
            dphi=dphi,
            uarea=self.params["uarea"],
            particle_mass=self.params["mass"],
        )
        self.data["sigma"] = sigma
