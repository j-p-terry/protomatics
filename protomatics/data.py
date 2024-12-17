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
    hdf5_df["mass"] = mass * np.ones_like(hdf5_df.x.to_numpy())

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
    sinks["mass"] = file["sinks"]["m"][()]
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
    def __init__(self, filename: str):
        self.filename = filename
        self.fp = None
        self.int_dtype = None
        self.real_dtype = None
        self.global_params = {}
        self.file_identifier = ""
        self.particle_df = pd.DataFrame()
        self.sinks_df = pd.DataFrame()

    """Reads binary phantom dumpfile
    Method takes significant inspiration from Sarracen (https://github.com/ttricco/sarracen)
    """

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

    def _read_fortran_block(self, size: int) -> bytes:
        start_tag = self.fp.read(4)
        data = self.fp.read(size)
        end_tag = self.fp.read(4)
        if start_tag != end_tag:
            raise RuntimeError("Fortran block boundary mismatch.")
        return data

    def _read_capture_pattern(self):
        start_tag = self.fp.read(4)
        def_types = [
            (np.int32, np.float64),
            (np.int32, np.float32),
            (np.int64, np.float64),
            (np.int64, np.float32),
        ]

        i1 = r1 = i2 = 0
        for def_int_dtype, def_real_dtype in def_types:
            i1 = np.frombuffer(
                self.fp.read(np.dtype(def_int_dtype).itemsize), dtype=def_int_dtype, count=1
            )[0]
            r1 = np.frombuffer(
                self.fp.read(np.dtype(def_real_dtype).itemsize), dtype=def_real_dtype, count=1
            )[0]
            i2 = np.frombuffer(
                self.fp.read(np.dtype(def_int_dtype).itemsize), dtype=def_int_dtype, count=1
            )[0]

            if (
                i1 == def_int_dtype(60769)
                and i2 == def_int_dtype(60878)
                and r1 == def_real_dtype(i2)
            ):
                self.int_dtype = def_int_dtype
                self.real_dtype = def_real_dtype
                break
            # else:
            # rewind
            self.fp.seek(-def_int_dtype().itemsize, 1)
            self.fp.seek(-def_real_dtype().itemsize, 1)
            self.fp.seek(-def_int_dtype().itemsize, 1)

        if (
            i1 != self.int_dtype(60769)
            or i2 != self.int_dtype(60878)
            or r1 != self.real_dtype(i2)
        ):
            raise RuntimeError(
                "Could not determine default int/float precision. Not a Phantom file?"
            )

        # version
        np.frombuffer(
            self.fp.read(np.dtype(self.int_dtype).itemsize), dtype=self.int_dtype, count=1
        )[0]
        i3 = np.frombuffer(
            self.fp.read(np.dtype(self.int_dtype).itemsize), dtype=self.int_dtype, count=1
        )[0]
        if i3 != self.int_dtype(690706):
            raise RuntimeError("Capture pattern error. i3 mismatch.")

        end_tag = self.fp.read(4)
        if start_tag != end_tag:
            raise RuntimeError("Capture pattern error. Fortran tags mismatch.")

    def _read_file_identifier(self):
        block = self._read_fortran_block(100)
        self.file_identifier = block.decode("ascii").strip()

    def _rename_duplicates(self, keys):
        seen = {}
        for i, k in enumerate(keys):
            if k not in seen:
                seen[k] = 1
            else:
                seen[k] += 1
                keys[i] = f"{k}_{seen[k]}"
        return keys

    def _read_global_header_block(self, dtype):
        nvars = np.frombuffer(self._read_fortran_block(4), dtype=np.int32, count=1)[0]
        keys = []
        data = []
        if nvars > 0:
            key_block = self._read_fortran_block(16 * nvars).decode("ascii")
            keys = [key_block[i : i + 16].strip() for i in range(0, 16 * nvars, 16)]
            databytes = self._read_fortran_block(np.dtype(dtype).itemsize * nvars)
            data = np.frombuffer(databytes, dtype=dtype, count=nvars)
        return keys, data

    def _read_global_header(self):
        # Maintain exact order from sarracen logic
        dtypes = [
            self.int_dtype,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            self.real_dtype,
            np.float32,
            np.float64,
        ]

        keys = []
        data = []
        for dt in dtypes:
            new_keys, new_data = self._read_global_header_block(dt)
            keys += new_keys
            data = np.append(data, new_data)
        keys = self._rename_duplicates(keys)
        self.global_params = dict(zip(keys, data))
        self.global_params["file_identifier"] = self.file_identifier

    def _read_array_block(self, df, n, nums):
        # Same dtype order as global parameters
        dtypes = [
            self.int_dtype,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            self.real_dtype,
            np.float32,
            np.float64,
        ]

        for i, count in enumerate(nums):
            dt = dtypes[i]
            for _ in range(count):
                tag = self._read_fortran_block(16).decode("ascii").strip()
                # ensure unique column name
                col_name = tag
                ccount = 1
                while col_name in df.columns:
                    ccount += 1
                    col_name = f"{tag}_{ccount}"
                arr_data = self._read_fortran_block(np.dtype(dt).itemsize * n)
                df[col_name] = np.frombuffer(arr_data, dtype=dt, count=n)
        return df

    def _read_array_blocks(self):
        # number of blocks
        nblocks = np.frombuffer(self._read_fortran_block(4), dtype=np.int32, count=1)[0]

        n_list = []
        nums_list = []
        for _ in range(nblocks):
            start_tag = self.fp.read(4)
            nval = np.frombuffer(self.fp.read(8), dtype=np.int64, count=1)[0]
            nums = np.frombuffer(self.fp.read(32), dtype=np.int32, count=8)
            end_tag = self.fp.read(4)
            if start_tag != end_tag:
                raise RuntimeError("Fortran tags mismatch in array blocks.")
            n_list.append(nval)
            nums_list.append(nums)

        main_df = pd.DataFrame()
        sink_df = pd.DataFrame()

        for i in range(nblocks):
            if i == 1:  # second block = sinks
                sink_df = self._read_array_block(sink_df, n_list[i], nums_list[i])
            else:
                main_df = self._read_array_block(main_df, n_list[i], nums_list[i])

        self.particle_df = main_df
        self.sinks_df = sink_df

    def _assign_masses(self):
        # If multiple itypes and separate_types != "all", mass may need to be assigned
        if "itype" in self.particle_df.columns and self.particle_df["itype"].nunique() > 1:
            # If multiple itypes in main
            self.particle_df["mass"] = self.global_params.get("massoftype", np.nan)
            for t in self.particle_df["itype"].unique():
                if t > 1:
                    key = f"massoftype_{t}"
                    if key in self.global_params:
                        self.particle_df.loc[
                            self.particle_df["itype"] == t, "mass"
                        ] = self.global_params[key]
        else:
            # Just set a global mass if possible
            if "massoftype" in self.global_params:
                self.global_params["mass"] = self.global_params["massoftype"]

    def read(
        self,
        separate_types: Optional[str] = "sinks",
        ignore_inactive: bool = True,
        return_params: bool = True,
    ) -> Union[
        pd.DataFrame, list[pd.DataFrame], tuple[Union[pd.DataFrame, list[pd.DataFrame]], dict]
    ]:
        self._open_file()
        try:
            self._read_capture_pattern()
            self._read_file_identifier()
            self._read_global_header()
            self._read_array_blocks()
        finally:
            self._close_file()

        if ignore_inactive and "h" in self.particle_df.columns:
            self.particle_df = self.particle_df[self.particle_df["h"] > 0]

        self._assign_masses()

        # Return logic:
        if separate_types is None:
            # Combine main and sinks
            combined = pd.concat([self.particle_df, self.sinks_df], ignore_index=True)
            if return_params:
                return combined, self.global_params
            return combined

        if separate_types == "sinks":
            # Return main and sinks separately
            if not self.sinks_df.empty:
                if return_params:
                    return (self.particle_df, self.sinks_df, self.global_params)
                return (self.particle_df, self.sinks_df)
            if return_params:
                return self.particle_df, self.global_params
            return self.particle_df

        if separate_types == "all":
            # Separate by itype plus sinks
            # If multiple itypes:
            if "itype" in self.particle_df.columns and self.particle_df["itype"].nunique() > 1:
                df_list = []
                for _tval, group in self.particle_df.groupby("itype"):
                    # Just separate the data, no mass recalc needed here
                    df_list.append(group.copy())
                if not self.sinks_df.empty:
                    df_list.append(self.sinks_df)
                if return_params:
                    return df_list, self.global_params
                return df_list
            # else:
            # Just one itype
            df_list = [self.particle_df]
            if not self.sinks_df.empty:
                df_list.append(self.sinks_df)
            if return_params:
                return df_list, self.global_params
            return df_list
        # else:
        # Unknown separate_types
        raise ValueError("Invalid value for separate_types. Choose None, 'sinks', or 'all'.")


def read_phantom(
    filename: str,
    separate_types: str = "sinks",
    ignore_inactive: bool = True,
    return_params: bool = True,
):
    """
    Convenience function to use PhantomFileReader and return pandas DataFrames and params.
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
        else:
            self.data, self.sink_data, self.params = read_phantom(
                file_path,
                ignore_inactive=ignore_inactive,
                separate_types=separate,
            )
            self.data["iorig"] = self.data["iorig"].astype(int)
            self.data["r"] = np.sqrt(self.data.x**2 + self.data.y**2)
            self.data["phi"] = np.arctan2(self.data.y, self.data.x)
            if "Bx" in self.data.columns:
                self.data["Br"] = self.data.Bx * np.cos(self.data.phi) + self.data.By * np.sin(
                    self.data.phi
                )
                self.data["Bphi"] = -self.data.Bx * np.sin(self.data.phi) + self.data.By * np.cos(
                    self.data.phi
                )
        self.params["usdensity"] = self.params["umass"] / (self.params["udist"] ** 2)
        self.params["udensity"] = self.params["umass"] / (self.params["udist"] ** 3)
        self.params["uvol"] = self.params["udist"] ** 3.0
        self.params["uarea"] = self.params["udist"] ** 2.0
        self.params["uvel"] = self.params["udist"] / self.params["utime"]
        if type(self.params["massoftype"]) == np.ndarray:
            self.params["mass"] = self.params["massoftype"][0]
        else:
            self.params["mass"] = self.params["massoftype"]

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
            usdense=self.params["usdensity"],
            particle_mass=self.params["mass"],
        )
        self.data["sigma"] = sigma

    def add_mass(self):
        if "mass" not in self.data.columns:
            self.data["mass"] = self.params["mass"] * np.ones_like(self.data["x"].to_numpy())

    def add_vphi(self):
        self.data["vphi"] = -self.data.vx * np.sin(self.data.phi) + self.data.vy * np.cos(
            self.data.phi
        )

    def add_vr(self):
        self.data["vr"] = self.data.vx * np.cos(self.data.phi) + self.data.vy * np.sin(
            self.data.phi
        )
