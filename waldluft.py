# -*- coding: utf-8 -*-
###############################################################################
"""
@author: Janosch Joerg, janosch@duck.com

V0.5.1

230910

"""
###############################################################################

import re
import json
from types import MappingProxyType
import datetime as dt
from pathlib import Path
from math import floor

import numba as nb

import numpy as np

import xarray as xr

import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

# get toolbox from https://github.com/janjoch/toolbox
import toolbox as tb

if tb.plot.CALLED_FROM_NOTEBOOK:
    from IPython.display import display, HTML

"""
data structures
===============

timeseries
----------
dict with each sensor
DataFrame for each Sensor
    index: timestamp (datetime.datetime)
    cols: T (float) (/ RH (float))

dateseries
----------
DataFrame
    index: datetime.date
    cols: multi-index:
        0: sensor: W1, S1, W2, S2, W3, ...
        1: key: 20-1, 27-1, ...
        2: unit: T, count (of datapoints in timerange)
"""


def extract_records_from_dateseries(
    dateseries,
    selection,
    frames,
    bin_key="default",
):
    # if delta T
    if isinstance(frames, tuple) or isinstance(frames, list):
        sub_1 = dateseries.loc[
            :,
            (selection, frames[0], "T"),
        ].droplevel(
            (
                1,
                2,
            ),
            axis=1,
        )
        sub_2 = dateseries.loc[
            :,
            (selection, frames[1], "T"),
        ].droplevel(
            (
                1,
                2,
            ),
            axis=1,
        )
        y = (sub_1 - sub_2).mean(axis=1)

    # if absolute value
    else:
        y = (
            dateseries.loc[
                :,
                (selection, frames, "T"),
            ]
            .droplevel(
                (
                    1,
                    2,
                ),
                axis=1,
            )
            .mean(axis=1)
        )

    x = dateseries[("binning", bin_key, "ref_T")]

    # clean nan values
    cleaned = ~pd.DataFrame({"x": x.isnull(), "y": y.isnull()}).any(axis=1)
    x = x.loc[cleaned]
    y = y.loc[cleaned]

    return x, y


@nb.jit(nopython=True, parallel=True)
def _bin_stats(slice_):
    return (
        slice_.mean(),
        slice_.std(),
        slice_.size,
    )


def _filter_bins(
    index,
    t,
    earliest_date,
    bins,
    bins_per_h,
):
    index_size = index.size
    boundaries = np.empty(bins + 1, np.uint)
    secs = 3600 / bins_per_h
    earliest_date = np.datetime64(earliest_date)

    # find first index
    i = 0
    for i in range(index_size):
        if index[i] >= earliest_date:
            break
    boundaries[0] = i

    # find intervals
    for bin_ in range(bins):
        while i < index_size and index[i] < (
            earliest_date + np.timedelta64(
                int((bin_ + 1) * secs * 1000000),
                "us",
            )
        ):
            i = i + 1
        boundaries[bin_ + 1] = i

    if bin_ < bins - 1:
        boundaries[bin_:] = 0

    mean = np.empty(bins)
    mean[:] = np.nan
    std = mean.copy()
    n = np.zeros(bins)

    for bin in nb.prange(bins):
        if boundaries[bin] < boundaries[bin + 1]:
            mean[bin], std[bin], n[bin] = _bin_stats(t[
                boundaries[bin]: boundaries[bin + 1]
            ])

    return mean, std, n


class Base:
    def __init__(self):
        """
        Global support methods for the waldluft project.
        """
        pass

    def _sensor_selection(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
    ):
        if sensor_manual is None:
            selection = []
            for sensor in self.wtdl_int:
                if (sensor_type is None or sensor_type == "wtdl") and (
                    sensor_locations is None or sensor in sensor_locations
                ):
                    selection.append("W" + str(sensor))

            for sensor in self.sht_int:
                if (sensor_type is None or sensor_type == "sht") and (
                    sensor_locations is None or sensor in sensor_locations
                ):
                    selection.append("S" + str(sensor))

            return selection

        else:
            return sensor_manual


class Timed(Base):
    def __init__(
        self,
        directory,
        feedback=True,
        sensor_labels=MappingProxyType({}),
        import_locs=None,
        import_wtdl=True,
        import_sht=True,
        encoding="ansi",
    ):
        """
        Imports all files from a directory.

        Parameters
        ----------
        directory: str or posix path object
        feedback: bool, optional
            Mute feedback about imported data.
        import_locs: tuple, optional
            Provide locations to import.
            Tuple of ints.
        encoding: str, optional
            File encoding of WTDL .csv files.
        """
        self.sensor_labels = sensor_labels
        self.timeseries = {}
        self.dateseries = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(
                (),
                names=("sensor", "key", "unit"),
            )
        )
        self.wtdl_int = []
        self.wtdl_str = []
        self.sht_int = []
        self.sht_str = []
        self.sht_metadata = {}
        self.sht_sn = {}
        self.filenames = {}
        directory = Path(directory)
        self.sources = pd.DataFrame(
            columns=("type", "loc", "id", "iterator", "comment", "filename",
                     "imported", "sn", "metadata"),
        )
        for fn in directory.iterdir():

            # find WTDL files
            match = re.match(
                (
                    r"^((W([0-9]+)(:?\.[0-9]+)?))"  # sensor-ID
                    r"(:?-[0-9]+)?"  # file iterator
                    r"_?([A-Za-z0-9_-]*)\.csv$"  # note
                ),
                fn.name,
            )
            if match:
                self.sources.loc[len(self.sources)] = [
                    "W",
                    match[3],
                    match[1],
                    match[5],
                    match[6],
                    match[0],
                    False,
                    "",
                    "",
                ]

            # find SHT files
            match = re.match(
                (
                    r"^((S([0-9]+)(:?\.[0-9]+)?))"  # sensor-ID
                    r"(:?-[0-9]+)?"  # file iterator
                    r"_?([A-Za-z0-9_-]*)\.edf$"  # note
                ),
                fn.name,
            )
            if match:
                self.sources.loc[len(self.sources)] = [
                    "S",
                    match[3],
                    match[1],
                    match[5],
                    match[6],
                    match[0],
                    False,
                    "",
                    "",
                ]

        # import WTDL files
        if import_wtdl:
            w_sources = self.sources.loc[self.sources['type'] == "W"]
            if import_locs is None:
                w_locs = w_sources["loc"].drop_duplicates()
            else:
                w_locs = import_locs
        else:
            w_locs = tuple()

        for loc in w_locs:
            files = w_sources.loc[w_sources["loc"] == str(loc)]
            if len(files) == 0:
                print(f"Warning: No file found for location {loc}")
                continue

            # self.timeseries["W" + str(loc)] = pd.concat([
            #     self.import_wtdl_file(
            #         directory,
            #         row["filename"],  # filename
            #         encoding,
            #     )
            #     self.sources.loc[index, "imported"] = True
            #     for index, row
            #     in files.iterrows()
            # ]).sort_index()

            data = []
            for index, row in files.iterrows():
                data.append(self.import_wtdl_file(
                    directory,
                    row["filename"],  # filename
                    encoding=encoding,
                ))
                self.sources.loc[index, "imported"] = True
            self.timeseries["W" + str(loc)] = pd.concat(data).sort_index()

            self.wtdl_str.append("W" + str(loc))
            self.wtdl_int.append(loc)

        # import SHT files
        if import_sht:
            s_sources = self.sources.loc[self.sources['type'] == "S"]
            if import_locs is None:
                s_locs = s_sources["loc"].drop_duplicates()
            else:
                s_locs = import_locs
        else:
            s_locs = tuple()

        for loc in s_locs:
            files = s_sources.loc[s_sources["loc"] == str(loc)]
            if len(files) == 0:
                print(f"Warning: No file found for location {loc}")
                continue

            data = []
            for index, row in files.iterrows():
                data_row, sn, metadata = self.import_sht_file(
                    directory,
                    row["filename"],  # filename
                )
                data.append(data_row)
                self.sources.loc[index, "imported"] = True
                self.sources.loc[index, "sn"] = sn
                self.sources.loc[index, "metadata"] = json.dumps(metadata)
            self.timeseries["S" + str(loc)] = pd.concat(data).sort_index()

            self.sht_str.append("S" + str(loc))
            self.sht_int.append(loc)

        self.wtdl_str.sort()
        self.wtdl_int.sort()
        self.sht_str.sort()
        self.sht_int.sort()

        # create single timeseries df
        for sensor in self.timeseries:
            self.timeseries[sensor]["T"].name = sensor

        if feedback:
            if tb.plot.CALLED_FROM_NOTEBOOK:
                display(HTML(self.sources._repr_html_()))
            else:
                print(self.sources)

    @staticmethod
    def import_wtdl_file(directory, filename, encoding="ansi"):
        data = pd.read_csv(
            directory / filename,
            delimiter=";",
            encoding=encoding,
        )
        data.rename(columns={"Temperatur [°C]": "T"}, inplace=True)
        data["timestamp"] = data["Zeit [s]"].apply(Timed._parse_wtdl_datetime)
        data.set_index("timestamp", inplace=True)
        return data

    @staticmethod
    def import_sht_file(directory, filename):
        data = pd.read_csv(
            directory / filename,
            header=9,
            delimiter="\t",
            encoding="UTF-8",
        )
        data.drop(
            data[data["T"].isin([130.0])].index,
            inplace=True,
        )
        data["timestamp"] = data["Epoch_UTC"].apply(
            Timed._parse_sht_datetime,
        )
        data.set_index("timestamp", inplace=True)
        with open(directory / filename) as f:
            metadata = {}
            for i in range(8):
                line = f.readline()
                match = re.match("^# ([A-Za-z]+)=(.+)$", line)
                if match:
                    metadata[match[1]] = match[2]
                else:
                    print("nothing found in " + line)
        sn = metadata["SensorId"]
        return data, sn, metadata

    @staticmethod
    def _parse_wtdl_datetime(time_str):
        """
        Parse WTDL timestamps.

        Input Format: 05.07.2022 22:53:15
        Output Format datetime.datetime
        """
        match = re.match(
            r"^\s*([0-9]+)"
            r"\.([0-9]+)"
            r"\.([0-9]+)"
            r" ([0-9]+)"
            r":([0-9]+)"
            r":([0-9]+)\s*$",
            time_str,
        )
        ints = np.zeros(6, dtype=np.int64)
        if match is not None:
            for i in range(6):
                ints[i] = int(match[i + 1])
        else:
            match = re.match(r"\s*([0-9]+).([0-9]+).([0-9]+)\s*", time_str)
            for i in range(3):
                ints[i] = int(match[i + 1])
        ints[0], ints[2] = ints[2], ints[0]
        return dt.datetime(*ints)

    @staticmethod
    def _parse_sht_datetime_local(time_str, drop_ms=True):
        """
        Parse SHT timestamps.

        Input Format: 2022-07-12T13:42:15.622628
        Output Format datetime.datetime
        """
        match = re.match(
            r"^\s*([0-9]+)"
            r"-([0-9]+)"
            r"-([0-9]+)"
            r"T([0-9]+)"
            r":([0-9]+)"
            r":([0-9]+)"
            r"\.([0-9]+)\s*$",
            time_str,
        )
        ints = np.zeros(7, dtype=np.int64)
        if match:
            for i in range(7):
                ints[i] = int(match[i + 1])
        else:
            raise Exception("nothing found in " + time_str)
        if drop_ms:
            ints[6] = 0
        return dt.datetime(*ints)

    @staticmethod
    def _parse_sht_datetime(time_int):
        """
        Parse SHT timestamps.

        Input Format: 1667091772.2 (epoch)
        Output Format datetime.datetime
        """
        return (
            dt.datetime.utcfromtimestamp(time_int)
            + dt.timedelta(hours=2)
        )

    def bin(
        self,
        sensors=None,
        bins_per_h=4,
        earliest_date=None,
        latest_date=None,
        feedback=True,
    ):
        secs = 3600 / bins_per_h

        if sensors is None:
            sensors = self._sensor_selection()

        # find time span
        if earliest_date is None:
            earliest_date = np.array([
                self.timeseries[sensor]["T"].dropna().index.min()
                for sensor in sensors
            ]).min().date()
            if feedback:
                print("found earliest date:", str(earliest_date))
        if latest_date is None:
            latest_date = np.array([
                self.timeseries[sensor]["T"].dropna().index.max()
                for sensor in sensors
            ]).max().date()
            if feedback:
                print(" found latest date: ", str(latest_date))
        bins = (
            ((latest_date - earliest_date).days + 1) * 24 * bins_per_h
        )

        # empty np arrays
        mean = np.empty((bins, len(sensors)))
        mean[:] = np.nan
        std = mean.copy()
        n = np.zeros((bins, len(sensors)))
        for i_s in nb.prange(len(sensors)):
            sensor = sensors[i_s]
            index = np.array(self.timeseries[sensor].index)
            t = np.array(self.timeseries[sensor]["T"], dtype=np.float64)

            mean[:, i_s], std[:, i_s], n[:, i_s] = _filter_bins(
                index, t, earliest_date, bins, bins_per_h)

        timestamp = [
            dt.datetime.combine(
                earliest_date,
                dt.time(),
            ) + dt.timedelta(seconds=secs*bin)
            for bin in range(bins)
        ]

        coords = ["timestamp", "sensor"]
        binned = xr.Dataset(
            data_vars=dict(
                mean=(coords, mean),
                std=(coords, std),
                n=(coords, n),
            ),
            coords=dict(
                timestamp=timestamp,
                sensor=list(sensors),
            )
        )

        return Binned(
            binned,
            self.sensor_labels,
            bins,
            bins_per_h,
            earliest_date,
        )

    @tb.plot.magic_plot_preset(
        xlabel="Datum/Zeit (MESZ)",
        ylabel="Temperatur / °C",
    )
    def plot_temp_time(
        self,
        sensors,
        plot_all=False,
        fig=None,
    ):
        if len(sensors) == 0:
            if plot_all:
                sensors = self._sensor_selection()
            else:
                raise Exception(
                    "To avoid overload, "
                    "please confirm plotting all sensors "
                    "with plot_all=True"
                )

        # for sensor in args[1:]:
        for sensor in sensors:
            fig.add_line(
                x=self.timeseries[sensor]["T"].dropna().index,
                y=self.timeseries[sensor]["T"].dropna(),
                label=self.sensor_labels.get(sensor, sensor),
            )

    def extract_dateseries(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
        timedelta_start={"hours": 20},
        timedelta_width={"hours": 1},
        date_earliest=None,
        date_latest=None,
        ignore_dates=None,
        key="20-1",
        average_alg="mean",
        min_count=5,
    ):
        self.selection = self._sensor_selection(
            sensor_type,
            sensor_locations,
            sensor_manual,
        )
        if type(timedelta_start) is dict:
            timedelta_start = dt.timedelta(**timedelta_start)
        if type(timedelta_width) is dict:
            timedelta_width = dt.timedelta(**timedelta_width)

        # iterate every sensor
        for sensor in self.selection:
            timeserie = self.timeseries[sensor]

            if date_earliest is None:
                date_earliest_sens = timeserie.index[0].date()
                date_earliest_sens = dt.datetime(
                    date_earliest_sens.year,
                    date_earliest_sens.month,
                    date_earliest_sens.day,
                )
            else:
                date_earliest_sens = date_earliest

            if date_earliest is None:
                date_latest_sens = timeserie.index[-1].date()
                date_latest_sens = dt.datetime(
                    date_latest_sens.year,
                    date_latest_sens.month,
                    date_latest_sens.day,
                )
            else:
                date_latest_sens = date_latest

            n_days = (date_latest_sens - date_earliest_sens).days + 1

            # iterate every day
            for day in range(n_days):
                date = date_earliest_sens + dt.timedelta(days=day)
                time_start = date + timedelta_start
                time_stop = time_start + timedelta_width
                filtered = timeserie[
                    time_start.strftime("%Y-%m-%d %H:%M:%S.%f"):
                    time_stop.strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    )
                ]

                if filtered.shape[0] >= min_count and (
                    not ignore_dates or date not in ignore_dates
                ):
                    if average_alg == "mean":
                        self.dateseries.loc[
                            date, (sensor, key, "T")
                        ] = filtered["T"].mean()
                    elif average_alg == "median":
                        self.dateseries.loc[
                            date, (sensor, key, "T")
                        ] = filtered["T"].median()
                    else:
                        raise Exception(
                            "average_alg " + average_alg + " does not exist"
                        )
                self.dateseries.loc[
                    date, (sensor, key, "count")
                ] = filtered.shape[0]

    def plot_dateseries_interactive(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
        key="20-1",
        title=None,
        xlabel="Datum",
        ylabel="Temperatur / °C",
    ):
        selection = self._sensor_selection(
            sensor_type,
            sensor_locations,
            sensor_manual,
        )
        fig = go.Figure()
        """
        timeseries = []
        for sensor in selection:
            timeseries.append(self.timeseries[sensor])
        px.line(timeseries)
        """
        fig = px.line(
            self.dateseries.loc[
                :,
                (selection, key, "T"),
            ].droplevel((1, 2), axis=1),
            labels={
                "index": xlabel,
                "value": ylabel,
                "sensor": "Sensor",
                "variable": "Sensor",
            },
            title=title,
        )
        fig.show()


def import_meteo_file(
    directory,
    filename,
    encoding="UTF-8",
):
    data = pd.read_csv(
        directory / filename,
        delimiter=";",
        encoding=encoding,
    )
    data.rename(columns={"tre200s0": "T"}, inplace=True)
    data["timestamp"] = data["time"].apply(
        _parse_meteo_datetime
    )
    data.set_index("timestamp", inplace=True)
    return data


def _parse_meteo_datetime(time_str):
    """
    Parse MeteoSwiss timestamps.

    Input Format: 05.07.2022 22:53:15
    Output Format datetime.datetime
    """
    return (
        tb.datetimeparser.iso_tight(str(time_str), regex_t="")
        + dt.timedelta(hours=2)
    )


def meteo_to_binned(
    dataframe,
    variables,
    sensor_key,
    sensor_label,
    bins_per_h=6,
):
    coords = ["timestamp", "sensor"]

    data_vars = {
        key: (coords, np.expand_dims(np.array(dataframe[df_key]), 1))
        for (key, df_key)
        in variables.items()
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            timestamp=dataframe.index,
            sensor=[sensor_key, ],
        )
    ).assign(n=lambda x: 0 * x["mean"] + 1)

    return Binned(
        ds,
        {sensor_key: sensor_label},
        len(dataframe),
        bins_per_h,
        dataframe.index[0].date(),
        vars=list(variables.keys()),
    )


class Binned(Base, tb.plot.NotebookInteraction):
    def __init__(
        self,
        binned,
        sensor_labels,
        bins,
        bins_per_h,
        earliest_date,
        vars=None,
    ):
        self.binned = binned
        self.sensor_labels = sensor_labels
        self.bins = bins
        self.bins_per_h = bins_per_h
        self.earliest_date = earliest_date
        self.vars = ("mean",) if vars is None else vars

    @tb.plot.magic_plot
    def plot(
        self,
        locs=None,
        wtdl=True,
        sht=True,
        plot_all=None,
        mode="mean",
        label=None,
        fig=None,
        **kwargs,
    ):
        """
        Plot the binned timeseries.

        Parameters
        ----------
        locs: tuple, optional
            Select locations to plot.
            If set to None, all available locations will be plotted.
            Default: None.
        wtdl, sht: bool, optional
            Display WTDL and SHT traces.
            Default: True.
        mode: str, optional
            (mean, std, n)
            Select whether to plot the mean, std or number of samples
            in each bin.
            Default: mean.
        **kwargs
            Keyword arguments for toolbox.plot.Plot.add_trace.
        """
        df = self.binned[mode].to_pandas()
        if wtdl and sht and plot_all is None:
            plot_all = True
        else:
            if wtdl and sht:
                regex = r"[WS]"
            elif wtdl:
                regex = r"W"
            elif sht:
                regex = r"S"
            else:
                print("Warning: Neither WTDL nor SHT selected...")
                regex = r""
            if locs is None:
                regex += r"[0-9]+"
            else:
                regex += "(" + "|".join([str(loc) for loc in locs]) + ")"

        for name, series in df.items():
            if plot_all or re.match(regex, name):
                fig.add_line(
                    series,
                    label=(
                        self.sensor_labels.get(name, name)
                        if label is None
                        else label
                    ),
                    **kwargs,
                )

    def frame(
        self,
        hours=(),
        bins=4,
        sensors=None,
        vars=None,
    ):
        bins_per_d = int(self.bins_per_h * 24)
        days = int(self.bins / bins_per_d)
        if vars is None:
            vars = self.vars

        sensors = (
            self.binned.coords["sensor"]
            if sensors is None
            else list(sensors)
        )
        vars_mean = [
            np.empty((days, len(sensors), len(hours)))
            for _
            in vars
        ]
        n = np.empty((days, len(sensors), len(hours)), dtype=int)

        # iterate hour-frames, days and sensors
        for i_h, hour in enumerate(hours):
            start_bin = int(hour * self.bins_per_h)

            for day in range(days):
                for i_s, sensor in enumerate(sensors):

                    # subset of Dataset
                    subset = self.binned.isel(
                        timestamp=slice(
                            day * bins_per_d + start_bin,
                            day * bins_per_d + start_bin + bins,
                        )
                    ).sel(sensor=sensor)
                    n[day, i_s, i_h] = int(subset["n"].sum())

                    # if no data available
                    if n[day, i_s, i_h] <= 0:
                        for i in range(len(vars)):
                            vars_mean[i][day, i_s, i_h] = np.nan

                    else:
                        for i, var_key in enumerate(vars):
                            vars_mean[i][day, i_s, i_h] = float(
                                (
                                    subset[var_key] * subset["n"]
                                ).sum() / n[day, i_s, i_h]
                            )

        date = [
            self.earliest_date + dt.timedelta(days=day)
            for day in range(days)
        ]

        coords = ["date", "sensor", "hour"]
        data_vars = {
            var_key: (coords, var_mean)
            for (var_key, var_mean)
            in zip(vars, vars_mean)
        }
        data_vars.update(dict(n=(coords, n)))
        framed = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                date=date,
                sensor=sensors,
                hour=list(hours),
            )
        )

        return tb.plot.ShowDataset(
            framed,
            default_var="mean",
            default_isel=dict(hour=0),
        )

    def frame_groups(
        self,
        group,
        hours=(),
        bins=4,
        vars=None,
    ):
        frame = self.frame(hours=hours, bins=bins, sensors=group, vars=vars)
        mean = frame.data.mean(dim="sensor", skipna=False)
        return tb.plot.ShowDataset(
            mean,
            default_var="mean",
        )

    def frame_ref_group(
        self,
        group,
        hour,
        bins=4,
        vars=None,
    ):
        frame = self.frame_groups(
            group=group, hours=(hour,), bins=bins, vars=vars)
        mean = frame.data.mean(dim="hour", skipna=False)
        return tb.plot.ShowDataset(
            mean,
            default_var="mean",
        )

    def frame_delta(
        self,
        hour_1,
        hour_2,
        bins=4,
        sensors=None,
        vars=vars,
    ):
        frame = self.frame(
            hours=(hour_1, hour_2), bins=bins, sensors=sensors, vars=vars)
        delta = frame.data.sel(hour=hour_2) - frame.data.sel(hour=hour_1)
        return tb.plot.ShowDataset(
            delta,
            default_var="mean",
        )

    def frame_groups_delta(
        self,
        group,
        hour_1,
        hour_2,
        bins=4,
    ):
        frame_groups = self.frame_groups(
            group=group,
            hours=(hour_1, hour_2),
            bins=bins,
        )
        groups_delta = (
            frame_groups.data.sel(hour=hour_2)
            - frame_groups.data.sel(hour=hour_1)
        )
        return tb.plot.ShowDataset(
            groups_delta,
            default_var="mean",
        )

    def regression(
        self,
        ref_group,
        group,
        ref_hour,
        hour_1,
        hour_2=None,
        filter_ref_temp=None,
        bins=4,
    ):
        if hour_2 is None:
            hour_2 = hour_1
            hour_1 = ref_hour

        return Regression(
            x=self.frame_ref_group(ref_group, ref_hour, bins=bins),
            y=self.frame_groups_delta(group, hour_1, hour_2, bins=bins),
            filter_x_range=filter_ref_temp,
        )

    def min(
        self,
        sensors=None,
    ):
        bins_per_d = int(self.bins_per_h * 24)
        offset = int(self.bins_per_h * -9)  # night starts the day before at 9h
        days = int(self.bins / bins_per_d)

        sensors = (
            self.binned.coords["sensor"]
            if sensors is None
            else list(sensors)
        )
        min = np.empty((days, len(sensors)))
        n = np.empty((days, len(sensors)), dtype=int)

        # iterate days and sensors

        for day in range(days):
            for i_s, sensor in enumerate(sensors):

                # subset of Dataset
                subset = self.binned.isel(
                    timestamp=slice(
                        day * bins_per_d + offset,
                        (day + 1) * bins_per_d + offset,
                    )
                ).sel(sensor=sensor)
                n[day, i_s] = int(subset["n"].sum())

                # if no data available
                if n[day, i_s] <= 0:
                    min[day, i_s] = np.nan

                else:
                    min[day, i_s] = float(subset["mean"].min())

        date = [
            self.earliest_date + dt.timedelta(days=day)
            for day in range(days)
        ]

        coords = ["date", "sensor"]
        min = xr.Dataset(
            data_vars=dict(
                min=(coords, min),
                n=(coords, n),
            ),
            coords=dict(
                date=date,
                sensor=sensors,
            )
        )

        return tb.plot.ShowDataset(
            min,
            default_var="min",
            # day_start=0,
        )

    def daily(
        self,
        hour_start=0,
        hour_end=24,
        sensors=None,
    ):
        """to be finished..."""
        sensors = (
            self.binned.coords["sensor"]
            if sensors is None
            else list(sensors)
        )

        bins_per_d = round(self.bins_per_h * 24)
        days = int(self.bins / bins_per_d)
        offset_start = floor(self.bins_per_h * hour_start)
        offset_end = floor(self.bins_per_h * hour_end)
        duration = offset_end - offset_start
        days -= floor(hour_end / 24. - 0.001)

        daily = np.empty((days, duration, len(sensors)))

        for i_s, sensor in enumerate(sensors):
            for day in range(days):
                subset = self.binned["mean"].isel(
                    timestamp=slice(
                        day * bins_per_d + offset_start,
                        day * bins_per_d + offset_end,
                    )
                ).sel(sensor=sensor).data

                if not np.isnan(subset).any():
                    daily[day, :, i_s] = subset
                else:
                    daily[day, :, i_s] = np.nan

        date = [
            self.earliest_date + dt.timedelta(days=day)
            for day in range(days)
        ]
        hours = [
            hour_start + (1. / self.bins_per_h) * i
            for i in range(duration)
        ]

        dims = ["date", "hour", "sensor"]
        daily_xr = xr.DataArray(
            daily,
            coords=dict(
                date=date,
                sensor=sensors,
                hour=hours,
            ),
            dims=dims,
        )

        return Daily(
            daily=daily_xr,
            sensor_labels=self.sensor_labels,
        )


class Daily(Base, tb.plot.NotebookInteraction):
    def __init__(
        self,
        daily,
        sensor_labels=None,
    ):
        self.daily = daily
        self.sensor_labels = sensor_labels

    def process_validity(
        self,
        sensors,
        filter_strict=False,
        feedback=True,
    ):
        boolarray = np.isnan(self.daily)
        if not filter_strict and sensors is not None:
            boolarray = boolarray.sel(sensor=list(sensors))
        self.valid_dates = ~boolarray.any(dim=("sensor", "hour"))
        valid_daily = self.daily.sel(date=self.valid_dates)
        valid_mean = valid_daily.mean(dim="date")

        if feedback:
            print(
                "{} verwendbare Tage in einer Zeitspanne von {} Tagen".format(
                    np.count_nonzero(self.valid_dates),
                    len(self.valid_dates),
                )
            )

        return valid_daily, valid_mean, self.valid_dates

    @tb.plot.magic_plot
    def plot(
        self,
        sensors=None,
        locs=None,
        wtdl=True,
        sht=True,
        filter_strict=False,
        feedback=True,
        fig=None,
        **kwargs,
    ):
        """
        Plot the daily temperatures.

        Parameters
        ----------
        sensors: tuple, optional
            Select sensors to plot.
            Disables selection with locs, wtdl and sht.
        locs: tuple, optional
            Select locations to plot.
            If set to None, all available locations will be plotted.
            Default: None.
        wtdl, sht: bool, optional
            Display WTDL and SHT traces.
            Default: True.
        mode: str, optional
            (mean, std, n)
            Select whether to plot the mean, std or number of samples
            in each bin.
            Default: mean.
        **kwargs
            Keyword arguments for toolbox.plot.Plot.add_trace.
        """
        _, valid_mean, _ = self.process_validity(
            sensors,
            filter_strict=filter_strict,
            feedback=feedback,
        )
        if sensors is None:
            if wtdl and sht:
                regex = r"[WS]"
            elif wtdl:
                regex = r"W"
            elif sht:
                regex = r"S"
            else:
                print("Warning: Neither WTDL nor SHT selected...")
                regex = r""
            if locs is None:
                regex += r"[0-9]+"
            else:
                regex += "(" + "|".join([str(loc) for loc in locs]) + ")"

        for name, series in valid_mean.to_pandas().items():
            if (sensors is None and re.match(regex, name)) or (
                sensors is not None and name in sensors
            ):
                fig.add_line(
                    series,
                    label=self.sensor_labels.get(name, name),
                    **kwargs,
                )

    @tb.plot.magic_plot
    def plot_sensor_core(
        self,
        sensor,
        valid_daily,
        valid_mean,
        show_daily=True,
        show_mean=True,
        daily_color="grey",
        daily_opacity=0.4,
        mean_color=None,
        fig=None,
        **kwargs,
    ):
        if show_daily:
            for date, series in valid_daily.sel(
                sensor=sensor
            ).to_pandas().iterrows():
                fig.add_line(
                    series,
                    show_legend=False,
                    color=daily_color,
                    opacity=daily_opacity,
                    **kwargs,
                )

        if show_mean:
            fig.add_line(
                valid_mean.sel(sensor=sensor).to_pandas(),
                label=sensor,
                color=mean_color,
                **kwargs,
            )

    def plot_sensor(
        self,
        sensor,
        filter_strict=False,
        feedback=True,
        **kwargs,
    ):
        # nan filtering
        valid_daily, valid_mean, _ = self.process_validity(
            (sensor,),
            filter_strict=filter_strict,
            feedback=feedback,
        )

        return self.plot_sensor_core(
            sensor,
            valid_daily=valid_daily,
            valid_mean=valid_mean,
            **kwargs,
        )

    def plot_sensors_grid(
        self,
        sensors,
        filter_strict=False,
        feedback=True,
        show_daily=True,
        show_mean=True,
        daily_color="grey",
        daily_opacity=0.4,
        mean_color=None,
        shared_xaxes="all",
        shared_yaxes="all",
        fig=None,
        **kwargs,
    ):
        # input verification
        if isinstance(sensors, (tuple, list)):
            sensors = np.array((sensors,))
        if isinstance(sensors, np.ndarray):
            if len(sensors.shape) != 2:
                raise ValueError("sensors numpy array must be 2D")
            sensors = sensors
            rows, cols = sensors.shape
        else:
            raise ValueError("sensors must be list, tuple or 2D numpy array")
        if fig is None:
            fig = tb.plot.Plot(
                shared_xaxes=shared_xaxes,
                shared_yaxes=shared_yaxes,
                cols=cols,
                rows=rows,
                **kwargs,
            )

        # nan filtering
        valid_daily, valid_mean, _ = self.process_validity(
            sensors.flatten(),
            filter_strict=filter_strict,
            feedback=feedback,
        )

        # grid iteration
        for i_row, sensors_row in enumerate(sensors):
            for i_col, sensor in enumerate(sensors_row):
                fig = self.plot_sensor_core(
                    sensor,
                    valid_daily=valid_daily,
                    valid_mean=valid_mean,
                    row=i_row,
                    col=i_col,
                    fig=fig,
                    show_daily=show_daily,
                    show_mean=show_mean,
                    daily_color=daily_color,
                    daily_opacity=daily_opacity,
                    mean_color=mean_color,
                )
        return fig


class Regression(tb.arraytools.LinearRegression, Base):
    def __init__(
        self,
        x,
        y,
        filter_x_range=None,
    ):
        # convert to np.array
        if isinstance(x, tb.plot.ShowDataset):
            index = x.data.coords["date"].data
            x = np.array(x.data["mean"])
        else:
            index = None
        if isinstance(y, tb.plot.ShowDataset):
            y = np.array(y.data["mean"])

        filter = np.ones((5, x.size), dtype=bool)

        # find nan days
        filter[1] = ~np.isnan(x)
        filter[2] = ~np.isnan(y)

        # find days out of boundary
        if filter_x_range is not None:
            if filter_x_range[0] is not None:
                filter[3] = x > filter_x_range[0]
            if filter_x_range[1] is not None:
                filter[4] = x < filter_x_range[1]

        # filter days
        filter[0] = filter[1:].all(0)
        self.filter = pd.DataFrame(filter.T, columns=(
            "RESULT",
            "x is valid number",
            "y is valid number",
            "within lower boundary",
            "within higher boundary",
        ), index=index)
        x = x[filter[0]]
        y = y[filter[0]]

        super().__init__(x, y)
