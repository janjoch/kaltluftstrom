# -*- coding: utf-8 -*-
###############################################################################
"""
@author: Janosch Joerg, mail@janjo.ch

V0.4.1

221231

"""
###############################################################################

import re
import json
import types
import datetime as dt
from pathlib import Path

import numba as nb

import numpy as np

import xarray as xr

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

# get toolbox from https://github.com/janjoch/toolbox
import toolbox as tb

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

    def _plot_regression(
        self,
        fig,
        ax,
        x,
        y,
        plot_ci=True,
        plot_pi=True,
        color="#006BA4",
        hatch=None,
        legend_annex="",
    ):
        # regression analysis
        # p, cov, y_model, t, resid, s_err,
        #     chi2_red, n, m = self.regression(x, y)
        reg = tb.arraytools.Regression(x, y)

        # Fit
        ax.plot(
            reg.x,
            reg.y_model,
            "-",
            color=tb.color.change_hex_brightness(color, 1.2),
            linewidth=1.5,
            alpha=0.5,
            label="Regressionsgerade" + legend_annex,
        )

        # Confidence Interval
        if plot_ci:
            ax.fill_between(
                reg.x2,
                reg.y2 + reg.ci,
                reg.y2 - reg.ci,
                color=tb.color.change_hex_brightness(color, 1.2),
                alpha=0.4,
                hatch=hatch,
                label="Konfidenzintervall 95%" + legend_annex,
            )

        # Prediction Interval
        if plot_pi:
            ax.plot(
                reg.x2,
                reg.y2 - reg.pi,
                "--",
                color=tb.color.change_hex_brightness(color, 1.5),
                label="Vorhersagegrenze 95%" + legend_annex,
            )
            ax.plot(
                reg.x2, reg.y2 + reg.pi, "--",
                color=tb.color.change_hex_brightness(color, 1.5),
            )

        return fig, ax, reg


class Timed(Base):
    def __init__(
        self,
        directory,
        feedback=True,
        sensor_labels=types.MappingProxyType({}),
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

            if False:  # deprecated
                # import SHT sensor data
                match = re.match(r"^(S([0-9]+))[A-Za-z0-9_-]*\.edf$", fn.name)
                if match:
                    self.timeseries[match[1]] = self.import_sht_file(
                        directory,
                        match[0],
                        match[1],
                    )
                    self.sht_str.append(match[1])
                    self.sht_int.append(int(match[2]))
                    self.filenames[match[1]] = match[0]

        self.wtdl_str.sort()
        self.wtdl_int.sort()
        self.sht_str.sort()
        self.sht_int.sort()

        # create single timeseries df
        for sensor in self.timeseries:
            self.timeseries[sensor]["T"].name = sensor

        if feedback:
            print(
                "found the following sensor data "
                f"from {str(directory)}:"
            )
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

    # @nb.jit(parallel=True, nopython=False)
    def bin(
        self,
        sensors=None,
        bins_per_h=4,
        earliest_date=None,
        latest_date=None,
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
        if latest_date is None:
            latest_date = np.array([
                self.timeseries[sensor]["T"].dropna().index.max()
                for sensor in sensors
            ]).max().date()
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


class Binned(Base, tb.plot.NotebookInteraction):
    def __init__(
        self,
        binned,
        sensor_labels,
        bins,
        bins_per_h,
        earliest_date,
    ):
        self.binned = binned
        self.sensor_labels = sensor_labels
        self.bins = bins
        self.bins_per_h = bins_per_h
        self.earliest_date = earliest_date

    @tb.plot.magic_plot
    def plot(
        self,
        locs=None,
        wtdl=True,
        sht=True,
        mode="mean",
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
            if re.match(regex, name):
                fig.add_line(
                    series,
                    label=self.sensor_labels.get(name, name),
                    **kwargs,
                )

    def frame(
        self,
        hours=(),
        bins=4,
        sensors=None,
    ):
        bins_per_d = int(self.bins_per_h * 24)
        days = int(self.bins / bins_per_d)

        sensors = (
            self.binned.coords["sensor"]
            if sensors is None
            else list(sensors)
        )
        mean = np.empty((days, len(sensors), len(hours)))
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
                        mean[day, i_s, i_h] = np.nan

                    else:
                        mean[day, i_s, i_h] = float(
                            (
                                subset["mean"] * subset["n"]
                            ).sum() / n[day, i_s, i_h]
                        )

        date = [
            self.earliest_date + dt.timedelta(days=day)
            for day in range(days)
        ]

        coords = ["date", "sensor", "hour"]
        framed = xr.Dataset(
            data_vars=dict(
                mean=(coords, mean),
                n=(coords, n),
            ),
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
        )


class Dated(Base):
    def __init__(
        self,
        timeseries,
        frames={
            "21-1": [21, 0, 1, 0],
            "27-1": [27, 0, 1, 0],
        },
        frame_ref=None,
        date_earliest=None,
        date_latest=None,
        ignore_dates=None,
        average_alg="mean",
        min_count=5,
    ):
        """
        Extract average temperatures of given time frames for each sensor
        and each day.

        Parameters
        ----------
        timeseries: Timed.timeseries dict->pd.Dataframe
        frames: dict->list, optional
            Define time frames to extract daily temperatures.
            Format:
                - dict key: name of frame
                - dict content: list with 4 elements:
                    [hours_start, mins_start, hours_delta, mins_delta]
        frame_ref: list, optional
            Define reference timeframe.
            Format identical entry in frames.
            If undefined, the first frame will be used.
        date_earliest, date_latest: dt.datetime, optional
            First and last date to include in further processing.
        ignore_dates: list->dt.datetime, optional
            Dates to skip in further processing.
        average_alg: str, optional
            Average algorithm to use:
                - mean
                - median
        """
        # init main dateseries DataFrame
        self.dateseries = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(
                (),
                names=("sensor", "key", "unit"),
            )
        )
        self.bins = {}
        self.binned = {}
        self.average_alg = average_alg

        # iterate through all timeframes
        for key, time_shift in frames.items():
            self._frame(
                timeseries,
                key,
                time_shift,
                date_earliest,
                date_latest,
                ignore_dates,
                average_alg,
                min_count,
            )
        if frame_ref:
            self.ref_key = "ref"
            self._frame(
                timeseries,
                "ref",
                frame_ref,
                date_earliest,
                date_latest,
                ignore_dates,
                average_alg,
                min_count,
            )
        else:
            self.ref_key = list(frames.keys())[0]

        self.dateseries.sort_index(inplace=True, axis=1)

    def _frame(
        self,
        timeseries,
        key,
        time_shift,
        date_earliest=None,
        date_latest=None,
        ignore_dates=None,
        average_alg="mean",
        min_count=5,
    ):
        """
        Read out a given time frame each day and for every sensor.
        Saves data under key in self.dateseries.

        Parameters
        ----------
        timeseries: Timed.timeseries dict->pd.Dataframe
        key: str
        time_shift: list of 4 elements
            [hours_start, mins_start, hours_delta, mins_delta]
        date_earliest, date_latest: dt.datetime, optional
            First and last date to include in further processing.
        ignore_dates: list->dt.datetime, optional
            Dates to skip in further processing.
        min_count: int, optional
            Minimum temperature readings required in given timeframe.
            Otherwise, value won't be computed, resulting in the day not
            showing up or a pd.nan.
        """
        timedelta_start = dt.timedelta(
            hours=time_shift[0],
            minutes=time_shift[1],
        )
        timedelta_width = dt.timedelta(
            hours=time_shift[2],
            minutes=time_shift[3],
        )

        # iterate every sensor
        for sensor, timeserie in timeseries.items():
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
                    if self.average_alg == "mean":
                        self.dateseries.loc[
                            date,
                            (sensor, key, "T"),
                        ] = filtered["T"].mean()
                    elif self.average_alg == "median":
                        self.dateseries.loc[
                            date,
                            (sensor, key, "T"),
                        ] = filtered["T"].median()
                    else:
                        raise Exception(
                            "average_alg "
                            + self.average_alg
                            + " does not exist"
                        )
                self.dateseries.loc[
                    date,
                    (sensor, key, "count"),
                ] = filtered.shape[0]

    def assign_bins(
        self,
        ref_sensors,
        key="default",
        bins=5,
        bounds=None,
        average_alg="mean",
    ):
        """
        Compute daily reference temperatures
        and assign each day to a bin based on ref temps.

        Method will add a new col to self.dateseries containing the bin info.

        Parameters
        ----------
        ref_sensors: list
            Reference sensors to use.
        key: str, optional
            Key to save reference frames.
        bins: int, optional
            Number of bins.
        """
        # format ref_sensors correctly
        if (
            not isinstance(ref_sensors, tuple)
            and not isinstance(ref_sensors, list)
        ):
            ref_sensors = (ref_sensors,)
        ref_sensors = ref_sensors
        self.bins[key] = {}

        # calculate daily reference temperatures based on ref_sensors
        if self.average_alg == "mean":
            self.dateseries.loc[
                :,
                ("binning", key, "ref_T"),
            ] = (
                self.dateseries.loc[
                    :,
                    (ref_sensors, self.ref_key, "T"),
                ]
                .droplevel(("key", "unit"), axis="columns")
                .mean(axis=1)
            )
        elif self.average_alg == "median":
            self.dateseries.loc[
                :,
                ("binning", key, "ref_T"),
            ] = (
                self.dateseries.loc[
                    :,
                    (ref_sensors, self.ref_key, "T"),
                ]
                .droplevel(("key", "unit"), axis="columns")
                .median(axis=1)
            )

        # assign bin
        (
            self.dateseries.loc[
                :,
                ("binning", key, "bin_nr"),
            ],
            self.bins[key],
        ) = pd.cut(
            self.dateseries.loc[
                :,
                ("binning", key, "ref_T"),
            ],
            bins,
            labels=False,
            retbins=True,
        )

    def plot_t_drop(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
        bin_key="default",
        frames=(
            "21-1",
            "27-1",
        ),
        boxplot=True,
        boxplot_and_line=False,
        fig_size=(10, 6),
        fig_dpi=140,
        face_color="white",
        fig_legend_loc=None,
        xlim=None,
        ylim=None,
        title=None,
        xlabel="Tages-Referenztemperatur / °C",
        ylabel="Temperaturabfall / °C",
        fig_export=False,
        fig_export_path="",
        fig_export_name="auto",
        fig_export_type="pdf",
        show_plot=True,
        annot_func=None,
    ):
        selection = self._sensor_selection(
            sensor_type,
            sensor_locations,
            sensor_manual,
        )

        fig, ax = self._plt_init(fig_size, fig_dpi, face_color)

        # compute data
        y = []
        x = []
        n = []
        for i in range(len(self.bins[bin_key]) - 1):
            subset = self.dateseries[
                self.dateseries[("binning", bin_key, "bin_nr")] == i
            ].loc[:, (selection,)]

            # delta
            if isinstance(frames, tuple) or isinstance(frames, list):
                sub_1 = subset.loc[
                    :,
                    (slice(None), frames[0], "T"),
                ].droplevel(
                    (
                        1,
                        2,
                    ),
                    axis=1,
                )
                sub_2 = subset.loc[
                    :,
                    (slice(None), frames[1], "T"),
                ].droplevel(
                    (
                        1,
                        2,
                    ),
                    axis=1,
                )
                y.append((sub_1 - sub_2).mean(axis=1))

            # absolute value
            else:
                y.append(
                    subset.loc[
                        :,
                        (slice(None), frames, "T"),
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

            x.append(i)
            n.append(len(subset))

        # boxplot (complex)
        if boxplot or boxplot_and_line:
            ax.boxplot(y, labels=x)
            ax.xaxis.set_tick_params(which="minor", bottom=False)

        # regular line plot (simple)
        if not boxplot or boxplot_and_line:
            y = [e.median() for e in y]
            if boxplot_and_line:
                ax.plot(np.array(x) + 1, y)
            else:
                x = (self.bins[bin_key][:-1] + self.bins[bin_key][1:]) / 2
                ax.plot(x, y, "x-")

        # rewrite x axis labels
        if boxplot or boxplot_and_line:
            ax.set_xticklabels(
                [
                    "{:.1f}\n({:.1f} - {:.1f})\n(n={})".format(
                        (
                            self.bins[bin_key][i] + self.bins[bin_key][i + 1]
                        ) / 2,
                        self.bins[bin_key][i],
                        self.bins[bin_key][i + 1],
                        n[i],
                    )
                    for i in range(len(self.bins[bin_key]) - 1)
                ]
            )

        # finish plot
        fig, ax = self._plt_finish(
            fig,
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            fig_legend_loc=fig_legend_loc,
            annot_func=annot_func,
        )

        if fig_export:
            fig, ax = self._plt_export(
                fig,
                ax,
                fig_export_name,
                fig_export_path,
                fig_export_type,
                title,
                selection,
                fig_size,
            )

        if show_plot:
            plt.show()

    def plot_scatter(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
        bin_key="default",
        frames=(
            "21-1",
            "27-1",
        ),
        plot_ci=True,
        plot_pi=True,
        fig_size=(10, 6),
        fig_dpi=140,
        face_color="white",
        fig_legend_loc="upper right",
        xlim=None,
        ylim=None,
        title=None,
        xlabel="Tages-Referenztemperatur / °C",
        ylabel="Temperaturabfall / °C",
        scatter_label=None,
        fig_export=False,
        fig_export_path="",
        fig_export_name="auto",
        fig_export_type="pdf",
        show_plot=True,
        annot_func=None,
        return_reg=False,
    ):
        selection = self._sensor_selection(
            sensor_type,
            sensor_locations,
            sensor_manual,
        )

        fig, ax = self._plt_init(fig_size, fig_dpi, face_color)

        # extract data
        x, y = extract_records_from_dateseries(
            self.dateseries, selection, frames, bin_key
        )

        # scatter plot
        ax.scatter(x, y, zorder=10, label=scatter_label)

        # confidence interval
        if plot_ci or plot_pi:
            fig, ax, reg = self._plot_regression(
                fig, ax, x, y, plot_ci=plot_ci, plot_pi=plot_pi
            )

        # finish plot
        fig, ax = self._plt_finish(
            fig,
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            fig_legend_loc=fig_legend_loc,
            annot_func=annot_func,
        )

        if fig_export:
            fig, ax = self._plt_export(
                fig,
                ax,
                fig_export_name,
                fig_export_path,
                fig_export_type,
                title,
                selection,
                fig_size,
            )

        if show_plot:
            plt.show()

        if return_reg:
            return reg


class Compare(Base):
    def __init__(
        self,
        dateseries_1,
        dateseries_2,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
        bin_key="default",
    ):
        self.dateseries_1 = dateseries_1
        self.dateseries_2 = dateseries_2
        self.bin_key = bin_key
        self.selection = self._sensor_selection(
            sensor_type,
            sensor_locations,
            sensor_manual,
        )

    def compare_regression(
        self,
        selection_1=None,
        selection_2=None,
        frames=(
            "21-1",
            "27-1",
        ),
    ):
        if not selection_1:
            selection_1 = self.selection
        if not selection_2:
            selection_2 = self.selection

        x1, y1 = extract_records_from_dateseries(
            self.dateseries_1, selection_1, frames, self.bin_key
        )
        x2, y2 = extract_records_from_dateseries(
            self.dateseries_2, selection_2, frames, self.bin_key
        )
        params_1 = self.regression(
            x1, y1
        )  # p, cov, y_model, t, resid, s_err, chi2_red, n, m
        params_2 = self.regression(x2, y2)
        return params_1, params_2

    def plot_scatter(
        self,
        selection_1=None,
        selection_2=None,
        bin_key="default",
        frames=(
            "21-1",
            "27-1",
        ),
        plot_ci=True,
        plot_pi=True,
        fig_size=(10, 6),
        fig_dpi=140,
        face_color="white",
        fig_legend_loc="lower right",
        xlim=None,
        ylim=None,
        title=None,
        xlabel="Tages-Referenztemperatur / °C",
        ylabel="Temperaturabfall / °C",
        scatter_labels=(None, None),
        fig_export=False,
        fig_export_path="",
        fig_export_name="auto",
        fig_export_type="pdf",
        show_plot=True,
        annot_func=None,
        return_reg=False,
    ):
        if not selection_1:
            selection_1 = self.selection
        if not selection_2:
            selection_2 = self.selection

        fig, ax = self._plt_init(fig_size, fig_dpi, face_color)

        # extract data 1
        x1, y1 = extract_records_from_dateseries(
            self.dateseries_1, selection_1, frames, bin_key
        )

        # scatter plot 1
        ax.scatter(x1, y1, color="C0", zorder=10, label=scatter_labels[0])

        # confidence interval 1
        if plot_ci or plot_pi:
            fig, ax, reg1 = self._plot_regression(
                fig,
                ax,
                x1,
                y1,
                color="#006BA4",
                hatch="///",
                plot_ci=plot_ci,
                plot_pi=plot_pi,
            )

        # extract data 2
        x2, y2 = extract_records_from_dateseries(
            self.dateseries_2, selection_2, frames, bin_key
        )

        # scatter plot 2
        ax.scatter(x2, y2, color="C1", zorder=10, label=scatter_labels[1])

        # confidence interval 2
        if plot_ci or plot_pi:
            fig, ax, reg2 = self._plot_regression(
                fig,
                ax,
                x2,
                y2,
                color="#FF800E",
                hatch="\\\\\\",
                plot_ci=plot_ci,
                plot_pi=plot_pi,
            )

        # finish plot
        fig, ax = self._plt_finish(
            fig,
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            fig_legend_loc=fig_legend_loc,
            annot_func=annot_func,
        )

        if fig_export:
            fig, ax = self._plt_export(
                fig,
                ax,
                fig_export_name,
                fig_export_path,
                fig_export_type,
                title,
                self.selection,
                fig_size,
            )

        if show_plot:
            plt.show()

        if return_reg:
            return reg1, reg2


class _BaseDeprecated:
    def _plt_init(
        self,
        fig_size=(10, 6),
        fig_dpi=140,
        face_color="white",
    ):
        plt.style.use("janjo-v0-0.mplstyle")
        fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
        fig.set_facecolor(face_color)
        ax = fig.subplots()

        return fig, ax

    def _plt_finish(
        self,
        fig,
        ax,
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        fig_legend_loc=None,
        annot_func=None,
    ):
        # axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # annotations
        if fig_legend_loc:
            leg = ax.legend(loc=fig_legend_loc)
            leg.set_zorder(20)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # custom annotation
        if annot_func is not None:
            fig, ax = annot_func(fig, ax)

        # clean up layout
        plt.tight_layout(pad=1.5)

        return fig, ax

    def _plt_export(
        self,
        fig,
        ax,
        fig_export_name,
        fig_export_path,
        fig_export_type,
        title,
        selection,
        fig_size,
    ):
        """
        Export a matplotlib.pyplot plot.

        Parameters
        ----------
        fig, ax: matplotlib.pyplot fig and ax objects
        fig_export_name: str
        fig_export_path: str or posix path
        fig_export_type: str or tuple of strs
        """
        if str(fig_export_name) == "auto":
            fig_export_name = str(title) + "_"
            fig_export_name += "".join(selection)
            fig_export_name += (
                "_size-" + str(fig_size[0]) + "-" + str(fig_size[1])
            )
            fig_export_name = (
                fig_export_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace(":", "")
                .replace("/", "-")
                .replace("\\", "-")
                .replace("\n", "_")
            )

        if (
            isinstance(fig_export_type, tuple)
            or isinstance(fig_export_type, list)
        ):
            fig_export_types = fig_export_type
        else:
            fig_export_types = (fig_export_type,)

        for fig_export_type in fig_export_types:
            img_path = (
                Path(fig_export_path)
                / (fig_export_name + "." + fig_export_type)
            )
            plt.savefig(img_path, bbox_inches="tight")
            print("image was saved at", img_path)

        return fig, ax


class _Deprecated:

    def __init__(self):
        pass

    def bin_deprecated(
        self,
        sensors=None,
        bins_per_h=4,
        earliest_date=None,
        latest_date=None,
    ):
        mins = 60 / bins_per_h
        if sensors is None:
            sensors = self._sensor_selection()
        # find time span
        if earliest_date is None:
            earliest_date = np.array([
                self.timeseries[sensor]["T"].dropna().index.min()
                for sensor in sensors
            ]).min().date()
        if latest_date is None:
            latest_date = np.array([
                self.timeseries[sensor]["T"].dropna().index.max()
                for sensor in sensors
            ]).max().date()
        days = (latest_date - earliest_date).days + 1
        bins_per_day = bins_per_h * 24

        # empty np arrays
        mean = np.empty((
            days * bins_per_day,
            len(sensors),
        ))
        mean[:] = np.nan
        std = mean.copy()
        n = np.zeros((
            days * bins_per_day,
            len(sensors),
        ))
        for i_s, sensor in enumerate(sensors):
            for bin in range(bins_per_day * days):
                time_start = dt.datetime.combine(
                    earliest_date,
                    dt.time(),
                ) + dt.timedelta(minutes=bin*mins)
                time_stop = time_start + dt.timedelta(minutes=mins)
                filtered = self.timeseries[sensor][
                    time_start.strftime("%Y-%m-%d %H:%M:%S.%f"):
                    time_stop.strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    )
                ]

                mean[
                    bin,
                    i_s,
                ] = filtered["T"].mean()
                std[
                    bin,
                    i_s,
                ] = filtered["T"].std()
                n[
                    bin,
                    i_s,
                ] = len(filtered["T"])

        timestamp = [
            dt.datetime.combine(
                earliest_date,
                dt.time(),
            ) + dt.timedelta(minutes=bin*mins)
            for bin in range(bins_per_day * days)
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
                # location=["A", "B"],
                sensor=list(sensors),
            )
        )

        return binned

    def plot_temp_time_old(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
        fig_size=(10, 6),
        fig_dpi=140,
        face_color="white",
        fig_legend_loc="upper right",
        xlim=None,
        ylim=None,
        title=None,
        xlabel="Datum/Zeit (MESZ)",
        ylabel="Temperatur / °C",
        fig_export=False,
        fig_export_path="",
        fig_export_name="auto",
        fig_export_type="pdf",
        show_plot=True,
        annot_func=None,
    ):
        selection = self._sensor_selection(
            sensor_type,
            sensor_locations,
            sensor_manual,
        )

        fig, ax = self._plt_init(fig_size, fig_dpi, face_color)

        for sensor in selection:
            ax.plot(
                self.timeseries[sensor].index,
                self.timeseries[sensor]["T"],
                label=self.sensor_labels.get(sensor, sensor),
                ms=None,
            )

        # finish plot
        fig, ax = self._plt_finish(
            fig,
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            fig_legend_loc=fig_legend_loc,
            annot_func=annot_func,
        )

        if fig_export:
            fig, ax = self._plt_export(
                fig,
                ax,
                fig_export_name,
                fig_export_path,
                fig_export_type,
                title,
                selection,
                fig_size,
            )

        if show_plot:
            plt.show()

    def plot_temp_time_interactive(
        self,
        *sensors,
        title=None,
        xlabel="Datum/Zeit (MESZ)",
        ylabel="Temperatur / °C",
        mode="lines",
        height=None,
        width=None,
        plot_all=False,
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
        fig = px.line(
            # self.timeseries[args[0]]["T"].dropna(),
            title=title,
            labels={
                "timestamp": xlabel,
                "value": ylabel,
                "x": xlabel,
                "y": ylabel,
                "sensor": "Sensor",
                "variable": "Sensor",
            },
            height=height,
            width=width,
            # name=sensor_labels.get(args[0], args[0]),
        )
        # for sensor in args[1:]:
        for sensor in sensors:
            fig.add_trace(
                go.Scatter(
                    x=self.timeseries[sensor]["T"].dropna().index,
                    y=self.timeseries[sensor]["T"].dropna(),
                    mode=mode,
                    name=self.sensor_labels.get(sensor, sensor),
                )
            )
        return fig
