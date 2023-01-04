# -*- coding: utf-8 -*-
################################################################################
"""
@author: Janosch Joerg, mail@janjo.ch

V0.3.0

221223

"""
################################################################################

import os
import re
import types
import datetime as dt
import functools as ft
from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

import scipy as sp

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

class Base:

    def __init__(self):
        """
        Global support methods for the waldluft project.
        """
        pass

    def _export_plot(
        self,
        fig,
        ax,
        file_export_name,
        file_export_path,
        file_export_type,
        title,
        selection,
        fig_size,
    ):
        """
        Export a matplotlib.pyplot plot.

        Parameters
        ----------
        fig, ax: matplotlib.pyplot fig and ax objects
        file_export_name: str
        file_export_path: str or posix path
        file_export_type: str or tuple of strs
        """
        if str(file_export_name) == "auto":
            file_export_name = str(title) + "_"
            file_export_name += "".join(selection)
            file_export_name += (
                "_size-"
                + str(fig_size[0])
                + "-" + str(fig_size[1])
            )
            file_export_name = (
                file_export_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace(":", "")
                .replace("/", "-")
                .replace("\\", "-")
                .replace("\n", "_")
            )

        if(
            isinstance(file_export_type, tuple)
            or isinstance(file_export_type, list)
        ):
            file_export_types = file_export_type
        else:
            file_export_types = (file_export_type, )

        for file_export_type in file_export_types:
            
            img_path = (
                Path(file_export_path)
                / (file_export_name + "." + file_export_type))
            plt.savefig(img_path, bbox_inches="tight")
            print("image was saved at", img_path)

        return fig, ax

    def _sensor_selection(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
    ):
        if(sensor_manual is None):
            selection = []
            for sensor in self.wtdl_int:
                if(
                    (sensor_type is None or sensor_type=="wtdl")
                    and (
                        sensor_locations is None
                        or sensor in sensor_locations
                    )
                ):
                    selection.append("W" + str(sensor))

            for sensor in self.sht_int:
                if(
                    (sensor_type is None or sensor_type=="sht")
                    and (
                        sensor_locations is None
                        or sensor in sensor_locations
                    )
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
        sensor_labels=types.MappingProxyType({}),
        encoding="ansi",
    ):
        """
        Imports all files from a directory.

        Parameters
        ----------
        directory: str or posix path object
        feedback: bool, optional
            Mute feedback about imported data.
        encoding: str, optional
            File encoding of WTDL .csv files.
        """
        self.sensor_labels = sensor_labels
        self.timeseries = {}
        self.dateseries = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(
                (),
                names=('sensor', 'key', 'unit'),
            )
        )
        self.wtdl_int = []
        self.wtdl_str = []
        self.sht_int = []
        self.sht_str = []
        self.sht_metadata = {}
        self.sht_sn = {}
        directory = Path(directory)
        for fn in directory.iterdir():

            # import WTDL sensor data
            match = re.match("^(W([0-9]+))[A-Za-z0-9_-]*\.csv$", fn.name)
            if(match):
                self.timeseries[match[1]] = self._import_wtld_file(
                    directory,
                    match[0],
                    encoding,
                )
                self.wtdl_str.append(match[1])
                self.wtdl_int.append(int(match[2]))

            # import SHT sensor data
            match = re.match("^(S([0-9]+))[A-Za-z0-9_-]*\.edf$", fn.name)
            if(match):
                self.timeseries[match[1]] = self._import_sht_file(
                    directory,
                    match[0],
                    match[1],
                )
                self.sht_str.append(match[1])
                self.sht_int.append(int(match[2]))

        # create single timeseries df
        for sensor in self.timeseries:
            self.timeseries[sensor]["T"].name = sensor

        if(feedback):
            print("Successfully imported the following sensor data:")
            print("    WTDL:")
            for wtdl in self.wtdl_str:
                print("        " + wtdl)
            print("    SHT:")
            for sht in self.sht_str:
                print("        " + sht + "  " + self.sht_sn[sht])

    def _import_wtld_file(self, directory, filename, encoding="ansi"):
        data = pd.read_csv(
            directory / filename,
            delimiter=";",
            encoding=encoding,
        )
        data.rename(columns={"Temperatur [°C]": "T"}, inplace=True)
        data["timestamp"] = data["Zeit [s]"].apply(self._parse_wtdl_datetime)
        data.set_index("timestamp", inplace=True)
        return data

    def _import_sht_file(self, directory, filename, sensor_code):
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
        data["timestamp"] = data["Local_Date_Time"].apply(
            self._parse_sht_datetime,
        )
        data.set_index("timestamp", inplace=True)
        with open(directory / filename) as f:
            self.sht_metadata[sensor_code] = {}
            for i in range(8):
                line = f.readline()
                match = re.match("^# ([A-Za-z]+)=(.+)$", line)
                if(match):
                    self.sht_metadata[sensor_code][match[1]] = match[2]
                else:
                    print("nothing found in " + line)
        self.sht_sn[sensor_code] = self.sht_metadata[sensor_code]["SensorId"]
        return data

    def _parse_wtdl_datetime(self, time_str):
        """
        Parse WTDL timestamps.
        
        Input Format: 05.07.2022 22:53:15
        Output Format datetime.datetime
        """
        match = re.match(
            r'^\s*([0-9]+)'
            r'\.([0-9]+)'
            r'\.([0-9]+)'
            r' ([0-9]+)'
            r':([0-9]+)'
            r':([0-9]+)\s*$',
            time_str
        )
        ints = np.zeros(6,dtype=np.int64)
        if(match is not None):
            for i in range(6):
                ints[i] = int(match[i+1])
        else:
            match = re.match(r'\s*([0-9]+).([0-9]+).([0-9]+)\s*', time_str)
            for i in range(3):
                ints[i] = int(match[i+1])
        ints[0], ints[2] = ints[2], ints[0]
        return dt.datetime(*ints)

    def _parse_sht_datetime(self, time_str, drop_ms=True):
        """
        Parse SHT timestamps.
        
        Input Format: 2022-07-12T13:42:15.622628
        Output Format datetime.datetime
        """
        match = re.match(
            r'^\s*([0-9]+)'
            r'-([0-9]+)'
            r'-([0-9]+)'
            r'T([0-9]+)'
            r':([0-9]+)'
            r':([0-9]+)'
            r'\.([0-9]+)\s*$',
            time_str,
        )
        ints = np.zeros(7,dtype=np.int64)
        if(match):
            for i in range(7):
                ints[i] = int(match[i+1])
        else:
            raise Exception("nothing found in "+time_str)
        if(drop_ms):
            ints[6] = 0
        return dt.datetime(*ints)

    def plot_temp_time(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
        fig_size=(10,6),
        fig_dpi=140,
        fig_legend_loc="upper right",
        xlim=None,
        ylim=None,
        title=None,
        xlabel="Datum/Zeit (MESZ)",
        ylabel="Temperatur / °C",
        file_export=False,
        file_export_path="",
        file_export_name="auto",
        file_export_type="pdf",
        show_plot=True,
        annot_func=None,
    ):
        selection = self._sensor_selection(
            sensor_type,
            sensor_locations,
            sensor_manual,
        )
        fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
        fig.set_facecolor("white")
        ax = fig.subplots()

        for sensor in selection:
            ax.plot(
                self.timeseries[sensor].index,
                self.timeseries[sensor]["T"],
                label=self.sensor_labels.get(sensor, sensor),
                ms=None,
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.legend(loc=fig_legend_loc)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if(annot_func is not None):
            fig, ax = annot_func(fig, ax)

        plt.tight_layout(pad=1.5)

        if(file_export):
            fig, ax = self._export_plot(
                fig,
                ax,
                file_export_name,
                file_export_path,
                file_export_type,
                title,
                selection,
                fig_size,
            )

        if(show_plot):
            plt.show()

    def plot_temp_time_interactive(
        self,
        *sensors,
        title=None,
        xlabel="Datum/Zeit (MESZ)",
        ylabel="Temperatur / °C",
        mode='lines',
        plot_all=False,
    ):
        if(len(sensors)==0):
            if(plot_all):
                sensors = self._sensor_selection()
            else:
                raise Exception(
                    "To avoid overload, "
                    "please confirm plotting all sensors "
                    "with plot_all=True"
                )
        fig = px.line(
            #self.timeseries[args[0]]["T"].dropna(),
            title=title,
            labels={
                "timestamp": xlabel,
                "value": ylabel,
                "sensor": "Sensor",
                "variable": "Sensor",
            },
            #name=sensor_labels.get(args[0], args[0]),
        )
        #for sensor in args[1:]:
        for sensor in sensors:
            fig.add_trace(
                go.Scatter(
                    x=self.timeseries[sensor]["T"].dropna().index,
                    y=self.timeseries[sensor]["T"].dropna(),
                    mode=mode,
                    name=self.sensor_labels.get(sensor, sensor),
                )
            )
        fig.show()

    def extract_dateseries(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
        timedelta_start={"hours":20},
        timedelta_width={"hours":1},
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
        if(type(timedelta_start) is dict):
            timedelta_start = dt.timedelta(**timedelta_start)
        if(type(timedelta_width) is dict):
            timedelta_width = dt.timedelta(**timedelta_width)

        # iterate every sensor
        for sensor in self.selection:
            timeserie = self.timeseries[sensor]

            if(date_earliest is None):
                date_earliest_sens = timeserie.index[0].date()
                date_earliest_sens = dt.datetime(
                    date_earliest_sens.year,
                    date_earliest_sens.month,
                    date_earliest_sens.day,
                )
            else:
                date_earliest_sens = date_earliest

            if(date_earliest is None):
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
                     time_start.strftime('%Y-%m-%d %H:%M:%S.%f')
                     : time_stop.strftime('%Y-%m-%d %H:%M:%S.%f')
                ]

                if(
                    filtered.shape[0] >= min_count
                    and (not ignore_dates or date not in ignore_dates)
                ):
                    if(average_alg == "mean"):
                        self.dateseries.loc[date, (sensor, key, "T")] = filtered["T"].mean()
                    elif(average_alg == "median"):
                        self.dateseries.loc[date, (sensor, key, "T")] = filtered["T"].median()
                    else:
                        raise Exception("average_alg " + average_alg + " does not exist")
                self.dateseries.loc[date, (sensor, key, "count")] = filtered.shape[0]

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

    def binned_delta(
        self,
        key_binned,
        key_ref,
        key_2,
        ref_sensors=None,
        n_bins=5,
        bounds=(None, None),
        average_alg="mean",
        detailed_analysis=False,
    ):
        if(ref_sensors is None):
            ref_sensors = self.selection

        return Binned(
            self.dateseries,
            key_ref,
            key_2,
            ref_sensors,
            n_bins,
            bounds,
            average_alg,
            detailed_analysis,
            self.wtdl_int,
            self.wtdl_str,
            self.sht_int,
            self.sht_str,
            self.sht_metadata,
            self.sht_sn,
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
                names=('sensor', 'key', 'unit'),
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
        if(frame_ref):
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
            if(date_earliest is None):
                date_earliest_sens = timeserie.index[0].date()
                date_earliest_sens = dt.datetime(
                    date_earliest_sens.year,
                    date_earliest_sens.month,
                    date_earliest_sens.day,
                )
            else:
                date_earliest_sens = date_earliest

            if(date_earliest is None):
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
                     time_start.strftime('%Y-%m-%d %H:%M:%S.%f')
                     : time_stop.strftime('%Y-%m-%d %H:%M:%S.%f')
                ]

                if(
                    filtered.shape[0] >= min_count
                    and (not ignore_dates or date not in ignore_dates)
                ):
                    if(self.average_alg == "mean"):
                        self.dateseries.loc[
                            date,
                            (sensor, key, "T"),
                        ] = filtered["T"].mean()
                    elif(self.average_alg == "median"):
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
        if(
            not isinstance(ref_sensors, tuple)
            and not isinstance(ref_sensors, list)
        ):
            ref_sensors = (ref_sensors, )
        ref_sensors = ref_sensors
        self.bins[key] = {}

        # calculate daily reference temperatures based on ref_sensors
        if(self.average_alg == "mean"):
            self.dateseries.loc[
                :,
                ("binning", key, "ref_T"),
            ] = self.dateseries.loc[
                :,
                (ref_sensors, self.ref_key, "T"),
            ].droplevel(("key", "unit"), axis="columns").mean(axis=1)
        elif(self.average_alg == "median"):
            self.dateseries.loc[
                :,
                ("binning", key, "ref_T"),
            ] = self.dateseries.loc[
                :,
                (ref_sensors, self.ref_key, "T"),
            ].droplevel(("key", "unit"), axis="columns").median(axis=1)

        # assign bin
        self.dateseries.loc[
            :,
            ("binning", key, "bin_nr"),
        ], self.bins[key] = pd.cut(
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
        frames=("21-1", "27-1",),
        boxplot=True,
        boxplot_and_line=False,
        fig_size=(10,6),
        fig_dpi=140,
        fig_legend_loc="upper right",
        xlim=None,
        ylim=None,
        title=None,
        xlabel="Tages-Referenztemperatur / °C",
        ylabel="Temperaturabfall / °C",
        file_export=False,
        file_export_path="",
        file_export_name="auto",
        file_export_type="pdf",
        show_plot=True,
        annot_func=None,
    ):
        selection = self._sensor_selection(
            sensor_type,
            sensor_locations,
            sensor_manual,
        )
        fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
        fig.set_facecolor("white")
        ax = fig.subplots()

        # compute data
        y = []
        x = []
        n = []
        for i in range(len(self.bins[bin_key]) - 1):
            subset = self.dateseries[self.dateseries[("binning", bin_key, "bin_nr")] == i].loc[:, (selection, )]

            # delta
            if(isinstance(frames, tuple) or isinstance(frames, list)):
                sub_1 = subset.loc[
                    :,
                    (slice(None), frames[0], "T"),
                ].droplevel((1,2,), axis=1)
                sub_2 = subset.loc[
                    :,
                    (slice(None), frames[1], "T"),
                ].droplevel((1,2,), axis=1)
                y.append((sub_1 - sub_2).mean(axis=1))

            # absolute value
            else:
                y.append(subset.loc[
                    :,
                    (slice(None), frames, "T"),
                ].droplevel((1,2,), axis=1).mean(axis=1))

            x.append(i)
            n.append(len(subset))

        # boxplot (complex)
        if(boxplot or boxplot_and_line):
            ax.boxplot(y, labels=x)
            ax.xaxis.set_tick_params(which='minor', bottom=False)

        # regular line plot (simple)
        if(not boxplot or boxplot_and_line):
            y = [e.median() for e in y]
            if(boxplot_and_line):
                ax.plot(np.array(x) + 1, y)
            else:
                x = (self.bins[bin_key][:-1] + self.bins[bin_key][1:]) / 2
                ax.plot(x, y, "x-")

        # rewrite x axis labels
        if(boxplot or boxplot_and_line):
            ax.set_xticklabels([
                "{:.1f}\n({:.1f} - {:.1f})\n(n={})".format(
                    (self.bins[bin_key][i] + self.bins[bin_key][i + 1]) / 2,
                    self.bins[bin_key][i],
                    self.bins[bin_key][i + 1],
                    n[i],
                )
                for i
                in range(len(self.bins[bin_key]) - 1)
            ])

        # format plot
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        #ax.legend(loc=fig_legend_loc)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if(annot_func is not None):
            fig, ax = annot_func(fig, ax)

        plt.tight_layout(pad=1.5)

        if(file_export):
            fig, ax = self._export_plot(
                fig,
                ax,
                file_export_name,
                file_export_path,
                file_export_type,
                title,
                selection,
                fig_size,
            )

        if(show_plot):
            plt.show()

    def plot_scatter(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
        bin_key="default",
        frames=("21-1", "27-1",),
        confidence_interval=True,
        fig_size=(10,6),
        fig_dpi=140,
        fig_legend_loc="upper right",
        xlim=None,
        ylim=None,
        title=None,
        xlabel="Tages-Referenztemperatur / °C",
        ylabel="Temperaturabfall / °C",
        file_export=False,
        file_export_path="",
        file_export_name="auto",
        file_export_type="pdf",
        show_plot=True,
        annot_func=None,
    ):
        selection = self._sensor_selection(
            sensor_type,
            sensor_locations,
            sensor_manual,
        )
        fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
        fig.set_facecolor("white")
        ax = fig.subplots()

        # compute data
        # delta
        if(isinstance(frames, tuple) or isinstance(frames, list)):
            sub_1 = self.dateseries.loc[
                :,
                (selection, frames[0], "T"),
            ].droplevel((1,2,), axis=1)
            sub_2 = self.dateseries.loc[
                :,
                (selection, frames[1], "T"),
            ].droplevel((1,2,), axis=1)
            y = (sub_1 - sub_2).mean(axis=1)

        # absolute value
        else:
            y = self.dateseries.loc[
                :,
                (selection, frames, "T"),
            ].droplevel((1,2,), axis=1).mean(axis=1)

        x = self.dateseries[("binning", bin_key, "ref_T")]

        # clean nan values
        cleaned = ~pd.DataFrame({"x": x.isnull(), "y": y.isnull()}).any(axis=1)
        x = x.loc[cleaned]
        y = y.loc[cleaned]

        # scatter plot
        ax.scatter(x, y, zorder=10)

        # confidence interval
        if(confidence_interval):
            p, cov = np.polyfit(x, y, 1, cov=True)  # parameters and covariance from of the fit of 1-D polynom.
            y_model = equation(p, x)                # model using the fit parameters; NOTE: parameters here are coefficients

            # Statistics
            n = y.size                              # number of observations
            m = p.size                              # number of parameters
            dof = n - m                             # degrees of freedom
            t = sp.stats.t.ppf(0.975, n - m)        # t-statistic; used for CI and PI bands

            # Estimates of Error in Data/Model
            resid = y - y_model                     # residuals; diff. actual data from predicted values
            chi2 = np.sum((resid / y_model)**2)     # chi-squared; estimates error in data
            chi2_red = chi2 / dof                   # reduced chi-squared; measures goodness of fit
            s_err = np.sqrt(np.sum(resid**2) / dof) # standard deviation of the error

            # Fit
            ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")  

            x2 = np.linspace(np.min(x), np.max(x), 100)
            y2 = equation(p, x2)

            # Confidence Interval (select one)
            ax = plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
            #plot_ci_bootstrap(x, y, resid, ax=ax)
               
            # Prediction Interval
            pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))   
            ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
            ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
            ax.plot(x2, y2 + pi, "--", color="0.5")



        # format plot
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.legend(loc=fig_legend_loc)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if(annot_func is not None):
            fig, ax = annot_func(fig, ax)

        plt.tight_layout(pad=1.5)

        if(file_export):
            fig, ax = self._export_plot(
                fig,
                ax,
                file_export_name,
                file_export_path,
                file_export_type,
                title,
                selection,
                fig_size,
            )

        if(show_plot):
            plt.show()

def equation(a, b):
    """Return a 1D polynomial."""
    return np.polyval(a, b)

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    r"""Return an axes of confidence bands using a simple approach.
    
    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}
    
    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb
    
    Example found on: https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
    """
    if ax is None:
        ax = plt.gca()
    
    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", alpha=0.5)

    return ax



################################################################################
################################################################################
################## DEPTECATED CODE BELOW #######################################
################################################################################
################################################################################





"""
        # calculate temperature drops for every sensor and every day
        self.t_drop = (
            dateseries.loc[
                :,
                (slice(None), key_ref, "T"),
            ].droplevel(("key", "unit"), axis="columns")
            - dateseries.loc[
                :,
                (slice(None), key_2, "T"),
            ].droplevel(("key", "unit"), axis="columns")
        )

        # bin temperature drop for every sensor
        self.binned_data = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(
                (),
                names=('sensor', 'value'),
            ),
            index=range(n_bins),
        )
        for sensor, _ in self.t_drop.iloc[0].items():
            self.binned_data.loc[
                :,
                (sensor, "t_drop_sum"),
            ] = np.zeros(n_bins)
            self.binned_data.loc[
                :,
                (sensor, "count"),
            ] = np.zeros(n_bins)
            self.binned_data.loc[
                :,
                (sensor, "t_drop_avg"),
            ] = np.zeros(n_bins)
        for day, ref_temp in self.ref_temps.items():
            bin_ = np.digitize(ref_temp, self.edges) - 1
            if(bin_ >= n_bins):
                continue
            for sensor, t_drop in self.t_drop.loc[day].items():
                if(pd.isnull(t_drop)):
                    continue
                self.binned_data.loc[
                    bin_,
                    (sensor, "t_drop_sum"),
                ] = self.binned_data.loc[
                    bin_,
                    (sensor, "t_drop_sum"),
                ] + t_drop
                self.binned_data.loc[
                    bin_,
                    (sensor, "count"),
                ] = self.binned_data.loc[
                    bin_,
                    (sensor, "count"),
                ] + 1
        for sensor, df in self.binned_data.groupby(level=0, axis=1):
            df = df.droplevel(
                "sensor",
                axis="columns",
            )
            self.binned_data.loc[
                :,
                (sensor, "t_drop_avg"),
            ] = df["t_drop_sum"] / df["count"]
        self.binned = self.binned_data.loc[
            :,
            (slice(None), "t_drop_avg"),
        ].droplevel(
            "value",
            axis="columns",
        )
        """


        



class Binned_deprecated:
    
    def __init__(
        self,
        dateseries,
        key_ref,
        key_2,
        ref_sensors,
        n_bins,
        bounds,
        average_alg,
        detailed_analysis,
        wtdl_int,
        wtdl_str,
        sht_int,
        sht_str,
        sht_metadata,
        sht_sn,
    ):

        self.wtdl_int = wtdl_int
        self.wtdl_str = wtdl_str
        self.sht_int = sht_int
        self.sht_str = sht_str
        self.sht_metadata = sht_metadata
        self.sht_sn = sht_sn

        # determine ref_sensors
        if(
            not isinstance(ref_sensors, tuple)
            and not isinstance(ref_sensors, list)
        ):
            ref_sensors = (ref_sensors, )
        self.ref_sensors = ref_sensors

        # determine boundaries for bins based on ref_sensors
        if(bounds is None or bounds[0] is None or bounds[1] is None):
            bounds_iter = pd.DataFrame(index=ref_sensors, columns=("min", "max"))
            for sensor in ref_sensors:
                bounds_iter.loc[sensor] = (
                    dateseries.loc[:, (sensor, key_ref, "T")].min(),
                    dateseries.loc[:, (sensor, key_ref, "T")].max(),
                )
            if(bounds is None or (bounds[0] is None and bounds[1] is None)):
                bounds = (
                    bounds_iter["min"].min(),
                    bounds_iter["max"].max(),
                )
            elif(bounds[0] is None):
                bounds = (
                    bounds_iter["min"].min(),
                    bounds[1],
                )
            else:
                bounds = (
                    bounds[0],
                    bounds_iter["max"].max(),
                )

        # derive edges of bins
        self.edges = np.linspace(*bounds, n_bins + 1)

        # calculate daily reference temperatures based on ref_sensors
        if(average_alg == "mean"):
            self.ref_temps = dateseries.loc[
                :,
                (ref_sensors, key_ref, "T"),
            ].droplevel(("key", "unit"), axis="columns").mean(axis=1)
        elif(average_alg == "median"):
            self.ref_temps = dateseries.loc[
                :,
                (ref_sensors, key_ref, "T"),
            ].droplevel(("key", "unit"), axis="columns").median(axis=1)

        # calculate temperature drops for every sensor and every day
        self.t_drop = (
            dateseries.loc[
                :,
                (slice(None), key_ref, "T"),
            ].droplevel(("key", "unit"), axis="columns")
            - dateseries.loc[
                :,
                (slice(None), key_2, "T"),
            ].droplevel(("key", "unit"), axis="columns")
        )

        # bin temperature drop for every sensor
        self.binned_data = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(
                (),
                names=('sensor', 'value'),
            ),
            index=range(n_bins),
        )
        for sensor, _ in self.t_drop.iloc[0].items():
            self.binned_data.loc[
                :,
                (sensor, "t_drop_sum"),
            ] = np.zeros(n_bins)
            self.binned_data.loc[
                :,
                (sensor, "count"),
            ] = np.zeros(n_bins)
            self.binned_data.loc[
                :,
                (sensor, "t_drop_avg"),
            ] = np.zeros(n_bins)
        for day, ref_temp in self.ref_temps.items():
            bin_ = np.digitize(ref_temp, self.edges) - 1
            if(bin_ >= n_bins):
                continue
            for sensor, t_drop in self.t_drop.loc[day].items():
                if(pd.isnull(t_drop)):
                    continue
                self.binned_data.loc[
                    bin_,
                    (sensor, "t_drop_sum"),
                ] = self.binned_data.loc[
                    bin_,
                    (sensor, "t_drop_sum"),
                ] + t_drop
                self.binned_data.loc[
                    bin_,
                    (sensor, "count"),
                ] = self.binned_data.loc[
                    bin_,
                    (sensor, "count"),
                ] + 1
        for sensor, df in self.binned_data.groupby(level=0, axis=1):
            df = df.droplevel(
                "sensor",
                axis="columns",
            )
            self.binned_data.loc[
                :,
                (sensor, "t_drop_avg"),
            ] = df["t_drop_sum"] / df["count"]
        self.binned = self.binned_data.loc[
            :,
            (slice(None), "t_drop_avg"),
        ].droplevel(
            "value",
            axis="columns",
        )

    def _sensor_selection(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
    ):
        if(sensor_manual is None):
            selection = []
            for sensor in self.wtdl_int:
                if(
                    (sensor_type is None or sensor_type=="wtdl")
                    and (
                        sensor_locations is None
                        or sensor in sensor_locations
                    )
                ):
                    selection.append("W" + str(sensor))
            for sensor in self.sht_int:
                if(
                    (sensor_type is None or sensor_type=="sht")
                    and (
                        sensor_locations is None
                        or sensor in sensor_locations
                    )
                ):
                    selection.append("S" + str(sensor))

        else:
            return sensor_manual

        return selection

    def plot_t_drop(
        self,
        sensor_type=None,
        sensor_locations=None,
        sensor_manual=None,
        fig_size=(10,6),
        fig_dpi=140,
        fig_legend_loc="upper right",
        xlim=None,
        ylim=None,
        title="Standortabhängiger Temperaturabfall nach Tages-Referenztemperatur",
        xlabel="Referenztemperatur / °C",
        ylabel="Temperaturabfall / °C",
        file_export=False,
        file_export_path="",
        file_export_name="auto",
        file_export_type="pdf",
        show_plot=True,
    ):
        selection = self._sensor_selection(
            sensor_type,
            sensor_locations,
            sensor_manual,
        )
        fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
        ax = fig.subplots()
        
        fig.set_facecolor("white")
        
        for sensor in selection:
            ax.plot(
                (self.edges[1:] + self.edges[:-1]) / 2,
                self.binned[sensor],
                label=sensor,
                ms=None,
            )
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        plt.legend(loc=fig_legend_loc)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.tight_layout(pad=1.5)

        if(file_export):
            if str(file_export_name) == "auto":
                file_export_name = title + "_"
                file_export_name += "".join(selection)
                file_export_name += (
                    "_size-"
                    + str(fig_size[0])
                    + "-" + str(fig_size[1])
                )
                file_export_name = (
                    file_export_name.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(",", "")
                    .replace("/", "-")
                    .replace("\\", "-")
                )
                
            img_path = os.path.join(
                file_export_path,
                file_export_name + "." + file_export_type,
            )
            plt.savefig(img_path, face_color="white", bbox_inches="tight")
            print("image was saved at", img_path)

        if(show_plot):
            plt.show()






