# -*- coding: utf-8 -*-
"""
@author: joerja

V0.2.0

220821
"""

import os
import re
import datetime as dt
import functools as ft

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

class Timed:

    def __init__(
        self,
        directory,
        feedback=True,
    ):
        """Initialize Waldluft."""
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
        objs = os.scandir(directory)
        for obj in objs:

            # import WTDL sensor data
            match = re.match("^(W([0-9]))+\.csv$", obj.name)
            if(match):
                self.timeseries[match[1]] = self._import_wtld_file(
                    directory,
                    match[0],
                )
                self.wtdl_str.append(match[1])
                self.wtdl_int.append(int(match[2]))

            # import SHT sensor data
            match = re.match("^(S([0-9]))+\.edf$", obj.name)
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
        self.timeseries_ = pd.concat(
            [self.timeseries[sensor]["T"] for sensor in self.timeseries],
            axis=1,
        )

        if(feedback):
            print("Successfully imported the following sensor data:")
            print("    WTDL:")
            for wtdl in self.wtdl_str:
                print("        " + wtdl)
            print("    SHT:")
            for sht in self.sht_str:
                print("        " + sht + "  " + self.sht_sn[sht])

    def _import_wtld_file(self, directory, filename):
        data = pd.read_csv(
            os.path.join(directory, filename),
            delimiter=";",
            encoding="ansi",
        )
        data.rename(columns={"Temperatur [°C]": "T"}, inplace=True)
        data["timestamp"] = data["Zeit [s]"].apply(self._parse_wtdl_datetime)
        data.set_index("timestamp", inplace=True)
        return data

    def _import_sht_file(self, directory, filename, sensor_code):
        data = pd.read_csv(
            os.path.join(directory, filename),
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
        with open(os.path.join(directory, filename)) as f:
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
                self.timeseries[sensor].index,
                self.timeseries[sensor]["T"],
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


    def plot_temp_time_interactive(
        self,
        *args,
        title=None,
        xlabel="Datum/Zeit (MESZ)",
        ylabel="Temperatur / °C",
        mode='lines',
        plot_all=False,
    ):
        if(len(args)==0):
            if(plot_all):
                args = self._sensor_selection()
            else:
                raise Exception(
                    "To avoid overload, "
                    "please confirm plotting all sensors "
                    "with plot_all=True"
                )
        fig = px.line(
            self.timeseries[args[0]]["T"].dropna(),
            title=title,
            labels={
                "timestamp": xlabel,
                "value": ylabel,
                "sensor": "Sensor",
                "variable": "Sensor",
            }
        )
        for arg in args[1:]:
            fig.add_trace(
                go.Scatter(
                    x=self.timeseries[arg]["T"].dropna().index,
                    y=self.timeseries[arg]["T"].dropna(),
                    mode=mode,
                    name=arg,
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
            self.wtdl_int,
            self.wtdl_str,
            self.sht_int,
            self.sht_str,
            self.sht_metadata,
            self.sht_sn,
        )

    def binned_delta_deprecated(
        self,
        key_binned,
        key_ref,
        key_2,
        ref_sensors=None,
        n_bins=5,
        bounds=(None, None),
        average_alg="mean",
    ):
        self.binned[key_binned] = {}

        # determine ref_sensors
        if(ref_sensors is None):
            ref_sensors = self.selection
        if(
            not isinstance(ref_sensors, tuple)
            and not isinstance(ref_sensors, list)
        ):
            ref_sensors = (ref_sensors, )

        # determine boundaries for bins based on ref_sensors
        if(bounds is None or bounds[0] is None or bounds[1] is None):
            bounds_iter = pd.DataFrame(index=ref_sensors, columns=("min", "max"))
            for sensor in ref_sensors:
                bounds_iter.loc[sensor] = (
                    self.dateseries.loc[:, (sensor, key_ref, "T")].min(),
                    self.dateseries.loc[:, (sensor, key_ref, "T")].max(),
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
        self.binned[key_binned]["edges"] = np.linspace(*bounds, n_bins + 1)

        # calculate daily reference temperatures based on ref_sensors
        if(average_alg == "mean"):
            self.binned[key_binned]["ref_temps"] = self.dateseries.loc[
                :,
                (ref_sensors, key_ref, "T"),
            ].droplevel(("key", "unit"), axis="columns").mean(axis=1)
        elif(average_alg == "median"):
            self.binned[key_binned]["ref_temps"] = self.dateseries.loc[
                :,
                (ref_sensors, key_ref, "T"),
            ].droplevel(("key", "unit"), axis="columns").median(axis=1)

        # calculate temperature drops for every sensor and every day
        self.binned[key_binned]["t_drop"] = (
            self.dateseries.loc[
                :,
                (slice(None), key_ref, "T"),
            ].droplevel(("key", "unit"), axis="columns")
            - self.dateseries.loc[
                :,
                (slice(None), key_2, "T"),
            ].droplevel(("key", "unit"), axis="columns")
        )

        # bin temperature drop for every sensor
        self.binned[key_binned]["hist"] = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(
                (),
                names=('sensor', 'value'),
            ),
            index=range(n_bins),
        )
        for sensor, _ in self.binned[key_binned]["t_drop"].iloc[0].iteritems():
            self.binned[key_binned]["hist"].loc[
                :,
                (sensor, "t_drop_sum"),
            ] = np.zeros(n_bins)
            self.binned[key_binned]["hist"].loc[
                :,
                (sensor, "count"),
            ] = np.zeros(n_bins)
            self.binned[key_binned]["hist"].loc[
                :,
                (sensor, "t_drop_avg"),
            ] = np.zeros(n_bins)
        for day, ref_temp in self.binned[key_binned]["ref_temps"].iteritems():
            bin_ = np.digitize(ref_temp, self.binned[key_binned]["edges"]) - 1
            if(bin_ >= n_bins):
                continue
            for sensor, t_drop in self.binned[key_binned]["t_drop"].loc[day].iteritems():
                if(pd.isnull(t_drop)):
                    continue
                self.binned[key_binned]["hist"].loc[
                    bin_,
                    (sensor, "t_drop_sum"),
                ] = self.binned[key_binned]["hist"].loc[
                    bin_,
                    (sensor, "t_drop_sum"),
                ] + t_drop
                self.binned[key_binned]["hist"].loc[
                    bin_,
                    (sensor, "count"),
                ] = self.binned[key_binned]["hist"].loc[
                    bin_,
                    (sensor, "count"),
                ] + 1
        for sensor, df in self.binned[key_binned]["hist"].groupby(level=0, axis=1):
            df = df.droplevel(
                "sensor",
                axis="columns",
            )
            self.binned[key_binned]["hist"].loc[
                :,
                (sensor, "t_drop_avg"),
            ] = df["t_drop_sum"] / df["count"]
        """
        self.binned[key_binned]["hist"].loc[
            :,
            (slice(None), "t_avg"),
        ] = self.binned[key_binned]["hist"].loc[
            :,
            (slice(None), "t_drop_sum"),
        ].droplevel(
            ("value"),
            axis="columns",
        ) / self.binned[key_binned]["hist"].loc[
            :,
            (slice(None), "count"),
        ].droplevel(
            ("value"),
            axis="columns",
        )
        """

            
class Binned:
    
    def __init__(
        self,
        dateseries,
        key_ref,
        key_2,
        ref_sensors,
        n_bins,
        bounds,
        average_alg,
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
        for sensor, _ in self.t_drop.iloc[0].iteritems():
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
        for day, ref_temp in self.ref_temps.iteritems():
            bin_ = np.digitize(ref_temp, self.edges) - 1
            if(bin_ >= n_bins):
                continue
            for sensor, t_drop in self.t_drop.loc[day].iteritems():
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
        title="Standordabhängiger Temperaturabfall nach Tages-Referenztemperatur",
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






