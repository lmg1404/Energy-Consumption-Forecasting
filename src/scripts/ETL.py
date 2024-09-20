"""
Object is to put in notebooks so you just do ETL with settings and run it
Should put .csvs into the data folder easily
"""

import pandas as pd
from collections import defaultdict
import requests
from typing import List
import os
import io
import numpy as np
import re
import glob
import csv

MI_LAT = [41.50, 47.50]
MI_LONG = [-90.5, -82.5]


class ETLConfig:
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)

  def __getattr__(self, name):
    return None

class ETL:
  def __init__(self, config):
    self.start_year = config.start_year
    self.end_year = config.end_year
    self.data_path = config.data_path
    self.station_file = config.station_file
    self.ghcd_path = config.ghcd_path
  
  def _check_columns(self, dataframes: pd.DataFrame) -> List[pd.DataFrame]:
    col_hash = defaultdict(list)
    for df in dataframes:
        col_len = len(df.columns)
        col_hash[col_len].append(df)
    max_col = max(col_hash.keys())
    while len(col_hash) > 1:
        min_col = min(col_hash.keys())
        cols = set(col_hash[max_col][0].columns)
        cur_df = col_hash[min_col].pop()
        cur_cols = cur_df.columns
        diff = list(cols.difference(cur_cols))
        cur_df[diff] = np.nan
        col_hash[max_col].append(cur_df)
        if len(col_hash[min_col]) == 0:
            del col_hash[min_col]
    return col_hash[max_col]
  
  def _extract_csvs(self) -> List[pd.DataFrame]:
    dfs = []
    for year in range(self.start_year, self.end_year+1):
        firsthalf_link = f"https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{year}_Jan_Jun.csv"
        sechalf_link = f"https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{year}_Jul_Dec.csv"
        
        r1 = requests.get(firsthalf_link)
        r2 = requests.get(sechalf_link)
        try:
            data1 = r1.content.decode('utf8')
            data2 = r2.content.decode('utf8')
            d1 = pd.read_csv(io.StringIO(data1), low_memory=False)
            d2 = pd.read_csv(io.StringIO(data2), low_memory=False)
            df = pd.concat([d1, d2], axis=0)
        
            cols = [col for col in df.columns if "Imputed" not in col and "Adjusted" in col]
            columns = list(df.columns[:4]) + cols + ["Region"]
            midw = df[(df['Region'] == "MIDW") & (df['Balancing Authority'] == "MISO")][columns]
            dfs.append(midw)
        except Exception as e:
            print(e)
    return dfs
  
  def balance_sheets(self) -> None:
    threshold = 0.8
    pattern = r'\([^()]*\)|\b(from|at|of)\b'
    print("  Extracting .CSV's")
    dfs = self._extract_csvs()
    dfs = self._check_columns(dfs)
    print("  Concatenating DataFrames and Fixing Columns")
    master_df = pd.concat(dfs, axis=0, ignore_index=True)
    master_df.columns = ["_".join(re.sub(pattern, '', col).lower().split()) for col in master_df.columns]
    master_df["local_time_end_hour"] = pd.to_datetime(master_df["local_time_end_hour"])
    master_df =  master_df \
        .sort_values("local_time_end_hour",  ignore_index=True) \
        .dropna(axis=1, thresh=int(len(master_df) * (1-threshold))) \
        .dropna(axis=0, thresh=7) \
        .bfill(axis=0)
    print("  Saved to a .CSV as balance_sheet.csv")
    master_df.to_csv(os.path.join(self.data_path, "balance_sheet.csv"), index=False)
    
  def dly_convert(self) -> None:
    fields = [
      ["ID", 1, 11],
      ["YEAR", 12, 15],
      ["MONTH", 16, 17],
      ["ELEMENT", 18, 21]
    ]
    
    offset = 22
    
    for value in range(1, 32):
      fields.append((f"VALUE{value}", offset,     offset + 4))
      fields.append((f"MFLAG{value}", offset + 5, offset + 5))
      fields.append((f"QFLAG{value}", offset + 6, offset + 6))
      fields.append((f"SFLAG{value}", offset + 7, offset + 7))
      offset += 8
    
    # Modify fields to use Python numbering
    fields = [[var, start - 1, end] for var, start, end in fields]
    fieldnames = [var for var, start, end in fields]
    for dly_filename in glob.glob(os.path.join(self.ghcd_path, '*.dly'), recursive=True): 
      path, name = os.path.split(dly_filename)
      csv_filename = os.path.join(path, f"{os.path.splitext(name)[0]}.csv")

      with open(dly_filename, newline='') as f_dly, open(csv_filename, 'w', newline='') as f_csv:
        csvout = csv.writer(f_csv)
        csvout.writerow(fieldnames)    # Write a header using the var names

        for line in f_dly:
          row = [line[start:end].strip() for var, start, end in fields]
          csvout.writerow(row)
    
  def get_station_df(self) -> pd.DataFrame:
    station_df = pd.read_fwf(
      self.station_file,
      header=None,
      names=["ID", "lat", "long", "elev", "city", "unk1", "unk2", "unk3"],
    )

    return station_df
    
  def get_station_list(self, df: pd.DataFrame, lat: List[float], lon: List[float], US: bool = False) -> List[str]:
    # df is the weather station dataframe created from txt file provided by NOAA
    # lat is a list of lattitude range (max length 2) with element 0 being the minimum and 1 being the max
    # lon is a list of longitude range (max length 2) with element 0 being the minimum and 1 being the max
    # set US to true to return only US listed weather stations

    if len(lat) > 2:
        return "Error you can only have 2 values for lattitude"
    if len(lon) > 2:
        return "Error you can only have 2 values for longitude"

    station_df = df[
        (df.lat >= lat[0])
        & (df.lat <= lat[1])
        & (df.long >= lon[0])
        & (df.long <= lon[1])
    ]

    if US:
        station_df = station_df.loc[station_df["ID"].str.contains("US")]

    station_lst = station_df.ID.tolist()

    return station_lst
  
  def combine_stations(self, stations: List[str]) -> pd.DataFrame:
    # path is file path where files are located

    # all files creates list of csv files within path folder
    all_files = glob.glob(os.path.join(self.ghcd_path, "*.csv"))

    df_list = []
    # iterate creating dfs for each csv to get all US weather stations
    for file in all_files:
        df = pd.read_csv(file, index_col=None, header=0, low_memory=False)
        df = df[df["ID"].isin(stations)]
        if len(df) > 0:
            df_list.append(df)

    weather_df = pd.concat(df_list, axis=0, ignore_index=True)

    return weather_df
  
  def filter_weather(self, df: pd.DataFrame, filtword: str = None) -> pd.DataFrame:
    # min year is the first year of desired data and max year is final year. One can be left blank if getting data up to or starting from a year. Year is type integer
    # filtword is the filter word (string) used to drop columns. If left none than no columns are dropped

    if filtword:
      col_list = df.columns.tolist()
      newcols = []
      for col in col_list:
          if filtword not in col:
              newcols.append(col)
    else:
      newcols = df.columns.tolist()

    final_df = df[newcols]

    if self.start_year:
      final_df = final_df[final_df.YEAR >= self.start_year]

    if self.end_year:
      final_df = final_df[final_df.YEAR <= self.end_year]

    return final_df
  
  def get_pivotdf(self, df: pd.DataFrame) -> pd.DataFrame:
    values = df.columns.tolist()[4:]
    indcols = df.columns.tolist()[:4]

    melt_df = pd.melt(df, id_vars=indcols, value_vars=values, var_name="DAY")

    pivot_df = pd.pivot_table(
      melt_df,
      values="value",
      index=["ID", "YEAR", "MONTH", "DAY"],
      columns="ELEMENT",
      aggfunc="first",
    ).reset_index()
    pivot_df["DAY"] = pivot_df["DAY"].str.replace(r"\D", "", regex=True).astype(int)
    pivot_df = pivot_df.sort_values(by=["YEAR", "MONTH", "DAY"])
    pivot_df.columns.name = None
    pivot_df = pivot_df \
      .replace(-9999.0, np.nan) \
      .dropna(how="all", axis=0) \
      .dropna(how="all", axis=1)

    return pivot_df
  
  def fill_missing(self, df: pd.DataFrame, limit: int = 7) -> pd.DataFrame:
    # limit is how many days prior to fill in missing values

    # get list of stations
    stations = df["ID"].unique().tolist()
    dfs = []
    for station in stations:
      res = df[df["ID"] == station]
      res = res.ffill(limit=limit)
      res = res.bfill(limit=limit)
      dfs.append(res)

    final_df = pd.concat(dfs, axis=0, ignore_index=True)

    return final_df
  
  def date_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    months = [2, 4, 6, 9, 11]
    days = [31]

    # make list of indices where 30 months have 31 days
    inds = df2[(df2["MONTH"].isin(months)) & (df2["DAY"].isin(days))].index.tolist()

    # add indices where february should only have 28 days and 29 days
    month = [2]
    leap_years = [2016, 2020, 2024]
    non_leap = [2015, 2017, 2018, 2019, 2021, 2022, 2023]
    noleapdays = [29, 30]
    leapday = [30]

    l_inds = df2[
        (df2["MONTH"].isin(month))
        & (df2["YEAR"].isin(leap_years))
        & (df2["DAY"].isin(leapday))
    ].index.tolist()

    nol_inds = df2[
        (df2["MONTH"].isin(month))
        & (df2["YEAR"].isin(non_leap))
        & (df2["DAY"].isin(noleapdays))
    ].index.tolist()

    inds_final = inds + l_inds + nol_inds

    # drop rows with days that shouldn't exist for months with 30 days and february accordingly
    df2.drop(inds_final, inplace=True)

    # create new date column and convert to date time
    df2["DATE"] = (
        df2["MONTH"].astype(str)
        + "/"
        + df2["DAY"].astype(str)
        + "/"
        + df2["YEAR"].astype(str)
    )

    df2["DATE"] = pd.to_datetime(df2["DATE"]).dt.date

    # reorder columns to show new date column and omit year month and day columns
    features = df2.columns.tolist()[4:-1]
    ids = df2.columns.tolist()[:1]
    ids.append("DATE")
    cols = ids + features

    return df2[cols]
  
  def add_location(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # df1 is the stations dataframe generated from the txt document
    # df2 is the cleaned up weather dataframe
    final_stations_l = df2.ID.unique().tolist()

    df1 = df1[df1["ID"].isin(final_stations_l)]
    df1 = df1[["ID", "city", "lat", "long", "elev"]]

    df_final = df2.merge(df1, left_on="ID", right_on="ID")

    cols = df_final.columns.tolist()
    cols_f = cols[:2] + cols[22:] + cols[2:22]

    df_final = df_final[cols_f]
    df_final = df_final.rename(columns={"lat": "latitude", "long": "longitude"})

    return df_final.sort_values(by=["ID", "DATE"])
  
  def generate_weather(self) -> None:
    print("  Getting Relevant Stations")
    station_df = self.get_station_df()
    stations = self.get_station_list(station_df, MI_LAT, MI_LONG, True)
    print("  Combining Stations")
    weather_df = self.combine_stations(stations)
    print("  Filtering Weather")
    weather_df = self.filter_weather(weather_df, "FLAG")
    weather_df = self.get_pivotdf(weather_df)
    print("  Cleaning Operations")
    weather_df = self.fill_missing(weather_df)
    weather_df = self.date_cleanup(weather_df)
    weather_df = self.add_location(station_df, weather_df)
    weather_df.to_csv(os.path.join(self.data_path, "WeatherReport.csv"), index=False)
  
  def run(self, balance_sheet: bool, dly_convert: bool, create_weather: bool) -> None:
    if balance_sheet:
      print("Starting Balance Sheet Processing")
      self.balance_sheets()
      
    if dly_convert:
      print("Starting .DLY Converts")
      self.dly_convert()
      
    if create_weather:
      print("Generating a Weather Report")
      self.generate_weather()
      
if __name__ == "__main__":
  etl = ETL()
  ETL.run(True, False, True)