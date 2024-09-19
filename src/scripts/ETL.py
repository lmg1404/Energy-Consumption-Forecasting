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

class ETL:
  def __init__(self):
    pass
  
  def _check_columns(dataframes: pd.DataFrame) -> List[pd.DataFrame]:
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
  
  def _extract_csvs(start: int, end: int) -> List[pd.DataFrame]:
    dfs = []
    for year in range(start, end+1):
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
    dfs = self._extract_csvs(2016, 2024)
    dfs = self._check_columns(dfs)
    master_df = pd.concat(dfs, axis=0, ignore_index=True)
    master_df.columns = ["_".join(re.sub(pattern, '', col).lower().split()) for col in master_df.columns]
    master_df["local_time_end_hour"] = pd.to_datetime(master_df["local_time_end_hour"])
    master_df =  master_df \
        .sort_values("local_time_end_hour",  ignore_index=True) \
        .dropna(axis=1, thresh=int(len(master_df) * (1-threshold))) \
        .dropna(axis=0, thresh=7) \
        .bfill(axis=0)
    master_df.to_csv("../../data/balance_sheet.csv", index=False)\\
    
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
    for dly_filename in glob.glob(r'../../data/ghcnd_hcn/*.dly', recursive=True): 
      path, name = os.path.split(dly_filename)
      csv_filename = os.path.join(path, f"{os.path.splitext(name)[0]}.csv")

      with open(dly_filename, newline='') as f_dly, open(csv_filename, 'w', newline='') as f_csv:
        csvout = csv.writer(f_csv)
        csvout.writerow(fieldnames)    # Write a header using the var names

        for line in f_dly:
          row = [line[start:end].strip() for var, start, end in fields]
          csvout.writerow(row)
    
  def run(self, balance_sheet:bool, dly_convert:bool) -> None:
    if balance_sheet:
      self.balance_sheets()
      
    # TODO
    if dly_convert:
      self.dly_convert()