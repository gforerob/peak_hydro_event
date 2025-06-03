from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import peak_hydro_event as phe


HPC_HOME = Path().home() / "hpchome"
INPUT_FOLDER = HPC_HOME / "usgs_qa_rev"
STATIONS_LIST = HPC_HOME / "2025_April25_QA_Rev" / "codes_for_review" / "usgs_stations_codes_validation_70plus_merged.csv"


def preprocess() -> tuple[NDArray[np.datetime64], NDArray[np.float64]]:
    
    # folder with the streamflow data in parquet files
    # csv with the station codes
    # usgs_id="08329900"  # to use another usgs code from the parquet folder
    df_codes = pd.read_csv(STATIONS_LIST, dtype={"USGS": str})
    usgs_id = df_codes["USGS"].iloc[0]
    file_path = INPUT_FOLDER / f"{usgs_id}.parquet"
    
    df = pd.read_parquet(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    
    # --- Keep nans and separate for calculation ---
    complete = df["Discharge_cfs"]
    valid = complete[complete.notna()]
    
    # --- Valids to numpy ---
    dates_array = valid.index.to_numpy()
    values_array = valid.values.copy()
    return dates_array, values_array
    


def process(dates_array: NDArray[np.datetime64], values_array: NDArray[np.float64]):
    
    
    df_baseflow, baseflow_array = phe.baseflow(values_array, dates_array)
    
    
    phe.plot_baseflow_and_streamflow(dates_array, values_array, baseflow_array, usgs_id=usgs_id)
    
    events, baseflow_array, events_mask, peaks_events_array, fechas_dt = phe.detect_all_events(
        streamflow=values_array,
        baseflow_array=baseflow_array,
        dates_array=dates_array
    )
    
    # count 
    print(f"Events detected: {len(events)}")
    
    phe.plot_all_events(
        events=events,
        streamflow=values_array,
        baseflow_array=baseflow_array,
        dates_array=dates_array,
        usgs_id=usgs_id
    )
    
    
    diff_array, valid_events = phe.filtered_events(events, values_array, baseflow_array)
    
    
    th_95 = np.percentile(diff_array[diff_array > 0], 95) #
    
    
    significant_peaks_mask = diff_array >= th_95
    significant_indexes = np.where(significant_peaks_mask)[0]
    
    # print results
    print(f"📈 95th percentile (baseflow ): {th_95:.2f}")
    print(f"🔍 Number of peaks after the filter: {len(significant_indexes)}")
    
    # Obtain significant events
    significant_events = [
        e for e in valid_events
        if diff_array[e["i_peak"]] >= th_95
    ]
    
    
    phe.plot_all_events(
        events=significant_events,
        streamflow=values_array,
        baseflow_array=baseflow_array,
        dates_array=dates_array,
        usgs_id=usgs_id
    )