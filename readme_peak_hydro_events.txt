peak_hydro_events

 baseflow separation using the Lyne-Hollick method, hydrologic event detection and filtering of significant peak flows with advanced visualizations.

---

##  Overview

The workflow consists of two main functions:

- `preprocess()`: Loads and cleans high-resolution discharge time series for a specified USGS station from a `.parquet` file.
- `process()`: Executes the event detection pipeline, generates plots, and identifies significant hydrologic events based on percentile filtering of quickflow.

---

##  Directory Structure

The script assumes the following directory structure rooted in the current user's `$HOME/hpchome`:

$HOME/hpchome/
├── usgs_qa_rev/
│ └── <station_id>.parquet
└── 2025_April25_QA_Rev/
└── codes_for_review/
└── usgs_stations_codes_validation_70plus_merged.csv

Copy
Edit

---

##  Dependencies

Install required packages:


pip install numpy pandas pyarrow plotly
Inputs required
Station CSV: Contains USGS station codes in a column named USGS.

Parquet Files: Each file contains a high-frequency streamflow time series under the column Discharge_cfs, with a datetime column for timestamps.

Usage
Step 1 – Load Data

dates_array, values_array = preprocess()

Selects the first USGS station from the CSV.

Loads the corresponding .parquet file.

Cleans, sorts, and deduplicates the time index.

Extracts non-null values of the discharge series as NumPy arrays.

Step 2 – Analyze and Visualize

process(dates_array, values_array)
This function:

Computes baseflow using Lyne-Hollick filtering.

Detects flow events where streamflow exceeds baseflow.

Visualizes baseflow and detected events by year.

Filters peaks based on the 95th percentile of the quickflow.

Highlights and plots statistically significant events.

Output
Interactive visualizations of baseflow separation and hydrologic events using Plotly.

Console output summarizing:

Total detected events.

Threshold used (e.g., 95th percentile).

Number of significant peaks.

Modular Functions
The script relies on the peak_hydro_event.py module, which should implement:

baseflow()

detect_all_events()

filtered_events()

plot_baseflow_and_streamflow()

plot_all_events()

These functions enable modular, testable, and reusable hydrologic analysis workflows.

Notes

Significant events are identified based on quickflow exceeding a high percentile.
Lyne-Hollick alpha value used is 0.987,


Author
Developed by [Gonzalo Alberto Forero Buitrago], 2025.
AHWA Laboratory
IIHR University of Iowa


