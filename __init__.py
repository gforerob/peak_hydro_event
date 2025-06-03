import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
import os
from numpy.typing import NDArray


def baseflow_lyne_hollick(values_array: NDArray[np.float64], dates_array: NDArray[np.datetime64], alpha: float = 0.987):
    import numpy as np
    import pandas as pd

    n = len(values_array)
    dates = pd.to_datetime(dates_array)

    if n < 2:
        raise ValueError("at least two elements")

    # --- Forward filter ---
    qf_forward = np.zeros(n)
    for i in range(1, n):
        dq = values_array[i] - values_array[i - 1]
        qf_forward[i] = alpha * qf_forward[i - 1] + ((1 + alpha) / 2) * dq

    # --- Backward filter ---
    values_reverse = values_array[::-1]
    qf_backward = np.zeros(n)
    for i in range(1, n):
        dq = values_reverse[i] - values_reverse[i - 1]
        qf_backward[i] = alpha * qf_backward[i - 1] + ((1 + alpha) / 2) * dq
    qf_backward = qf_backward[::-1]

    # --- baseflow and quickflow ---
    quickflow = (qf_forward + qf_backward) / 2
    baseflow = values_array - quickflow
    baseflow = np.clip(baseflow, 0, values_array)

    df_baseflow = pd.DataFrame({
        "datetime": dates,
        "Q": values_array,
        "baseflow": baseflow,
        "quickflow": quickflow
    })

    return df_baseflow, baseflow


def baseflow(values_array: NDArray[np.float64], dates_array: NDArray[np.datetime64], **kw_args):
    # if method=="Lyne-Hollick"
    return baseflow_lyne_hollick(values_array, dates_array, **kw_args)

    
def plot_baseflow_and_streamflow(fechas_array: np.ndarray,
                                         valores_array: np.ndarray,
                                         baseflow_array: np.ndarray,
                                         usgs_id: str = ""):
   
    df = pd.DataFrame({
        "datetime": pd.to_datetime(fechas_array),
        "Q": valores_array,
        "baseflow": baseflow_array
    })

    df["year"] = df["datetime"].dt.year

    for year in sorted(df["year"].unique()):
        df_year = df[df["year"] == year]

        fig = go.Figure()

        # Streamflow
        fig.add_trace(go.Scatter(
            x=df_year["datetime"], y=df_year["Q"],
            mode="lines", name="Streamflow Q(t)",
        ))

        # Baseflow (calculated across full series)
        fig.add_trace(go.Scatter(
            x=df_year["datetime"], y=df_year["baseflow"],
            mode="lines", name="Baseflow (LH full)",
            line=dict(dash="dot")
        ))

        fig.update_layout(
            title=f"{usgs_id} – Baseflow via Lyne & Hollick – Year {year}",
            xaxis_title="Date",
            yaxis_title="Discharge (cfs)",
            template="plotly_white",
            height=500
        )

        fig.show()

def detect_all_events(
    streamflow: np.ndarray,
    baseflow_array: np.ndarray,
    dates_array: np.ndarray
) -> Tuple[
    List[dict],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex
]:
    
    dates_dt = pd.to_datetime(dates_array)
    n = len(streamflow)

    mask = streamflow > baseflow_array
    changes = np.diff(mask.astype(int))

    idx_start = np.where(changes == 1)[0]  # one position before the crossing
    idx_end = np.where(changes == -1)[0] + 1

    # Corrección de bordes
    if mask[0]:
        idx_start = np.insert(idx_start, 0, 0)
    if mask[-1]:
        idx_end = np.append(idx_end, n - 1)

    # Alinear índices
    if len(idx_start) == 0 or len(idx_end) == 0:
        return [], baseflow_array, mask, np.zeros(n), dates_dt
    if idx_end[0] < idx_start[0]:
        idx_end = idx_end[1:]
    if len(idx_start) > len(idx_end):
        idx_start = idx_start[:len(idx_end)]

    events = []
    events_mask = np.zeros(n, dtype=bool)
    peaks_events_array = np.zeros(n)

    for i_start, i_end in zip(idx_start, idx_end):
        if i_end <= i_start:
            continue  # inválid

        i_peak = i_start + np.argmax(streamflow[i_start:i_end + 1])

        # Reglas: pico no puede coincidir con inicio o fin
        if i_peak == i_start or i_peak == i_end:
            continue

        events.append({
            "i_start": i_start,
            "i_peak": i_peak,
            "i_end": i_end,
            "start": dates_dt[i_start],
            "peak": dates_dt[i_peak],
            "end": dates_dt[i_end]
        })
        events_mask[i_start:i_end + 1] = True
        peaks_events_array[i_peak] = streamflow[i_peak]

    return events, baseflow_array, events_mask, peaks_events_array, dates_dt

def plot_all_events(events, streamflow, baseflow_array, dates_array, usgs_id=""):
    
    df = pd.DataFrame({
        "datetime": pd.to_datetime(dates_array),
        "Q": streamflow,
        "baseflow": baseflow_array,
    })
    df["year"] = df["datetime"].dt.year

    events_df = pd.DataFrame(events)
    events_df["year"] = pd.to_datetime(events_df["start"]).dt.year

    for year in sorted(df["year"].unique()):
        df_year = df[df["year"] == year]
        events_year = events_df[events_df["year"] == year]

        fig = go.Figure()

        # Streamflow
        fig.add_trace(go.Scatter(
            x=df_year["datetime"], y=df_year["Q"],
            name="Streamflow", line=dict(color="blue")
        ))

        # Baseflow
        fig.add_trace(go.Scatter(
            x=df_year["datetime"], y=df_year["baseflow"],
            name="Baseflow", line=dict(color="green", dash="dot")
        ))

        # Event markers
        if not events_year.empty:
            fig.add_trace(go.Scatter(
                x=events_year["start"],
                y=streamflow[events_year["i_start"]],
                mode="markers", name="Start",
                marker=dict(symbol="diamond", size=10, color="orange")
            ))
            fig.add_trace(go.Scatter(
                x=events_year["peak"],
                y=streamflow[events_year["i_peak"]],
                mode="markers", name="Peak",
                marker=dict(symbol="star", size=12, color="black")
            ))
            fig.add_trace(go.Scatter(
                x=events_year["end"],
                y=streamflow[events_year["i_end"]],
                mode="markers", name="End",
                marker=dict(symbol="x", size=10, color="red")
            ))

        fig.update_layout(
            title=f"{usgs_id} – Detected Events (Q > Qb) – Year {year}",
            xaxis_title="Date",
            yaxis_title="Discharge (cfs)",
            template="plotly_white",
            height=500
        )

        fig.show()


def filtered_events(
    events: list,
    values_array: np.ndarray,
    baseflow_array: np.ndarray
) -> tuple[np.ndarray, list]:
    
    diff_array = np.zeros_like(values_array)
    valid_events = []

    for event in events:
        i_peak = event.get("i_peak")
        if i_peak is not None and 0 <= i_peak < len(values_array):
            bf = baseflow_array[i_peak]
            if bf > 0:
                diff_array[i_peak] = values_array[i_peak] - bf
                valid_events.append(event)

    return diff_array, valid_events



