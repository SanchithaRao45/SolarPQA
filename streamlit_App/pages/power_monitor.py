import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import math

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title=" Power Quality Analyzer", layout="wide", page_icon="")

# -------------------------------------------------
# Base readings (from your measurements)
# -------------------------------------------------
base_data = {
    "Bulb 60W": {
        "Voltage (V)": 220.4, "Current (A)": 0.2639, "Power (W)": 58.20,
        "Reactive (VAR)": -0.832, "Apparent (VA)": 58.07, "Power Factor": 1.000,
        "Frequency (Hz)": 50.35, "ITHD (%)": 2.101, "VTHD (%)": 1.914, "Energy (Wh)": 0.049
    },
    "Bulb 5W (LED)": {
        "Voltage (V)": 219.1, "Current (A)": 0.0258, "Power (W)": 5.280,
        "Reactive (VAR)": -1.999, "Apparent (VA)": 5.625, "Power Factor": 0.934,
        "Frequency (Hz)": 50.35, "ITHD (%)": 5.048, "VTHD (%)": 0.642, "Energy (Wh)": 0.049
    },
    "Bulb 9W (LED)": {
        "Voltage (V)": 218.9, "Current (A)": 0.0331, "Power (W)": 7.232,
        "Reactive (VAR)": -0.282, "Apparent (VA)": 7.299, "Power Factor": 0.995,
        "Frequency (Hz)": 50.35, "ITHD (%)": 10.59, "VTHD (%)": 0.767, "Energy (Wh)": 0.050
    }
}

# -------------------------------------------------
# Session state init
# -------------------------------------------------
if "data_history" not in st.session_state:
    st.session_state.data_history = []
if "event_log" not in st.session_state:
    st.session_state.event_log = []
if "cycle_index" not in st.session_state:
    st.session_state.cycle_index = 0
if "load_state" not in st.session_state:
    st.session_state.load_state = "Bulb 60W"

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header(" Load Selection")
    for load_name in base_data.keys():
        if st.button(load_name, use_container_width=True):
            st.session_state.load_state = load_name
            st.session_state.data_history = []
            st.session_state.event_log = []
            st.rerun()
    st.info(f"Current Load: **{st.session_state.load_state}**")
    st.caption("Auto-updating every 10 seconds")

# -------------------------------------------------
# Cyclic fluctuation generator
# -------------------------------------------------
def get_cyclic_readings(load_name, cycle_index):
    base = base_data[load_name]
    t = cycle_index / 10
    fluct = 0.4 * math.sin(t)

    readings = {}
    for key, val in base.items():
        if "Voltage" in key:
            readings[key] = np.clip(219 + 0.5 * math.sin(t), 218, 220)
        elif isinstance(val, (int, float)):
            readings[key] = val + val * fluct * 0.01
        else:
            readings[key] = val
    readings["timestamp"] = datetime.now()
    return readings

# -------------------------------------------------
# RCA + event logging
# -------------------------------------------------
def rca_event(load, ITHD):
    if load == "Bulb 60W":
        return "Normal Operation — Resistive load; PF≈1; low THD."
    elif load == "Bulb 5W (LED)":
        return "Normal Operation — Small LED; acceptable PF & ITHD."
    elif load == "Bulb 9W (LED)":
        if ITHD > 10:
            return "Acceptable — ITHD increasing; monitor LED driver."
        else:
            return "Normal Operation."
    else:
        return "Load status unknown."

# -------------------------------------------------
# Data update
# -------------------------------------------------
st.session_state.cycle_index += 1
data = get_cyclic_readings(st.session_state.load_state, st.session_state.cycle_index)
st.session_state.data_history.append(data)
if len(st.session_state.data_history) > 100:
    st.session_state.data_history = st.session_state.data_history[-100:]

# Create dataframe
df = pd.DataFrame(st.session_state.data_history)

# Event logging condition
current_load = st.session_state.load_state
ithd_value = float(df.tail(1)["ITHD (%)"])
event_text = rca_event(current_load, ithd_value)
event_entry = {
    "Time": datetime.now().strftime("%H:%M:%S"),
    "Load": current_load,
    "ITHD (%)": round(ithd_value, 3),
    "Event": event_text
}
st.session_state.event_log.append(event_entry)
if len(st.session_state.event_log) > 50:
    st.session_state.event_log = st.session_state.event_log[-50:]

# -------------------------------------------------
# Display current readings
# -------------------------------------------------
st.title("  Power Quality Analyzer with RCA and Event Log")

st.subheader(" Current Readings")
st.table(df.tail(1).style.format("{:.3f}"))

# -------------------------------------------------
# Graphs (10 parameters)
# -------------------------------------------------
st.header(" Real-time Graphs (auto-update 10 s)")
params = [
    "Voltage (V)", "Current (A)", "Power (W)", "Reactive (VAR)",
    "Apparent (VA)", "Power Factor", "Frequency (Hz)",
    "ITHD (%)", "VTHD (%)", "Energy (Wh)"
]

for p in params:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[p], mode="lines+markers", name=p))
    fig.update_layout(title=p, xaxis_title="Time", yaxis_title=p, height=300, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Event log + CSV download
# -------------------------------------------------
st.header(" Event Log — Root Cause Analysis")
elog_df = pd.DataFrame(st.session_state.event_log)
st.dataframe(elog_df, use_container_width=True)

csv = elog_df.to_csv(index=False).encode("utf-8")
st.download_button(" Download Event Log CSV", csv, "event_log.csv", "text/csv")

# -------------------------------------------------
# Auto-refresh every 10 s
# -------------------------------------------------
time.sleep(10)
st.rerun()
