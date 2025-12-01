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
st.set_page_config(page_title="Power Quality Analyzer", layout="wide", page_icon="")

# -------------------------------------------------
# Base readings (Modified for 60W, 9W, and Laptop Charger)
# NOTE: The Laptop Charger data is mocked to represent a high ITHD load.
# -------------------------------------------------
base_data = {
    "Bulb 60W": {
        # Normal, resistive load
        "Voltage (V)": 220.4, "Current (A)": 0.2639, "Power (W)": 58.20,
        "Reactive (VAR)": -0.832, "Apparent (VA)": 58.07, "Power Factor": 1.000,
        "Frequency (Hz)": 50.35, "ITHD (%)": 2.101, "VTHD (%)": 1.914, "Energy (Wh)": 0.049
    },
    "Bulb 9W (LED)": {
        # Small LED, showing minor harmonic distortion (Warning)
        "Voltage (V)": 218.9, "Current (A)": 0.0331, "Power (W)": 7.232,
        "Reactive (VAR)": -0.282, "Apparent (VA)": 7.299, "Power Factor": 0.995,
        "Frequency (Hz)": 50.35, "ITHD (%)": 10.59, "VTHD (%)": 0.767, "Energy (Wh)": 0.050
    },
    "Laptop Charger": {
        # SMPS, non-linear load, showing severe harmonic distortion (Critical)
        # Placeholder values added for the demo:
        "Voltage (V)": 220.0, "Current (A)": 0.500, "Power (W)": 100.00,
        "Reactive (VAR)": 70.0, "Apparent (VA)": 122.06, "Power Factor": 0.819,
        "Frequency (Hz)": 50.00, "ITHD (%)": 65.0, "VTHD (%)": 3.50, "Energy (Wh)": 0.080
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
    # Default to the first load in the updated list
    st.session_state.load_state = list(base_data.keys())[0]

# -------------------------------------------------
# Sidebar - Load Selection
# -------------------------------------------------
with st.sidebar:
    st.header("Load Selection (Simulator)")
    
    # Check if data is coming from the STM32 (Normal Load) or Simulator (App Selection)
    st.caption("Select a load type below to **simulate** its power quality characteristics.")
    
    for load_name in base_data.keys():
        if st.button(load_name, use_container_width=True):
            st.session_state.load_state = load_name
            st.session_state.data_history = []
            st.session_state.event_log = []
            st.rerun()
            
    st.info(f"Current Simulated Load: **{st.session_state.load_state}**")
    st.caption("Data is auto-updating every 10 seconds.")


# -------------------------------------------------
# Cyclic fluctuation generator
# -------------------------------------------------
def get_cyclic_readings(load_name, cycle_index):
    """Generates readings based on the selected load with minor time-based fluctuations."""
    base = base_data[load_name]
    t = cycle_index / 10
    
    # Minor sine-wave based fluctuation (less than 1% change)
    fluct = 0.5 * math.sin(t) * 0.01

    readings = {}
    for key, val in base.items():
        if "Voltage" in key:
            # Voltage has a slight, independent fluctuation
            readings[key] = np.clip(base["Voltage (V)"] + 0.5 * math.sin(t * 0.5), 218, 221)
        elif isinstance(val, (int, float)):
            # Apply fluctuation to other numeric values
            readings[key] = val + val * fluct
        else:
            readings[key] = val
            
    readings["timestamp"] = datetime.now()
    return readings

# -------------------------------------------------
# RCA + event logging (Simplified to Load Type)
# -------------------------------------------------
def rca_event(load_name):
    """Simplified RCA based on user request: fixed status per load type."""
    if load_name == "Bulb 60W":
        return "NORMAL: Purely resistive load; low THD and Unity PF."
    elif load_name == "Bulb 9W (LED)":
        return "WARNING: Small Harmonic Distortion due to capacitor-based LED driver."
    elif load_name == "Laptop Charger":
        return "CRITICAL: Severe Harmonic Distortion (ITHD > 50%) characteristic of a non-PFC corrected SMPS."
    else:
        return "Load status unknown."

# -------------------------------------------------
# Data update loop (Runs every 10 seconds due to st.rerun/time.sleep)
# -------------------------------------------------
st.session_state.cycle_index += 1
data = get_cyclic_readings(st.session_state.load_state, st.session_state.cycle_index)
st.session_state.data_history.append(data)
if len(st.session_state.data_history) > 100:
    st.session_state.data_history = st.session_state.data_history[-100:]

# Create dataframe from history
df = pd.DataFrame(st.session_state.data_history)

# Event logging based on simplified RCA
current_load = st.session_state.load_state
ithd_value = float(df.tail(1)["ITHD (%)"])
event_text = rca_event(current_load)

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
# Display Current Readings & Status
# -------------------------------------------------
st.title("Power Quality Analyzer with RCA and Event Log")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Current Meter Readings (Simulated)")
    st.dataframe(df.tail(1), use_container_width=True, hide_index=True)

with col2:
    st.subheader("RCA Assessment")
    # Determine Status Color
    if "CRITICAL" in event_text:
        status_color = "red"
    elif "WARNING" in event_text:
        status_color = "orange"
    else:
        status_color = "green"
    
    st.markdown(f"""
        <div style='
            background-color: {status_color}; 
            padding: 15px; 
            border-radius: 10px; 
            color: white; 
            font-size: 1.2em;
            font-weight: bold;'>
            CURRENT STATUS: {event_text.split(':')[0]}
        </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------
# Graphs (10 parameters)
# -------------------------------------------------
st.header("Real-time Trends (Auto-update 10 s)")
params = [
    "Voltage (V)", "Current (A)", "Power Factor", "ITHD (%)", 
    "VTHD (%)", "Power (W)", "Reactive (VAR)", "Apparent (VA)", 
    "Frequency (Hz)", "Energy (Wh)"
]

cols = st.columns(5) # 5 graphs per row
for i, p in enumerate(params):
    with cols[i % 5]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[p], mode="lines", name=p))
        
        # Add IEEE 519 ITHD limit for visual context
        if p == "ITHD (%)":
            fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="IEEE 519 Limit (20%)", annotation_position="top left")
        
        fig.update_layout(
            title=p, 
            xaxis_title="Time", 
            yaxis_title=p, 
            height=250, 
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_dark",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# -------------------------------------------------
# Event log + CSV download
# -------------------------------------------------
st.header("Event Log — Root Cause Analysis Summary")
elog_df = pd.DataFrame(st.session_state.event_log)
st.dataframe(elog_df.iloc[::-1], use_container_width=True) # Reverse order to show newest first

csv = elog_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Event Log CSV", csv, "pq_rca_event_log.csv", "text/csv")

# -------------------------------------------------
# Auto-refresh every 10 s
# -------------------------------------------------
time.sleep(10)
st.rerun()
