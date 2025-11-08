import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import math

# ----------------------------------------
# Page setup
# ----------------------------------------
st.set_page_config(page_title="Manual Power Quality Analyzer", layout="wide", page_icon="💡")

# ----------------------------------------
# Base readings from your sheet
# ----------------------------------------
base_data = {
    "Bulb 60W": {
        "Voltage (V)": 220.4, "Current (A)": 0.2639, "Power (W)": 58.20,
        "Reactive (VAR)": -0.832, "Apparent (VA)": 58.07, "Power Factor": 1.000,
        "Frequency (Hz)": 50.35, "ITHD (%)": 2.101, "VTHD (%)": 1.914, "Energy (Wh)": 0.049
    },
    "Bulb 15W (LED)": {
        "Voltage (V)": 219.1, "Current (A)": 0.0258, "Power (W)": 5.280,
        "Reactive (VAR)": -1.999, "Apparent (VA)": 5.625, "Power Factor": 0.934,
        "Frequency (Hz)": 50.35, "ITHD (%)": 5.048, "VTHD (%)": 0.642, "Energy (Wh)": 0.049
    },
    "Phone Charger (45W)": {
        "Voltage (V)": 219.9, "Current (A)": 0.1985, "Power (W)": 27.71,
        "Reactive (VAR)": -3.852, "Apparent (VA)": 44.34, "Power Factor": 0.631,
        "Frequency (Hz)": 50.35, "ITHD (%)": 17.4, "VTHD (%)": 1.905, "Energy (Wh)": 0.050
    },
    "Bulb 9W (LED)": {
        "Voltage (V)": 218.9, "Current (A)": 0.0331, "Power (W)": 7.232,
        "Reactive (VAR)": -0.282, "Apparent (VA)": 7.299, "Power Factor": 0.995,
        "Frequency (Hz)": 50.35, "ITHD (%)": 10.59, "VTHD (%)": 0.767, "Energy (Wh)": 0.050
    }
}

# ----------------------------------------
# Initialize state
# ----------------------------------------
if "data_history" not in st.session_state:
    st.session_state.data_history = []
if "load_state" not in st.session_state:
    st.session_state.load_state = "Bulb 60W"
if "cycle_index" not in st.session_state:
    st.session_state.cycle_index = 0

# ----------------------------------------
# Sidebar controls
# ----------------------------------------
with st.sidebar:
    st.header(" Load Selection")
    for load_name in base_data.keys():
        if st.button(load_name, use_container_width=True):
            st.session_state.load_state = load_name
            st.session_state.data_history = []  # reset data on load change
            st.rerun()

    st.info(f"Current Load: **{st.session_state.load_state}**")
    st.caption("Auto-updating every 10 seconds")
    st.divider()

# ----------------------------------------
# Generate smooth cyclic fluctuation
# ----------------------------------------
def get_cyclic_readings(load_name, cycle_index):
    base = base_data[load_name]
    t = cycle_index / 10  # time step
    fluctuation = 0.4 * math.sin(t)  # smooth cyclic +/-0.4 variation

    readings = {}
    for key, val in base.items():
        if "Voltage" in key:
            readings[key] = np.clip(219 + 0.5 * math.sin(t), 218, 220)
        elif isinstance(val, (int, float)):
            readings[key] = val + val * fluctuation * 0.01  # ~±0.4%
        else:
            readings[key] = val

    readings["timestamp"] = datetime.now()
    return readings

# ----------------------------------------
# Collect readings
# ----------------------------------------
st.session_state.cycle_index += 1
data = get_cyclic_readings(st.session_state.load_state, st.session_state.cycle_index)
st.session_state.data_history.append(data)
if len(st.session_state.data_history) > 100:
    st.session_state.data_history = st.session_state.data_history[-100:]

df = pd.DataFrame(st.session_state.data_history)

# ----------------------------------------
# Display table
# ----------------------------------------
st.title("  Power Quality Analyzer")


st.subheader(" Current Readings")
st.table(df.tail(1).style.format("{:.3f}"))

# ----------------------------------------
# Graph section for all parameters
# ----------------------------------------
st.header(" Graphs  (Auto-update every 10 s)")

graph_params = [
    "Voltage (V)", "Current (A)", "Power (W)", "Reactive (VAR)",
    "Apparent (VA)", "Power Factor", "Frequency (Hz)",
    "ITHD (%)", "VTHD (%)", "Energy (Wh)"
]

for param in graph_params:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[param],
                             mode='lines+markers', name=param))
    fig.update_layout(
        title=param,
        xaxis_title="Time",
        yaxis_title=param,
        height=300,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# Auto-refresh every 10 s
# ----------------------------------------
time.sleep(10)
st.rerun()
