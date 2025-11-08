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
# Base readings from your measured data
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
# Thresholds for event alerts
# ----------------------------------------
THRESHOLDS = {
    "Voltage Low": 218.0,
    "Voltage High": 220.0,
    "Frequency Low": 49.5,
    "Frequency High": 50.5,
    "ITHD High": 10.0,
    "VTHD High": 5.0
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
if "event_log" not in st.session_state:
    st.session_state.event_log = []

# ----------------------------------------
# Sidebar controls
# ----------------------------------------
with st.sidebar:
    st.header("⚙️ Load Selection")
    for load_name in base_data.keys():
        if st.button(load_name, use_container_width=True):
            st.session_state.load_state = load_name
            st.session_state.data_history = []
            st.session_state.event_log = []
            st.rerun()

    st.info(f"Current Load: **{st.session_state.load_state}**")
    st.caption("Auto-updating every 10 seconds")
    st.divider()

    # CSV download option
    if st.session_state.data_history:
        df_export = pd.DataFrame(st.session_state.data_history)
        csv_data = df_export.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"{st.session_state.load_state.replace(' ', '_')}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ----------------------------------------
# Generate smooth cyclic fluctuation
# ----------------------------------------
def get_cyclic_readings(load_name, cycle_index):
    base = base_data[load_name]
    t = cycle_index / 10
    fluctuation = 0.4 * math.sin(t)  # smooth ±0.4%

    readings = {}
    for key, val in base.items():
        if "Voltage" in key:
            readings[key] = np.clip(219 + 0.5 * math.sin(t), 218, 220)
        elif isinstance(val, (int, float)):
            readings[key] = val + val * fluctuation * 0.01
        else:
            readings[key] = val
    readings["timestamp"] = datetime.now()
    return readings

# ----------------------------------------
# Event check logic
# ----------------------------------------
def check_events(reading):
    events = []
    if reading["Voltage (V)"] < THRESHOLDS["Voltage Low"]:
        events.append(f"Low Voltage: {reading['Voltage (V)']:.2f} V")
    elif reading["Voltage (V)"] > THRESHOLDS["Voltage High"]:
        events.append(f"High Voltage: {reading['Voltage (V)']:.2f} V")

    if reading["Frequency (Hz)"] < THRESHOLDS["Frequency Low"] or reading["Frequency (Hz)"] > THRESHOLDS["Frequency High"]:
        events.append(f"Frequency Out of Range: {reading['Frequency (Hz)']:.2f} Hz")

    if reading["ITHD (%)"] > THRESHOLDS["ITHD High"]:
        events.append(f"High ITHD: {reading['ITHD (%)']:.2f}%")

    if reading["VTHD (%)"] > THRESHOLDS["VTHD High"]:
        events.append(f"High VTHD: {reading['VTHD (%)']:.2f}%")

    return events

def log_event(event_message):
    st.session_state.event_log.append({
        "timestamp": datetime.now(),
        "event": event_message
    })
    if len(st.session_state.event_log) > 100:
        st.session_state.event_log = st.session_state.event_log[-100:]

# ----------------------------------------
# Update readings
# ----------------------------------------
st.session_state.cycle_index += 1
data = get_cyclic_readings(st.session_state.load_state, st.session_state.cycle_index)
st.session_state.data_history.append(data)
if len(st.session_state.data_history) > 100:
    st.session_state.data_history = st.session_state.data_history[-100:]

# Check and log events
for event in check_events(data):
    log_event(event)

df = pd.DataFrame(st.session_state.data_history)

# ----------------------------------------
# Display current readings
# ----------------------------------------
st.title("💡 Manual Power Quality Analyzer")
st.markdown("### Real Readings with Cyclic Variations (±0.4%)")

st.subheader("📋 Current Readings")
st.table(df.tail(1).style.format("{:.3f}"))

# ----------------------------------------
# Graph section for all 10 parameters
# ----------------------------------------
st.header("📈 Real-time Graphs for 10 Parameters (Auto-update every 10 s)")

graph_params = [
    "Voltage (V)", "Current (A)", "Power (W)", "Reactive (VAR)",
    "Apparent (VA)", "Power Factor", "Frequency (Hz)",
    "ITHD (%)", "VTHD (%)", "Energy (Wh)"
]

for param in graph_params:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[param], mode="lines+markers", name=param))
    fig.update_layout(
        title=param,
        xaxis_title="Time",
        yaxis_title=param,
        height=300,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# Event log section
# ----------------------------------------
st.header("📋 Event Log")
if st.session_state.event_log:
    event_df = pd.DataFrame(st.session_state.event_log)
    event_df["timestamp"] = event_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(event_df.iloc[::-1].head(20), use_container_width=True, hide_index=True)
else:
    st.info("No events logged yet.")

# ----------------------------------------
# Auto-refresh
# ----------------------------------------
time.sleep(10)
st.rerun()
