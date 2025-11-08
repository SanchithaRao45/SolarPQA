import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# Page setup
st.set_page_config(page_title="Single-Phase Power Quality Analyzer", layout="wide", page_icon="⚡")

# Initialize session state
if 'data_history' not in st.session_state:
    st.session_state.data_history = []
if 'load_state' not in st.session_state:
    st.session_state.load_state = "Load 1"

# ===================================
# Simulated Real-time Data Function
# ===================================
def get_current_readings():
    base_voltage = 230.0
    base_current = 10.0
    base_pf = 0.9
    base_freq = 50.0

    variation = np.random.uniform(-0.03, 0.03)
    load_factor = {"Load 1": 1.0, "Load 2": 1.5, "Load 3": 0.7}[st.session_state.load_state]

    voltage = base_voltage * (1 + variation) * load_factor
    current = base_current * (1 + variation) * load_factor
    pf = np.clip(base_pf + variation, 0.7, 1.0)
    freq = base_freq + variation * 0.2

    power_W = voltage * current * pf
    apparent_power_VA = voltage * current
    reactive_power_VAR = np.sqrt(max(apparent_power_VA**2 - power_W**2, 0))
    vthd = np.clip(2.5 + variation * 2, 0, 8)
    ithd = np.clip(4.0 + variation * 3, 0, 12)
    wh = power_W * 0.002  # Simulated Wh increment

    return {
        "timestamp": datetime.now(),
        "Voltage (V)": voltage,
        "Current (A)": current,
        "Power (W)": power_W,
        "Reactive (VAR)": reactive_power_VAR,
        "Apparent (VA)": apparent_power_VA,
        "Power Factor": pf,
        "Frequency (Hz)": freq,
        "ITHD (%)": ithd,
        "VTHD (%)": vthd,
        "Energy (Wh)": wh
    }

# ===================================
# Sidebar Controls
# ===================================
with st.sidebar:
    st.header("⚙️ Controls")
    st.subheader("Load Selection")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Load 1", use_container_width=True):
            st.session_state.load_state = "Load 1"
            st.rerun()
    with col2:
        if st.button("Load 2", use_container_width=True):
            st.session_state.load_state = "Load 2"
            st.rerun()
    with col3:
        if st.button("Load 3", use_container_width=True):
            st.session_state.load_state = "Load 3"
            st.rerun()

    st.info(f"Current Load: **{st.session_state.load_state}**")
    st.divider()

    history_length = st.slider("History Length (points)", 10, 200, 100)
    st.caption("Auto refresh every 10 seconds")

# ===================================
# Data Acquisition
# ===================================
data = get_current_readings()
st.session_state.data_history.append(data)
if len(st.session_state.data_history) > history_length:
    st.session_state.data_history = st.session_state.data_history[-history_length:]

df = pd.DataFrame(st.session_state.data_history)

# ===================================
# Display Current Table
# ===================================
st.title("⚡ Single-Phase Power Quality Analyzer")
st.markdown("### Real-time Monitoring of All Electrical Parameters")

st.subheader("📋 Current Readings")
st.table(df.tail(1).style.format("{:.2f}"))

# ===================================
# Graphs for All 10 Parameters
# ===================================
st.header("📈 Real-time Graphs (Updates Every 10s)")

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

# ===================================
# Auto Refresh (Every 10 seconds)
# ===================================
time.sleep(10)
st.rerun()
