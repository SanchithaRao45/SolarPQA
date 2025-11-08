import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config
st.set_page_config(page_title="Single-Phase Power Quality Analyzer", layout="wide", page_icon="⚡")

# Initialize session state
if 'data_history' not in st.session_state:
    st.session_state.data_history = []
if 'event_log' not in st.session_state:
    st.session_state.event_log = []
if 'load_state' not in st.session_state:
    st.session_state.load_state = "Load 1"
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# ================================
# Simulated Real-time Data
# ================================
def get_current_readings():
    base_voltage = 230.0
    base_current = 10.0
    base_pf = 0.9
    base_freq = 50.0

    # Random variation
    variation = np.random.uniform(-0.03, 0.03)

    # Load adjustments
    load_factor = {"Load 1": 1.0, "Load 2": 1.5, "Load 3": 0.7}[st.session_state.load_state]

    voltage = base_voltage * (1 + variation) * load_factor
    current = base_current * (1 + variation) * load_factor
    pf = np.clip(base_pf + variation, 0.7, 1.0)
    frequency = base_freq + variation * 0.2

    # Derived parameters
    power_W = voltage * current * pf
    apparent_power_VA = voltage * current
    reactive_power_VAR = np.sqrt(max(apparent_power_VA**2 - power_W**2, 0))
    vthd = np.clip(2.5 + variation * 2, 0, 8)
    ithd = np.clip(4.0 + variation * 3, 0, 12)
    wh = power_W * 0.002  # Simulated energy consumption (Wh increment)

    return {
        "timestamp": datetime.now(),
        "voltage": voltage,
        "current": current,
        "power_W": power_W,
        "VAR": reactive_power_VAR,
        "VA": apparent_power_VA,
        "pf": pf,
        "freq": frequency,
        "ithd": ithd,
        "vthd": vthd,
        "Wh": wh
    }

# ================================
# Sidebar Controls
# ================================
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
    st.subheader("Display Settings")
    auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    if st.button("🔄 Manual Refresh", use_container_width=True):
        st.rerun()

    refresh_interval = st.slider("Refresh Interval (sec)", 1, 10, 2)
    history_length = st.slider("History Length (points)", 10, 100, 50)

    st.divider()
    if st.session_state.data_history:
        df = pd.DataFrame(st.session_state.data_history)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"single_phase_pq_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ================================
# Data Acquisition
# ================================
data = get_current_readings()
st.session_state.data_history.append(data)
if len(st.session_state.data_history) > history_length:
    st.session_state.data_history = st.session_state.data_history[-history_length:]

# ================================
# Display Section
# ================================
st.title("⚡ Single-Phase Power Quality Analyzer")
st.markdown("### Real-time monitoring and logging of single-phase electrical parameters")

# --- Display Current Readings Table ---
st.header("📊 Current Readings")
df_display = pd.DataFrame([{
    "Voltage (V)": data["voltage"],
    "Current (A)": data["current"],
    "Power (W)": data["power_W"],
    "Reactive (VAR)": data["VAR"],
    "Apparent (VA)": data["VA"],
    "Power Factor": data["pf"],
    "Frequency (Hz)": data["freq"],
    "ITHD (%)": data["ithd"],
    "VTHD (%)": data["vthd"],
    "Energy (Wh)": data["Wh"]
}])

st.table(df_display.style.format("{:.2f}"))

# ================================
# Graphs Section
# ================================
if len(st.session_state.data_history) > 1:
    df = pd.DataFrame(st.session_state.data_history)

    st.header("📈 Real-Time Graphs")

    # Voltage Graph
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=df['timestamp'], y=df['voltage'], mode='lines', name='Voltage (V)'))
    fig_v.update_layout(title="Voltage vs Time", xaxis_title="Time", yaxis_title="Voltage (V)", height=300)
    st.plotly_chart(fig_v, use_container_width=True)

    # Current Graph
    fig_i = go.Figure()
    fig_i.add_trace(go.Scatter(x=df['timestamp'], y=df['current'], mode='lines', name='Current (A)', line_color='orange'))
    fig_i.update_layout(title="Current vs Time", xaxis_title="Time", yaxis_title="Current (A)", height=300)
    st.plotly_chart(fig_i, use_container_width=True)

    # Power Graph
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=df['timestamp'], y=df['power_W'], mode='lines', name='Active Power (W)', line_color='green'))
    fig_p.update_layout(title="Power (W)", xaxis_title="Time", yaxis_title="Power (W)", height=300)
    st.plotly_chart(fig_p, use_container_width=True)

    # THD Graph
    fig_thd = go.Figure()
    fig_thd.add_trace(go.Scatter(x=df['timestamp'], y=df['vthd'], mode='lines', name='VTHD (%)', line_color='purple'))
    fig_thd.add_trace(go.Scatter(x=df['timestamp'], y=df['ithd'], mode='lines', name='ITHD (%)', line_color='red'))
    fig_thd.update_layout(title="Total Harmonic Distortion", xaxis_title="Time", yaxis_title="THD (%)", height=300)
    st.plotly_chart(fig_thd, use_container_width=True)

    # Power Factor Graph
    fig_pf = go.Figure()
    fig_pf.add_trace(go.Scatter(x=df['timestamp'], y=df['pf'], mode='lines', name='Power Factor', line_color='blue'))
    fig_pf.update_layout(title="Power Factor", xaxis_title="Time", yaxis_title="PF", height=300)
    st.plotly_chart(fig_pf, use_container_width=True)

# ================================
# Auto Refresh
# ================================
if st.session_state.auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
