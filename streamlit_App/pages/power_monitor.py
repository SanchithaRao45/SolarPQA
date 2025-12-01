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

# --- Constants for Waveform Generation ---
SAMPLE_POINTS = 500  # Resolution for one cycle
FREQUENCY = 50.0     # Base frequency (Hz)
PHASE_SHIFT = np.pi / 4 # 45 degrees phase shift for non-unity PF

# -------------------------------------------------
# Base readings (Modified for 60W, 9W, and Laptop Charger)
# NOTE: The ITHD values directly control the distortion visible on the waveform plot.
# -------------------------------------------------
base_data = {
    "Bulb 60W": {
        # Normal, resistive load (Very Low ITHD)
        "Voltage (V)": 220.4, "Current (A)": 0.2639, "Power (W)": 58.20,
        "Reactive (VAR)": -0.832, "Apparent (VA)": 58.07, "Power Factor": 1.000,
        "Frequency (Hz)": 50.35, "ITHD (%)": 2.10, "VTHD (%)": 1.914, "Energy (Wh)": 0.049
    },
    "Bulb 9W (LED)": {
        # Small LED, showing minor harmonic distortion (Warning)
        "Voltage (V)": 218.9, "Current (A)": 0.0331, "Power (W)": 7.232,
        "Reactive (VAR)": -0.282, "Apparent (VA)": 7.299, "Power Factor": 0.995,
        "Frequency (Hz)": 50.35, "ITHD (%)": 10.59, "VTHD (%)": 0.767, "Energy (Wh)": 0.050
    },
    "Laptop Charger": {
        # SMPS, non-linear load, showing severe harmonic distortion (Critical)
        "Voltage (V)": 220.0, "Current (A)": 0.500, "Power (W)": 100.00,
        "Reactive (VAR)": 70.0, "Apparent (VA)": 122.06, "Power Factor": 0.819,
        "Frequency (Hz)": 50.00, "ITHD (%)": 65.0, "VTHD (%)": 3.50, "Energy (Wh)": 0.080
    }
}

# -------------------------------------------------
# Waveform Generation Functions
# -------------------------------------------------

def generate_sine_wave(rms_value, phase_rad=0):
    """Generates a pure sine wave (V or A) for one cycle."""
    t = np.linspace(0, 1/FREQUENCY, SAMPLE_POINTS)
    peak = rms_value * np.sqrt(2)
    waveform = peak * np.sin(2 * np.pi * FREQUENCY * t + phase_rad)
    return t * 1000, waveform # Convert time to milliseconds

def generate_power_curve(V_rms, I_rms, PF):
    """Generates the instantaneous Active Power (P) curve for one cycle."""
    t = np.linspace(0, 1/FREQUENCY, SAMPLE_POINTS)
    # Convert PF to phase angle
    phi = np.arccos(PF)
    V_peak = V_rms * np.sqrt(2)
    I_peak = I_rms * np.sqrt(2)
    
    # P(t) = V(t) * I(t)
    P_t = V_peak * np.sin(2 * np.pi * FREQUENCY * t) * I_peak * np.sin(2 * np.pi * FREQUENCY * t - phi)
    return t * 1000, P_t

def generate_harmonic_wave(I_rms, ITHD_percent, phase_rad=0):
    """Generates a distorted current waveform based on ITHD value."""
    t = np.linspace(0, 1/FREQUENCY, SAMPLE_POINTS)
    I_peak = I_rms * np.sqrt(2)
    
    # Fundamental component
    I_fund = I_peak * np.sin(2 * np.pi * FREQUENCY * t + phase_rad)
    
    # Scale harmonics based on ITHD. Higher ITHD means higher harmonic contribution.
    ITHD_factor = ITHD_percent / 100.0
    
    # Add 3rd and 5th harmonics (common non-linear distortion)
    harmonic_3 = (ITHD_factor / 2) * I_peak * np.sin(2 * np.pi * (3 * FREQUENCY) * t + phase_rad + 0.5)
    harmonic_5 = (ITHD_factor / 4) * I_peak * np.sin(2 * np.pi * (5 * FREQUENCY) * t + phase_rad - 1.0)
    
    I_distorted = I_fund + harmonic_3 + harmonic_5
    return t * 1000, I_distorted

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
    st.session_state.load_state = list(base_data.keys())[0]

# -------------------------------------------------
# Sidebar - Load Selection
# -------------------------------------------------
with st.sidebar:
    st.header("Load Selection (Simulator)")
    st.caption("Select a load to simulate its PQ characteristics.")
    
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
    fluct = 0.5 * math.sin(t) * 0.01 # Fluctuation factor
    
    readings = {}
    for key, val in base.items():
        if key in ["Voltage (V)", "Frequency (Hz)"]:
            readings[key] = val + 0.2 * math.sin(t)
        elif isinstance(val, (int, float)):
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
# Data update loop
# -------------------------------------------------
st.session_state.cycle_index += 1
data = get_cyclic_readings(st.session_state.load_state, st.session_state.cycle_index)
st.session_state.data_history.append(data)
if len(st.session_state.data_history) > 100:
    st.session_state.data_history = st.session_state.data_history[-100:]

df = pd.DataFrame(st.session_state.data_history)
last_readings = df.tail(1).iloc[0].to_dict() # Get the latest readings

# Event logging based on simplified RCA
current_load = st.session_state.load_state
ithd_value = float(last_readings.get("ITHD (%)", 0))
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
st.title("Power Quality Analyzer with RCA and Waveform Visualization")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Current Meter Readings (Simulated)")
    st.dataframe(df.tail(1), use_container_width=True, hide_index=True)

with col2:
    st.subheader("RCA Assessment")
    if "CRITICAL" in event_text:
        status_color = "#E53935" # Red
    elif "WARNING" in event_text:
        status_color = "#FFB300" # Orange
    else:
        status_color = "#43A047" # Green
    
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
# 1. Real-time Waveform Visualization (Single Cycle)
# -------------------------------------------------
st.header("1. Real-time Waveform Visualization (Instantaneous)")
st.caption("Plots are fixed to one 50Hz cycle to show instantaneous distortion and phase angle.")

V_rms = last_readings.get("Voltage (V)", 0)
I_rms = last_readings.get("Current (A)", 0)
P_factor = last_readings.get("Power Factor", 1.0)
ITHD_p = last_readings.get("ITHD (%)", 0)

# Calculate Apparent and Reactive Power RMS (using S = V*I and Q = sqrt(S^2 - P^2))
S_rms = V_rms * I_rms
Q_rms = last_readings.get("Reactive (VAR)", 0)

# Generate waveform data
t_ms, V_wave = generate_sine_wave(V_rms)
t_ms, P_wave = generate_power_curve(V_rms, I_rms, P_factor)
t_ms, I_wave_distorted = generate_harmonic_wave(I_rms, ITHD_p, PHASE_SHIFT)
t_ms, S_wave = generate_sine_wave(S_rms, PHASE_SHIFT) # S is represented by a sine wave scaled to Apparent Power
t_ms, Q_wave = generate_sine_wave(Q_rms, PHASE_SHIFT - np.pi / 2) # Q is 90 degrees out of phase

# --- Plotting Grid for Waveforms ---
colA, colB, colC = st.columns(3)

# 1. Voltage and Current Waveform (Distorted)
with colA:
    fig_vi = go.Figure()
    fig_vi.add_trace(go.Scatter(x=t_ms, y=V_wave, mode='lines', name='Voltage (V)', line=dict(color='blue')))
    fig_vi.add_trace(go.Scatter(x=t_ms, y=I_wave_distorted, mode='lines', name='Current (A) (Distorted)', line=dict(color='red', width=3)))
    fig_vi.update_layout(title='Voltage & Distorted Current', xaxis_title='Time (ms)', yaxis_title='Magnitude', height=350, template="plotly_dark")
    st.plotly_chart(fig_vi, use_container_width=True, config={'displayModeBar': False})

# 2. Instantaneous Power Curve (P, Q, S)
with colB:
    fig_pqs = go.Figure()
    fig_pqs.add_trace(go.Scatter(x=t_ms, y=P_wave, mode='lines', name='Active Power (W)', line=dict(color='yellow')))
    fig_pqs.add_trace(go.Scatter(x=t_ms, y=S_wave, mode='lines', name='Apparent Power (VA) (Sine)', line=dict(color='orange', dash='dash')))
    fig_pqs.add_trace(go.Scatter(x=t_ms, y=Q_wave, mode='lines', name='Reactive Power (VAR) (Sine)', line=dict(color='purple', dash='dot')))
    fig_pqs.update_layout(title='Instantaneous Power Curves', xaxis_title='Time (ms)', yaxis_title='Power (W/VA/VAR)', height=350, template="plotly_dark")
    st.plotly_chart(fig_pqs, use_container_width=True, config={'displayModeBar': False})

# 3. Harmonic Profile (VTHD/ITHD)
with colC:
    fig_harm = go.Figure(data=[
        go.Bar(name='VTHD', x=['VTHD (%)'], y=[last_readings.get("VTHD (%)", 0)], marker_color='lightblue'),
        go.Bar(name='ITHD', x=['ITHD (%)'], y=[last_readings.get("ITHD (%)", 0)], marker_color='coral')
    ])
    fig_harm.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="VTHD Limit (5%)", annotation_position="top right")
    fig_harm.update_layout(title='Harmonic Distortion Levels', yaxis_title='Percentage (%)', height=350, template="plotly_dark")
    st.plotly_chart(fig_harm, use_container_width=True, config={'displayModeBar': False})


# -------------------------------------------------
# 2. Real-time Trends (Time Series)
# -------------------------------------------------
st.header("2. Real-time Trends (Time-Series)")
st.caption("Plots show parameter values over the 10-second update cycles.")

params_ts = ["Frequency (Hz)", "Power Factor", "Energy (Wh)"]

cols_ts = st.columns(3)
for i, p in enumerate(params_ts):
    with cols_ts[i % 3]:
        fig_ts = go.Figure()
        
        # Energy and Time must be a linear graph
        if p == "Energy (Wh)":
            mode_type = "lines"
        else:
            mode_type = "lines+markers"
            
        fig_ts.add_trace(go.Scatter(x=df["timestamp"], y=df[p], mode=mode_type, name=p))
        
        # Add a PF target line
        if p == "Power Factor":
             fig_ts.add_hline(y=0.95, line_dash="dash", line_color="lightgreen", annotation_text="Target PF (0.95)", annotation_position="bottom right")

        fig_ts.update_layout(
            title=p, 
            xaxis_title="Time", 
            yaxis_title=p, 
            height=300, 
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_dark",
            showlegend=False
        )
        st.plotly_chart(fig_ts, use_container_width=True, config={'displayModeBar': False})


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
