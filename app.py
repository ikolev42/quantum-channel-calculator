import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="Channel Capacity Calculator", layout="wide")

# --- Helper functions ---
def h(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# --- Classical Channels ---
classical_channels = {
    'Binary Symmetric': {
        'param': 'α', 'range': [0.0, 0.5], 'formula': 'C = 1 - H(α)',
        'capacity': lambda a: 1 - h(a),
        'achievable': lambda lam, a: h(lam + a - 2*a*lam) - h(a),
        'desc': 'Bit flip with probability α.'
    },
    'Binary Erasure': {
        'param': 'α', 'range': [0.0, 1.0], 'formula': 'C = �1 - α',
        'capacity': lambda a: 1 - a,
        'achievable': lambda lam, a: -sum(p*np.log2(p) for p in [lam*(1-a), (1-lam)*(1-a), a] if p > 0),
        'desc': 'Bit erased with probability α.'
    }
}

# --- Quantum Channels ---
quantum_channels = {
    'Bit-flip': {
        'param': 'p', 'range': [0.0, 0.5], 'formula': 'R = H(ρ) - λH(ρ₁) - (1-λ)H(ρ₂)',
        'achievable': lambda rho1, rho2, l, p: (
            entropy((1-p)*(l*rho1 + (1-l)*rho2) + p*(X@(l*rho1 + (1-l)*rho2)@X.conj().T))
            - l*entropy((1-p)*rho1 + p*(X@rho1@X.conj().T))
            - (1-l)*entropy((1-p)*rho2 + p*(X@rho2@X.conj().T))
        ),
        'desc': 'Qubit flipped with probability p.'
    },
    'Phase-flip': {
        'param': 'p', 'range': [0.0, 0.5], 'formula': 'C = H(λ)',
        'capacity': lambda p: 1.0,
        'achievable': lambda lam, p: h(lam),
        'desc': 'Phase flipped with probability p.'
    }
}

# Pauli X
X = np.array([[0, 1], [1, 0]], dtype=complex)

def entropy(rho):
    eigvals = np.linalg.eigvals(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return -np.sum(eigvals * np.log2(eigvals)).real if len(eigvals) > 0 else 0.0

# --- Sidebar ---
st.sidebar.title("Channel Capacity Calculator")
mode = st.sidebar.radio("Mode", ["Classical", "Quantum"], horizontal=True)
channels = classical_channels if mode == "Classical" else quantum_channels
channel_name = st.sidebar.selectbox("Channel", list(channels.keys()))
channel = channels[channel_name]

param = st.sidebar.slider(
    f"{channel['param']}",
    min_value=float(channel['range'][0]),
    max_value=float(channel['range'][1]),
    value=0.1,
    step=0.01
)

if mode == "Classical":
    lambda_input = st.sidebar.slider("λ (P(X=0))", 0.0, 1.0, 0.5, 0.01)
else:
    st.sidebar.markdown("**ρ = λ ρ₁ + (1-λ) ρ₂**")
    lambda_input = st.sidebar.slider("λ", 0.0, 1.0, 0.5, 0.01)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.number_input("ρ₁[0,0]", 0.0, 1.0, 0.5, 0.01)
        b = st.number_input("ρ₂[0,0]", 0.0, 1.0, 0.5, 0.01)
    with col2:
        c_re = st.number_input("Re(c)", -1.0, 1.0, 0.0, 0.01)
        c_im = st.number_input("Im(c)", -1.0, 1.0, 0.0, 0.01)
        e_re = st.number_input("Re(e)", -1.0, 1.0, 0.0, 0.01)
        e_im = st.number_input("Im(e)", -1.0, 1.0, 0.0, 0.01)
    
    rho1 = np.array([[a, c_re + 1j*c_im], [c_re - 1j*c_im, 1-a]], dtype=complex)
    rho2 = np.array([[b, e_re + 1j*e_im], [e_re - 1j*e_im, 1-b]], dtype=complex)

# --- Generate Random Example ---
if st.sidebar.button("Generate Random Example"):
    if mode == "Classical":
        lambda_input = random.uniform(0.2, 0.8)
    else:
        lambda_input = random.uniform(0.2, 0.8)
        a = random.uniform(0.2, 0.8)
        b = 1 - a
        max_coh = np.sqrt(a * b)
        c_re = random.uniform(-max_coh, max_coh)
        c_im = random.uniform(-max_coh, max_coh)
        e_re = random.uniform(-max_coh, max_coh)
        e_im = random.uniform(-max_coh, max_coh)
        # Force rerun
        st.experimental_rerun()

# --- Calculate Button ---
if st.sidebar.button("Calculate"):
    pass  # Trigger recalc

# --- Main ---
st.markdown(f"### {channel_name}")
st.latex(channel['formula'])

try:
    if mode == "Classical":
        rate = channel['achievable'](lambda_input, param)
        cap = channel.get('capacity', lambda x: 0)(param)
    else:
        if channel_name == "Bit-flip":
            rate = channel['achievable'](rho1, rho2, lambda_input, param)
            cap = max([channel['achievable'](rho1, rho2, ll, param) for ll in np.linspace(0,1,20)])
        else:
            rate = channel['achievable'](lambda_input, param)
            cap = channel.get('capacity', lambda x: 0)(param)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Achievable Rate R", f"{rate:.4f}")
    with col2:
        st.metric("Channel Capacity C", f"{cap:.4f}")
except Exception as e:
    st.error(f"Error: {e}")

# --- Graphs ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
params = np.linspace(channel['range'][0], channel['range'][1], 200)

# Capacity
ax1.clear()
if mode == "Classical":
    caps = [channel['capacity'](p) for p in params]
elif channel_name == "Bit-flip":
    caps = [max([channel['achievable'](rho1, rho2, ll, p) for ll in np.linspace(0,1,10)]) for p in params]
else:
    caps = [channel.get('capacity', lambda x: 0)(p) for p in params]
ax1.plot(params, caps, 'b-', linewidth=2)
ax1.set_xlabel(channel['param'])
ax1.set_ylabel("Capacity C")
ax1.set_title("Channel Capacity")
ax1.grid(alpha=0.3)

# Rate
ax2.clear()
if mode == "Classical":
    rates = [channel['achievable'](lambda_input, p) for p in params]
elif channel_name == "Bit-flip":
    rates = [channel['achievable'](rho1, rho2, lambda_input, p) for p in params]
else:
    rates = [channel['achievable'](lambda_input, p) for p in params]
ax2.plot(params, rates, 'r-', linewidth=2)
ax2.set_xlabel(channel['param'])
ax2.set_ylabel("Rate R")
ax2.set_title(f"Achievable Rate (λ = {lambda_input:.2f})")
ax2.grid(alpha=0.3)

st.pyplot(fig)

st.markdown("---")
st.markdown("**Free version** – Pro version with export & save coming soon!")
