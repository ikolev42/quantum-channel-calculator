import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="Channel Capacity Calculator", layout="wide")

# --- Helper functions ---
def h(p):
    if p <= 0 or p >= 1: return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def entropy(rho):
    eigvals = np.linalg.eigvals(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return -np.sum(eigvals * np.log2(eigvals)).real if len(eigvals) > 0 else 0.0

# --- Channels ---
classical_channels = {
    'Binary Symmetric': {
        'param': 'α', 'range': [0.0, 0.5], 'formula': 'C = 1 - H(α)',
        'capacity': lambda a: 1 - h(a),
        'achievable': lambda lam, a: h(lam + a - 2*a*lam) - h(a)
    },
    'Binary Erasure': {
        'param': 'α', 'range': [0.0, 1.0], 'formula': 'C = 1 - α',
        'capacity': lambda a: 1 - a,
        'achievable': lambda lam, a: -sum(p*np.log2(p) for p in [lam*(1-a), (1-lam)*(1-a), a] if p > 0)
    }
}

quantum_channels = {
    'Bit-flip': {
        'param': 'p', 'range': [0.0, 0.5], 'formula': 'R = H(ρ) - λH(ρ₁) - (1-λ)H(ρ₂)',
        'achievable': lambda rho1, rho2, l, p: (
            entropy((1-p)*(l*rho1 + (1-l)*rho2) + p*(X@(l*rho1 + (1-l)*rho2)@X.conj().T))
            - l*entropy((1-p)*rho1 + p*(X@rho1@X.conj().T))
            - (1-l)*entropy((1-p)*rho2 + p*(X@rho2@X.conj().T))
        )
    },
    'Phase-flip': {
        'param': 'p', 'range': [0.0, 0.5], 'formula': 'C = H(λ)',
        'capacity': lambda p: 1.0,
        'achievable': lambda lam, p: h(lam)
    },
    'Depolarizing': {
        'param': 'p', 'range': [0.0, 1.0], 'formula': 'C = 1 - H(p/2) - p log₂(3)/2',
        'capacity': lambda p: 1 - h(p/2) - p * np.log2(3)/2,
        'achievable': lambda lam, p: h((1-p)*lam + p/2) - h(p/2)
    },
    'Amplitude Damping': {
        'param': 'γ', 'range': [0.0, 1.0], 'formula': 'C = H(λ + (1-λ)γ) - (1-λ)H(γ)',
        'achievable': lambda lam, gamma: h(lam + (1 - lam) * gamma) - (1 - lam) * h(gamma)
    }
}

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

# --- Input State ---
if mode == "Classical":
    st.sidebar.markdown("**P(X=0) = λ**")
    lambda_input = st.sidebar.slider("λ", 0.0, 1.0, 0.5, 0.01, key="lam_class")
    
    st.sidebar.markdown("**Transition Matrix:**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.number_input("P(0|0)", 0.0, 1.0, 0.5, 0.01, key="a_class")
        st.write(f"**1-a = {1-a:.4f}**")
    with col2:
        st.write(f"**1-a = {1-a:.4f}**")
        st.write(f"**a = {a:.4f}**")
else:
    st.sidebar.markdown("**ρ = λ ρ₁ + (1-λ) ρ₂**")
    lambda_input = st.sidebar.slider("λ", 0.0, 1.0, 0.5, 0.01, key="lam_quant")
    
    st.sidebar.markdown("**Density Matrices:**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.number_input("ρ₁[0,0]", 0.0, 1.0, 0.5, 0.01, key="a1")
        c_re = st.number_input("Re(c)", -1.0, 1.0, 0.0, 0.01, key="c_re")
        c_im = st.number_input("Im(c)", -1.0, 1.0, 0.0, 0.01, key="c_im")
        d_re = st.number_input("Re(d)", -1.0, 1.0, 0.0, 0.01, key="d_re")
        d_im = st.number_input("Im(d)", -1.0, 1.0, 0.0, 0.01, key="d_im")
        st.write(f"**1-a = {1-a:.4f}**")
    with col2:
        b = st.number_input("ρ₂[0,0]", 0.0, 1.0, 0.5, 0.01, key="a2")
        e_re = st.number_input("Re(e)", -1.0, 1.0, 0.0, 0.01, key="e_re")
        e_im = st.number_input("Im(e)", -1.0, 1.0, 0.0, 0.01, key="e_im")
        f_re = st.number_input("Re(f)", -1.0, 1.0, 0.0, 0.01, key="f_re")
        f_im = st.number_input("Im(f)", -1.0, 1.0, 0.0, 0.01, key="f_im")
        st.write(f"**1-b = {1-b:.4f}**")
    
    rho1 = np.array([[a, c_re + 1j*c_im], [d_re + 1j*d_im, 1-a]], dtype=complex)
    rho2 = np.array([[b, e_re + 1j*e_im], [f_re + 1j*f_im, 1-b]], dtype=complex)
