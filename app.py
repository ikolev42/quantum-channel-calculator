# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import random

st.set_page_config(page_title="Channel Capacity Calculator", layout="wide")

# --- Pauli matrices ---
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


# --- Helper functions ---
def h(p):
    if p <= 0 or p >= 1: return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def entropy(rho):
    eigvals = np.linalg.eigvals(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return -np.sum(eigvals * np.log2(eigvals)).real


# --- Channels ---
classical_channels = {
    'Binary Symmetric': {
        'param': 'α', 'range': [0, 0.5], 'formula': 'C = 1 - H(α)',
        'capacity': lambda a: 1 - h(a),
        'achievable': lambda lam, a: h(lam + a - 2 * a * lam) - h(a)
    },
    'Binary Erasure': {
        'param': 'α', 'range': [0, 1], 'formula': 'C = 1 - α',
        'capacity': lambda a: 1 - a,
        'achievable': lambda lam, a: -sum(p * np.log2(p) for p in [lam * (1 - a), (1 - lam) * (1 - a), a] if p > 0)
    }
}

quantum_channels = {
    'Bit-flip': {
        'param': 'p', 'range': [0, 0.5], 'formula': 'R = H(ρ) - λH(ρ₁) - (1-λ)H(ρ₂)',
        'achievable': lambda rho1, rho2, l, p: (
                entropy((1 - p) * (l * rho1 + (1 - l) * rho2) + p * (X @ (l * rho1 + (1 - l) * rho2) @ X.conj().T))
                - l * entropy((1 - p) * rho1 + p * (X @ rho1 @ X.conj().T))
                - (1 - l) * entropy((1 - p) * rho2 + p * (X @ rho2 @ X.conj().T))
        )
    },
    'Phase-flip': {
        'param': 'p', 'range': [0, 0.5], 'formula': 'C = H(λ)',
        'capacity': lambda p: 1.0,
        'achievable': lambda lam, p: h(lam)
    }
}

# --- Sidebar ---
with st.sidebar:
    st.title("Channel Capacity Calculator")
    mode = st.radio("Mode", ["Classical", "Quantum"], horizontal=True)
    channels = classical_channels if mode == "Classical" else quantum_channels
    channel_name = st.selectbox("Channel", list(channels.keys()))
    channel = channels[channel_name]

    param = st.slider(f"Parameter {channel['param']}",
                      min_value=float(channel['range'][0]),
                      max_value=float(channel['range'][1]),
                      value=0.1, step=0.01)

    if mode == "Classical":
        lam = st.slider("λ (input probability)", 0.0, 1.0, 0.5, 0.01)
    else:
        st.markdown("**ρ = λ ρ₁ + (1-λ) ρ₂**")
        l = st.slider("λ", 0.0, 1.0, 0.5, 0.01)
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("ρ₁[0,0]", 0.0, 1.0, 0.5, 0.01)
            b = st.number_input("ρ₂[0,0]", 0.0, 1.0, 0.5, 0.01)
        with col2:
            c = st.number_input("Re(c)", -1.0, 1.0, 0.0, 0.01)
            d = st.number_input("Im(c)", -1.0, 1.0, 0.0, 0.01)
            e = st.number_input("Re(e)", -1.0, 1.0, 0.0, 0.01)
            f = st.number_input("Im(e)", -1.0, 1.0, 0.0, 0.01)

        rho1
