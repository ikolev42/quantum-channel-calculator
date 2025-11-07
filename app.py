import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="Channel Capacity Calculator", layout="wide")

# --- Pauli X ---
X = np.array([[0, 1], [1, 0]], dtype=complex)

# --- Entropy ---
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
    lambda_input = st.sidebar.slider("λ", 0.0, 1.0, 0.5, 0.01, key="lambda_classical")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.number_input("P(0|0)", 0.0, 1.0, 0.5, 0.01, key="a_classical")
        st.write(f"**1-a = {1-a:.4f}**")
    with col2:
        st.write(f"**1-a = {1-a:.4f}**")
        st.write(f"**a = {a:.4f}**")
else:
    st.sidebar.markdown("**ρ = λ ρ₁ + (1-λ) ρ₂**")
    lambda_input = st.sidebar.slider("λ", 0.0, 1.0, 0.5, 0.01, key="lambda_quantum")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.number_input("ρ₁[0,0]", 0.0, 1.0, 0.5, 0.01, key="a1")
        c_re = st.number_input("Re(c)", -1.0, 1.0, 0.0, 0.01, key="c_re")
        c_im = st.number_input("Im(c)", -1.0, 1.0, 0.0, 0.01, key="c_im")
        st.write(f"**1-a = {1-a:.4f}**")
    with col2:
        b = st.number_input("ρ₂[0,0]", 0.0, 1.0, 0.5, 0.01, key="a2")
        e_re = st.number_input("Re(e)", -1.0, 1.0, 0.0, 0.01, key="e_re")
        e_im = st.number_input("Im(e)", -1.0, 1.0, 0.0, 0.01, key="e_im")
        st.write(f"**1-b = {1-b:.4f}**")
    
    rho1 = np.array([[a, c_re + 1j*c_im], [c_re - 1j*c_im, 1-a]], dtype=complex)
    rho2 = np.array([[b, e_re + 1j*e_im], [e_re - 1j*e_im, 1-b]], dtype=complex)

# --- Buttons ---
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Generate Random"):
        if mode == "Classical":
            st.experimental_set_query_params(random_classical=round(random.uniform(0.2, 0.8), 4))
        else:
            l = round(random.uniform(0.2, 0.8), 4)
            a_val = round(random.uniform(0.2, 0.8), 4)
            b_val = round(1 - a_val, 4)
            max_coh = np.sqrt(a_val * b_val)
            st.experimental_set_query_params(
                random_quantum=l,
                a1=a_val, a2=b_val,
                c_re=round(random.uniform(-max_coh, max_coh), 4),
                c_im=round(random.uniform(-max_coh, max_coh), 4),
                e_re=round(random.uniform(-max_coh, max_coh), 4),
                e_im=round(random.uniform(-max_coh, max_coh), 4)
            )
        st.rerun()

with col2:
    if st.button("Calculate"):
        st.rerun()

# --- Load from URL ---
query_params = st.experimental_get_query_params()
if mode == "Classical" and "random_classical" in query_params:
    lambda_input = float(query_params["random_classical"][0])
    st.experimental_set_query_params()  # clear
elif mode == "Quantum" and "random_quantum" in query_params:
    lambda_input = float(query_params["random_quantum"][0])
    st.experimental_set_query_params()  # clear

# --- Main Calculation ---
st.markdown(f"### {channel_name}")
st.latex(channel['formula'])

try:
    if mode == "Classical":
        lam = lambda_input
        rate = channel['achievable'](lam, param)
        cap = channel.get('capacity', lambda x: 0)(param)
    else:
        lam = lambda_input
        if channel_name == "Bit-flip":
            rate = channel['achievable'](rho1, rho2, lam, param)
            cap = max([channel['achievable'](rho1, rho2, ll, param) for ll in np.linspace(0,1,20)])
        else:
            rate = channel['achievable'](lam, param)
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
    rates = [channel['achievable'](lam, p) for p in params]
elif channel_name == "Bit-flip":
    rates = [channel['achievable'](rho1, rho2, lam, p) for p in params]
else:
    rates = [channel['achievable'](lam, p) for p in params]
ax2.plot(params, rates, 'r-', linewidth=2)
ax2.set_xlabel(channel['param'])
ax2.set_ylabel("Rate R")
ax2.set_title(f"Achievable Rate (λ = {lam:.2f})")
ax2.grid(alpha=0.3)

st.pyplot(fig)

st.markdown("---")
st.markdown("**Free version** – Pro: export, dark mode, save.")
