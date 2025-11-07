import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="Quantum Channel Calculator", layout="wide")

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

# --- All Channels (4 Quantum + 2 Classical) ---
classical_channels = {
    'Binary Symmetric': {
        'param': 'α', 'range': [0.0, 0.5], 'formula': 'C = 1 - H(α)',
        'capacity': lambda a: 1 - h(a),
        'achievable': lambda lam, a: h(lam + a - 2*a*lam) - h(a),
        'desc': 'Each bit is flipped with probability α. Capacity is 1 - H(α).'
    },
    'Binary Erasure': {
        'param': 'α', 'range': [0.0, 1.0], 'formula': 'C = 1 - α',
        'capacity': lambda a: 1 - a,
        'achievable': lambda lam, a: -sum(p*np.log2(p) for p in [lam*(1-a), (1-lam)*(1-a), a] if p > 0),
        'desc': 'Each bit is erased with probability α. Capacity is 1 - α.'
    }
}

quantum_channels = {
    'Bit-flip': {
        'param': 'p', 'range': [0.0, 0.5], 'formula': 'R = H(ρ) - λH(ρ₁) - (1-λ)H(ρ₂)',
        'achievable': lambda rho1, rho2, l, p: (
            entropy((1-p)*(l*rho1 + (1-l)*rho2) + p*(X@(l*rho1 + (1-l)*rho2)@X.conj().T))
            - l*entropy((1-p)*rho1 + p*(X@rho1@X.conj().T))
            - (1-l)*entropy((1-p)*rho2 + p*(X@rho2@X.conj().T))
        ),
        'desc': 'Qubit is flipped with probability p (Pauli X). Uses full density matrix.'
    },
    'Phase-flip': {
        'param': 'p', 'range': [0.0, 0.5], 'formula': 'C = H(λ)',
        'capacity': lambda p: 1.0,
        'achievable': lambda lam, p: h(lam),
        'desc': 'Qubit phase is flipped with probability p (Pauli Z). Capacity is 1.'
    },
    'Depolarizing': {
        'param': 'p', 'range': [0.0, 1.0], 'formula': 'C = 1 - H(p/2) - p log₂(3)/2',
        'capacity': lambda p: 1 - h(p/2) - p * np.log2(3)/2,
        'achievable': lambda lam, p: h((1-p)*lam + p/2) - h(p/2),
        'desc': 'Qubit becomes fully mixed (I/2) with probability p.'
    },
    'Amplitude Damping': {
        'param': 'γ', 'range': [0.0, 1.0], 'formula': 'C = H(λ + (1-λ)γ) - (1-λ)H(γ)',
        'achievable': lambda lam, gamma: h(lam + (1-lam)*gamma) - (1-lam)*h(gamma),
        'desc': 'Qubit decays from |1⟩ to |0⟩ with probability γ (spontaneous emission).'
    }
}

# === APP OVERVIEW (ТОП ПОЛЕ) ===
st.markdown("""
## App Overview
This tool calculates **classical and quantum channel capacities** interactively.

### How to Use:
1. **Mode**: Switch between **Classical** and **Quantum**.
2. **Channel**: Select a channel (e.g., Bit-flip).
3. **Parameter**: Adjust noise (α, p, γ).
4. **Input State**: 
   - **Classical**: λ = P(X=0), 4-box matrix shows transition probabilities.
   - **Quantum**: λ, ρ₁, ρ₂ → 8 input fields (a, Re(c), Im(c), d, b, Re(e), Im(e), f).
5. **Buttons**: 
   - **Generate Random**: Fills valid random values.
   - **Calculate**: Updates rate and graphs.
6. **Results**: 
   - **Rate R**: Achievable rate for current input.
   - **Capacity C**: Maximum possible rate.
   - **Graphs**: Capacity vs parameter (blue), Rate vs parameter (red).

> All formulas are from quantum information theory (Holevo, Shannon, etc.).
""")

# --- Sidebar ---
st.sidebar.title("Controls")
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

# --- Buttons ---
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Generate Random"):
        if mode == "Classical":
            st.session_state.lam_class = round(random.uniform(0.2, 0.8), 4)
            st.session_state.a_class = round(random.uniform(0.2, 0.8), 4)
        else:
            st.session_state.lam_quant = round(random.uniform(0.2, 0.8), 4)
            a_val = round(random.uniform(0.2, 0.8), 4)
            b_val = round(1 - a_val, 4)
            max_coh = np.sqrt(a_val * b_val)
            st.session_state.update({
                'a1': a_val, 'a2': b_val,
                'c_re': round(random.uniform(-max_coh, max_coh), 4),
                'c_im': round(random.uniform(-max_coh, max_coh), 4),
                'd_re': round(random.uniform(-max_coh, max_coh), 4),
                'd_im': round(random.uniform(-max_coh, max_coh), 4),
                'e_re': round(random.uniform(-max_coh, max_coh), 4),
                'e_im': round(random.uniform(-max_coh, max_coh), 4),
                'f_re': round(random.uniform(-max_coh, max_coh), 4),
                'f_im': round(random.uniform(-max_coh, max_coh), 4)
            })
        st.rerun()

with col2:
    if st.button("Calculate"):
        st.rerun()

# --- Channel Description ---
st.markdown(f"### {channel_name}")
st.latex(channel['formula'])
st.write(channel['desc'])

# --- Calculation ---
try:
    if mode == "Classical":
        lam = st.session_state.get('lam_class', 0.5)
        rate = channel['achievable'](lam, param)
        cap = channel.get('capacity', lambda x: 0)(param)
    else:
        lam = st.session_state.get('lam_quant', 0.5)
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
