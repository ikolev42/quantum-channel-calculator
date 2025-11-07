import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize_scalar
import random

# --- Helper functions ---
def h(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def entropy(probs):
    return -sum(pr * np.log2(pr) for pr in probs if pr > 0)

# --- Classical channels ---
classical_channels = {
    'Binary Symmetric': {
        'param_name': 'α',
        'range': [0, 0.5],
        'formula': 'C = 1 - H(α)',
        'capacity': lambda a: 1 - h(a),
        'achievable': lambda lam, a: h(lam + a - 2 * a * lam) - h(a),
        'description': 'Binary Symmetric Channel (BSC) is a classical channel where each bit has probability α to be flipped (0 -> 1 or 1 -> 0). It models noise in classical communications. Capacity C is the maximum information rate, achievable rate R depends on input distribution λ. α is the noise parameter, between 0 and 0.5. Binary entropy H is used for calculation.'
    },
    'Binary Erasure': {
        'param_name': 'α',
        'range': [0, 1],
        'formula': 'C = 1 - α',
        'capacity': lambda a: 1 - a,
        'achievable': lambda lam, a: entropy([lam * (1 - a), (1 - lam) * (1 - a), a]) - h(a),
        'description': 'Binary Erasure Channel (BEC) is a classical channel where each bit has probability α to be erased (replaced with "?" or error symbol). It models information loss. Capacity C is directly 1 - α, achievable rate R depends on λ. α is the erasure parameter, between 0 and 1. Entropy is used to calculate probabilities for 0, 1, and erasure.'
    }
}

# --- Quantum channels ---
quantum_channels = {
    'Bit-flip': {
        'param_name': 'p',
        'range': [0, 0.5],
        'formula': 'C = H(λ + p - 2pλ) - H(p)',
        'capacity': lambda p: 1 - h(p),
        'achievable': lambda lam, p: h(lam + p - 2 * p * lam) - h(p),
        'description': 'Bit-flip channel is a quantum channel where the qubit has probability p to be flipped (Pauli X operator applied). It models classical errors in quantum systems. Capacity C is 1 - H(p), achievable rate R depends on the input state λ. p is the noise parameter, between 0 and 0.5. The density matrix ρ must be valid (a + b = 1, c^2 + e^2 ≤ a*b). Uses Holevo-Schumacher-Westmoreland theorem for calculation.'
    },
    'Phase-flip': {
        'param_name': 'p',
        'range': [0, 0.5],
        'formula': 'C = H(λ)',
        'capacity': lambda p: 1.0,
        'achievable': lambda lam, p: h(lam),
        'description': 'Phase-flip channel is a quantum channel where the qubit has probability p to undergo phase flip (Pauli Z operator). It models phase errors. Capacity C is 1.0 (independent of p), achievable rate R is H(λ). p is the noise parameter, between 0 and 0.5. The input state ρ must be valid. This channel preserves diagonal elements but reduces coherence.'
    },
    'Depolarizing': {
        'param_name': 'p',
        'range': [0, 1],
        'formula': 'C = H((1-p)λ + p/2) - H(p/2)',
        'capacity': lambda p: 1 - h(p / 2),
        'achievable': lambda lam, p: h((1 - p) * lam + p / 2) - h(p / 2),
        'description': 'Depolarizing channel is a quantum channel where the state has probability p to be replaced with the fully mixed state (I/2). It models complete depolarization. Capacity C is 1 - H(p/2), achievable rate R depends on λ. p is the noise parameter, between 0 and 1. The density matrix ρ must be valid. This channel is symmetric and commonly used to model general errors.'
    },
    'Amplitude Damping': {
        'param_name': 'γ',
        'range': [0, 1],
        'formula': 'C = H(λ + (1-λ)γ) - (1-λ)H(γ)',
        'achievable': lambda lam, gamma: h(lam + (1 - lam) * gamma) - (1 - lam) * h(gamma),
        'description': 'Amplitude Damping channel is a quantum channel modeling energy loss (relaxation) in quantum systems, e.g., spontaneous emission. It transfers the state from |1> to |0> with probability γ. Capacity C is complex, achievable rate R depends on λ and γ. γ is the damping parameter, between 0 and 1. The input state ρ must be valid. This channel is asymmetric and important for real quantum devices.'
    }
}

def get_capacity(channel, param):
    if 'capacity' in channel:
        return channel['capacity'](param)
    else:
        def neg_ach(lam):
            return -channel['achievable'](lam, param)
        res = minimize_scalar(neg_ach, bounds=(0, 1), method='bounded')
        return -res.fun

# --- Main application ---
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Channel Capacity Calculator")
root.geometry("1350x800")

# === LEFT PANEL ===
left_frame = ctk.CTkFrame(root, width=340)
left_frame.pack(side='left', padx=15, pady=15, fill='y')
left_frame.pack_propagate(False)

# A: Mode slider
slider_frame = ctk.CTkFrame(left_frame)
slider_frame.pack(pady=(10, 20))
ctk.CTkLabel(slider_frame, text="Classical", font=("Arial", 12)).pack(side='left', padx=(0, 10))
mode_slider = ctk.CTkSlider(slider_frame, from_=0, to=1, number_of_steps=1, width=100, height=20, button_length=35)
mode_slider.pack(side='left')
mode_slider.set(0)
ctk.CTkLabel(slider_frame, text="Quantum", font=("Arial", 12)).pack(side='left', padx=(10, 0))

# B: Channel
ctk.CTkLabel(left_frame, text="B: Channel", font=("Arial", 14, "bold")).pack(pady=(5, 5))
combo = ctk.CTkComboBox(left_frame, values=[], width=280)
combo.pack()

param_label = ctk.CTkLabel(left_frame, text="Parameter:")
param_label.pack(pady=(10, 0))
param_entry = ctk.CTkEntry(left_frame, placeholder_text="0.0", width=280)
param_entry.pack()

# C: Input State
ctk.CTkLabel(left_frame, text="C: Input State", font=("Arial", 14, "bold")).pack(pady=(25, 5))
c_frame = ctk.CTkFrame(left_frame)
c_frame.pack(pady=5, fill='x')

# --- Classical mode: 4 fields ---
classical_matrix_frame = ctk.CTkFrame(c_frame)

top_row_classical = ctk.CTkFrame(classical_matrix_frame)
top_row_classical.pack()
ctk.CTkLabel(top_row_classical, text="a:").pack(side='left', padx=5)
a_classical_entry = ctk.CTkEntry(top_row_classical, width=80)
a_classical_entry.pack(side='left', padx=5)
ctk.CTkLabel(top_row_classical, text="1-a:").pack(side='left', padx=5)
one_minus_a_classical_entry = ctk.CTkEntry(top_row_classical, width=80, state="disabled")
one_minus_a_classical_entry.pack(side='left', padx=5)

bottom_row_classical = ctk.CTkFrame(classical_matrix_frame)
bottom_row_classical.pack(pady=5)
ctk.CTkLabel(bottom_row_classical, text="1-a:").pack(side='left', padx=5)
one_minus_a_lower_entry = ctk.CTkEntry(bottom_row_classical, width=80, state="disabled")
one_minus_a_lower_entry.pack(side='left', padx=5)
ctk.CTkLabel(bottom_row_classical, text="a:").pack(side='left', padx=5)
a_lower_entry = ctk.CTkEntry(bottom_row_classical, width=80, state="disabled")
a_lower_entry.pack(side='left', padx=5)

# --- Quantum matrix ---
quantum_matrix_frame = ctk.CTkFrame(c_frame)

# lambda
ctk.CTkLabel(quantum_matrix_frame, text="lambda", font=("Helvetica", 16)).grid(row=1, column=0, rowspan=3, padx=5)
entry_lambda = ctk.CTkEntry(quantum_matrix_frame, width=75, height=30, justify="center")
entry_lambda.grid(row=2, column=1)
ctk.CTkLabel(quantum_matrix_frame, text="1-lambda", font=("Helvetica", 16)).grid(row=6, column=0, rowspan=3, padx=5)
entry_one_minus_lambda = ctk.CTkEntry(quantum_matrix_frame, width=75, height=30, justify="center", state="disabled")
entry_one_minus_lambda.grid(row=7, column=1)

# rho0
ctk.CTkLabel(quantum_matrix_frame, text="rho0", font=("Arial", 16)).grid(row=0, column=2, columnspan=4)
for i in range(1, 4):
    ctk.CTkLabel(quantum_matrix_frame, text="|", font=("Helvetica", 16)).grid(row=i, column=2)
    ctk.CTkLabel(quantum_matrix_frame, text="|", font=("Helvetica", 16)).grid(row=i, column=5)

entry_a = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="a", width=75, height=30, justify="center")
entry_a.grid(row=1, column=3)
entry_c = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="c", width=75, height=30, justify="center")
entry_c.grid(row=1, column=4)
entry_d = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="d", width=75, height=30, justify="center")
entry_d.grid(row=3, column=3)
entry_one_minus_a = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="1-a", width=75, height=30, justify="center", state="disabled")
entry_one_minus_a.grid(row=3, column=4)

ctk.CTkLabel(quantum_matrix_frame, text="+", font=("Arial", 24)).grid(row=4, column=2, columnspan=4)

# rho1
ctk.CTkLabel(quantum_matrix_frame, text="rho1", font=("Arial", 16)).grid(row=5, column=2, columnspan=4)
for i in range(6, 9):
    ctk.CTkLabel(quantum_matrix_frame, text="|", font=("Helvetica", 16)).grid(row=i, column=2)
    ctk.CTkLabel(quantum_matrix_frame, text="|", font=("Helvetica", 16)).grid(row=i, column=5)

entry_b = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="b", width=75, height=30, justify="center")
entry_b.grid(row=6, column=3)
entry_e = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="e", width=75, height=30, justify="center")
entry_e.grid(row=6, column=4)
entry_f = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="f", width=75, height=30, justify="center")
entry_f.grid(row=8, column=3)
entry_one_minus_b = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="1-b", width=75, height=30, justify="center", state="disabled")
entry_one_minus_b.grid(row=8, column=4)

# rho
ctk.CTkLabel(quantum_matrix_frame, text="rho", font=("Arial", 16)).grid(row=10, column=2, columnspan=4)
for i in range(11, 14):
    ctk.CTkLabel(quantum_matrix_frame, text="|", font=("Helvetica", 16)).grid(row=i, column=2)
    ctk.CTkLabel(quantum_matrix_frame, text="|", font=("Helvetica", 16)).grid(row=i, column=5)

entry_rho_11 = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="...", width=75, height=30, justify="center", state="disabled")
entry_rho_11.grid(row=11, column=3)
entry_rho_12 = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="...", width=75, height=30, justify="center", state="disabled")
entry_rho_12.grid(row=11, column=4)
entry_rho_21 = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="...", width=75, height=30, justify="center", state="disabled")
entry_rho_21.grid(row=13, column=3)
entry_rho_22 = ctk.CTkEntry(quantum_matrix_frame, placeholder_text="...", width=75, height=30, justify="center", state="disabled")
entry_rho_22.grid(row=13, column=4)

ctk.CTkLabel(quantum_matrix_frame, text="||", font=("Helvetica", 16)).grid(row=9, column=2, columnspan=4)

# --- Generate Example Button ---
example_button = ctk.CTkButton(left_frame, text="Generate Example", width=280)
example_button.pack(pady=20)

# === RIGHT PANEL ===
right_frame = ctk.CTkFrame(root)
right_frame.pack(side='right', padx=15, pady=15, fill='both', expand=True)

ctk.CTkLabel(right_frame, text="D: Channel Description", font=("Arial", 14, "bold")).pack(anchor='w')
d_text = ctk.CTkTextbox(right_frame, height=200)
d_text.pack(pady=5, fill='x')

e_frame = ctk.CTkFrame(right_frame)
e_frame.pack(pady=10, fill='x')
formula_label = ctk.CTkLabel(e_frame, text="Formula: —", width=320, anchor='w')
formula_label.pack(side='left', padx=10)
capacity_label = ctk.CTkLabel(e_frame, text="Rate: —", width=220)
capacity_label.pack(side='left', padx=20)
calculate_button = ctk.CTkButton(e_frame, text="Calculate", width=130)
calculate_button.pack(side='right', padx=10)

# Plots
plot_frame1 = ctk.CTkFrame(right_frame, height=300)
plot_frame1.pack(pady=10, fill='both', expand=True)
plot_frame1.pack_propagate(False)
figure_capacity = plt.Figure(figsize=(6, 3.5), dpi=100)
ax_capacity = figure_capacity.add_subplot(111)
canvas_capacity = FigureCanvasTkAgg(figure_capacity, plot_frame1)
canvas_capacity.get_tk_widget().pack(fill='both', expand=True)

plot_frame2 = ctk.CTkFrame(right_frame, height=300)
plot_frame2.pack(pady=10, fill='both', expand=True)
plot_frame2.pack_propagate(False)
figure_ach = plt.Figure(figsize=(6, 3.5), dpi=100)
ax_ach = figure_ach.add_subplot(111)
canvas_ach = FigureCanvasTkAgg(figure_ach, plot_frame2)
canvas_ach.get_tk_widget().pack(fill='both', expand=True)

# --- Global ---
current_rho1 = current_rho2 = current_l = None

# --- Functions ---
def safe_float(e, default=0.0):
    try: return float(e.get().strip()) if e.get().strip() else default
    except: return default

def safe_complex(e, default=0j):
    try: return complex(e.get().strip()) if e.get().strip() else default
    except: return default

def update_classical_a(*args):
    try:
        a = safe_float(a_classical_entry, 0.5)
        val = f"{1-a:.6f}"
        for entry in [one_minus_a_classical_entry, one_minus_a_lower_entry]:
            entry.configure(state="normal"); entry.delete(0, "end"); entry.insert(0, val); entry.configure(state="disabled")
        a_lower_entry.configure(state="normal"); a_lower_entry.delete(0, "end"); entry.insert(0, f"{a:.6f}"); a_lower_entry.configure(state="disabled")
    except: pass

a_classical_entry.bind("<KeyRelease>", update_classical_a)

def update_rho():
    global current_rho1, current_rho2, current_l
    try:
        l = safe_float(entry_lambda, 0.5)
        a = safe_float(entry_a, 0.5)
        b = safe_float(entry_b, 0.5)
        c = safe_complex(entry_c, 0j)
        d = safe_complex(entry_d, 0j)
        e = safe_complex(entry_e, 0j)
        f = safe_complex(entry_f, 0j)

        rho1 = np.array([[a, c], [d, 1-a]], dtype=complex)
        rho2 = np.array([[b, e], [f, 1-b]], dtype=complex)
        rho = l * rho1 + (1-l) * rho2

        entry_one_minus_lambda.configure(state="normal"); entry_one_minus_lambda.delete(0, "end"); entry_one_minus_lambda.insert(0, f"{1-l:.6f}"); entry_one_minus_lambda.configure(state="disabled")
        entry_one_minus_a.configure(state="normal"); entry_one_minus_a.delete(0, "end"); entry_one_minus_a.insert(0, f"{1-a:.6f}"); entry_one_minus_a.configure(state="disabled")
        entry_one_minus_b.configure(state="normal"); entry_one_minus_b.delete(0, "end"); entry_one_minus_b.insert(0, f"{1-b:.6f}"); entry_one_minus_b.configure(state="disabled")

        for entry, val in zip([entry_rho_11, entry_rho_12, entry_rho_21, entry_rho_22], [rho[0,0], rho[0,1], rho[1,0], rho[1,1]]):
            entry.configure(state="normal"); entry.delete(0, "end")
            entry.insert(0, f"{val.real:.4f}" if val.imag == 0 else f"{val:.4f}")
            entry.configure(state="disabled")

        current_rho1, current_rho2, current_l = rho1, rho2, l
    except: pass

def calculate():
    try:
        mode = int(round(mode_slider.get()))
        name = combo.get()
        channels = classical_channels if mode == 0 else quantum_channels
        channel = channels.get(name)
        if not channel: raise ValueError("Select channel")

        p = safe_float(param_entry, 0.1)
        if not (channel['range'][0] <= p <= channel['range'][1]):
            raise ValueError("Parameter out of range")

        if mode == 0:
            lam = safe_float(a_classical_entry, 0.5)
            rate = channel['achievable'](lam, p)
        else:
            update_rho()
            if name == 'Bit-flip' and current_rho1 is not None:
                rate = channel['achievable'](current_rho1, current_rho2, current_l, p)
            else:
                rate = channel['achievable'](current_l, p)

        capacity_label.configure(text=f"Rate: {rate:.4f} bit")
        return current_l if mode == 0 else current_l, p
    except Exception as e:
        capacity_label.configure(text=f"Error: {str(e)[:50]}")
        return None, None

calculate_button.configure(command=lambda: (update_rho() if int(round(mode_slider.get())) == 1 else None, calculate()))

def generate_graph(capacity=False):
    try:
        mode = int(round(mode_slider.get()))
        name = combo.get()
        channels = classical_channels if mode == 0 else quantum_channels
        channel = channels.get(name)
        if not channel: return

        lam, _ = calculate() or (0.5, 0.1)
        params = np.linspace(channel['range'][0], channel['range'][1], 200)

        if capacity:
            ax, canvas, title = ax_capacity, canvas_capacity, f"Capacity: {name}"
            if mode == 0:
                values = [channel['capacity'](p) for p in params]
            elif name == 'Bit-flip':
                values = [get_capacity(channel, p, current_rho1, current_rho2, lam) for p in params]
            else:
                values = [channel.get('capacity', lambda x: 0)(p) for p in params]
        else:
            ax, canvas, title = ax_ach, canvas_ach, f"Rate at λ = {lam:.3f}"
            if mode == 0:
                values = [channel['achievable'](lam, p) for p in params]
            elif name == 'Bit-flip':
                values = [channel['achievable'](current_rho1, current_rho2, lam, p) for p in params]
            else:
                values = [channel['achievable'](lam, p) for p in params]

        ax.clear()
        ax.plot(params, values, 'b-' if capacity else 'r-', linewidth=2)
        ax.set_xlabel(channel['param_name'])
        ax.set_ylabel("Capacity C" if capacity else "Rate R")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        canvas.draw()
    except Exception as e:
        print("Graph error:", e)

def generate_example():
    mode = int(round(mode_slider.get()))
    if mode == 0:
        a = random.uniform(0.2, 0.8)
        a_classical_entry.delete(0, "end")
        a_classical_entry.insert(0, f"{a:.6f}")
        update_classical_a()
    else:
        l = random.uniform(0.2, 0.8); entry_lambda.delete(0, "end"); entry_lambda.insert(0, f"{l:.6f}")
        a = random.uniform(0.2, 0.8); b = 1 - a
        max_coh = np.sqrt(a * b)
        c = random.uniform(-max_coh, max_coh); d = random.uniform(-max_coh, max_coh)
        e = random.uniform(-max_coh, max_coh); f = random.uniform(-max_coh, max_coh)
        entry_a.delete(0, "end"); entry_a.insert(0, f"{a:.6f}")
        entry_c.delete(0, "end"); entry_c.insert(0, f"{c:.6f}")
        entry_d.delete(0, "end"); entry_d.insert(0, f"{d:.6f}")
        entry_b.delete(0, "end"); entry_b.insert(0, f"{b:.6f}")
        entry_e.delete(0, "end"); entry_e.insert(0, f"{e:.6f}")
        entry_f.delete(0, "end"); entry_f.insert(0, f"{f:.6f}")

    update_rho() if mode == 1 else None
    calculate(); generate_graph(True); generate_graph(False)

example_button.configure(command=generate_example)

def update_mode(val):
    mode = int(round(val))
    channels = classical_channels if mode == 0 else quantum_channels
    combo.configure(values=list(channels.keys()))
    combo.set(list(channels.keys())[0])
    quantum_matrix_frame.pack_forget() if mode == 0 else classical_matrix_frame.pack_forget()
    classical_matrix_frame.pack(pady=10) if mode == 0 else quantum_matrix_frame.pack(pady=10)
    if mode == 0:
        a_classical_entry.delete(0, "end"); a_classical_entry.insert(0, "0.5"); update_classical_a()
    else:
        for e in [entry_lambda, entry_a, entry_c, entry_d, entry_b, entry_e, entry_f]: e.delete(0, "end")
        entry_lambda.insert(0, "0.5"); entry_a.insert(0, "0.5"); entry_b.insert(0, "0.5")
        update_rho()
    update_channel(combo.get())

def update_channel(name):
    mode = int(round(mode_slider.get()))
    channels = classical_channels if mode == 0 else quantum_channels
    channel = channels.get(name)
    if not channel: return
    d_text.delete("0.0", "end")
    d_text.insert("0.0", f"Channel: {name}\nParameter: {channel['param_name']} ∈ [{channel['range'][0]}, {channel['range'][1]}]\n\nFormula:\n{channel['formula']}\n\nDescription:\n{channel['description']}")
    formula_label.configure(text=f"Formula: {channel['formula']}")
    param_label.configure(text=f"Parameter {channel['param_name']}:")
    calculate(); generate_graph(True); generate_graph(False)

# Events
mode_slider.configure(command=update_mode)
combo.configure(command=update_channel)
for widget in [a_classical_entry, param_entry, entry_lambda, entry_a, entry_b, entry_c, entry_d, entry_e, entry_f]:
    widget.bind("<KeyRelease>", lambda e: (update_rho() if int(round(mode_slider.get())) == 1 else update_classical_a(), calculate(), generate_graph(False)))

update_mode(0)
root.mainloop()
