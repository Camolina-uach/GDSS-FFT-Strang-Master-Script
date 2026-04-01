# GDSS FFT + Strang Master Script

This repository contains a pseudo-spectral Fourier solver for a **generalized Davey–Stewartson system (GDSS)** with **three coupled equations**, advanced in time with **Strang splitting**.

The script is designed as a **master benchmarking workflow**. It combines:

- a library of initial conditions,
- a time-step sweep in `dt`,
- invariant monitoring,
- a final production run with stored snapshots,
- 2D and 3D post-processing plots,
- and lightweight animations for qualitative inspection.

## Main file

- `gdss_fft_strang_master_polished.py`

## Features

The script includes:

1. **Initial-condition selector**
   - Gaussian pulse
   - Sech stripe
   - Sech lump
   - Ring Gaussian
   - Gaussian vortex
   - Two-Gaussian collision
   - Modulated plane wave
   - Super-Gaussian stripe

2. **Temporal benchmark (`dt` sweep)**
   - Runs the same case for multiple time steps
   - Reports CPU time and conservation metrics
   - Optionally computes self-errors between successive runs

3. **Invariant diagnostics**
   - Mass `M`
   - Momenta `Jx`, `Jy`
   - Diagnostic energy `Ediag`
   - Peak amplitude `max|u|`

4. **Final production run**
   - Stores snapshots of `|u|`
   - Computes final long-wave fields `w` and `v`

5. **Visualization utilities**
   - Invariant plots
   - 2D animation of `|u|`
   - Animated 1D line cut
   - `x–t` heatmap at a fixed `y` slice
   - 2D final fields: `|u|`, `w`, `v`
   - 3D final surfaces: `|u|`, `w`, `v`

---

## Numerical method

The code uses:

- **FFT pseudo-spectral discretization** in space,
- **2/3-rule dealiasing**,
- **Strang splitting** for time integration,
- and a **Fourier-space solution of the coupled elliptic subsystem** for the long-wave potentials.

The script is intended for reproducible numerical experiments and benchmarking studies on GDSS dynamics.

---

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `numpy`
- `matplotlib`
- `tqdm`
- `tabulate` *(optional but recommended for cleaner benchmark tables)*

---

## Recommended Python version

- **Python 3.10+**

The script should also work on nearby modern Python versions as long as the listed dependencies are installed.

---

## How to run

Run the script directly:

```bash
python gdss_fft_strang_master_polished.py
```

---

## Basic workflow inside the script

The main block follows this sequence:

1. Define **GDSS parameters** in `params`
2. Define the **base simulation configuration** in `base_config`
3. Build the **initial-condition library**
4. Select an initial condition with `IC_NAME`
5. Run a **time-step sweep** through `dt_sweep_2d(...)`
6. Choose the finest `dt`
7. Run a **final simulation with snapshots** using `run_simulation_2d(...)`
8. Generate the selected plots and animations

---

## Editing the simulation setup

### 1. Change the physical/numerical parameters

In the `__main__` section:

```python
params = {
    "alpha": 1.0, "beta": 1.0,
    "gamma": 1.0, "xi": 1.0,
    "psi": 2.0, "eta": 2.0,
    "phi": 2.0, "chi": 1.0,
    "theta": 1.0,
    "coupling_sign": +1.0,
}
```

### 2. Change the grid and final time

```python
base_config = dict(
    Nx=256, Ny=256,
    Lx=80.0, Ly=80.0,
    Tmax=2.0,
    dealias_on=True,
    monitor_points=80,
    store_snapshots=False,
    nframes=180
)
```

### 3. Select the initial condition

```python
IC_NAME = "ring"
```

Available options:

- `gaussian`
- `sech_stripe`
- `sech_lump`
- `ring`
- `vortex_m1`
- `two_gaussians_collision`
- `plane_wave_modulated`
- `supergauss_stripe`

### 4. Override preset parameters without editing the library

```python
IC_OVERRIDE = dict(A=1.15, kx0=1.0)
```

Set `IC_OVERRIDE = None` if no override is needed.

### 5. Change the benchmark time steps

```python
dt_list = [5e-3, 2.5e-3, 1.25e-3]
```

---

## Outputs

Depending on the switches in the main block, the script can produce:

- benchmark tables in the console,
- conservation summaries,
- invariant curves,
- 2D animations,
- line-cut animations,
- `x–t` heatmaps,
- 2D final-field plots,
- 3D final-field plots.

The animation utility also supports saving a GIF when `save_gif=True`.

---

## Notes on interpretation

- `M` is the total mass.
- `Jx` and `Jy` are momentum diagnostics associated with the short-wave field.
- `Ediag` is a diagnostic energy made of kinetic, local nonlinear, and field contributions.
- `max|u|` is useful for tracking focusing, spreading, or amplitude growth.

For convergence-style studies, the script reports:

- final relative or absolute error in the monitored invariants,
- maximum drift over time,
- and a self-error between consecutive `dt` runs.

---

## Repository suggestions

For a cleaner public repository, consider adding:

- a small folder for saved figures or GIFs,
- a `LICENSE` file,
- a short example section with one recommended test case,
- and, if relevant for publication, a DOI-backed archive in **Zenodo**.

A minimal repository structure could be:

```text
.
├── gdss_fft_strang_master_polished.py
├── README.md
├── requirements.txt
└── outputs/
```

---

## Citation / reuse

If you use or adapt this script in academic work, it is good practice to cite the associated paper, preprint, or repository release that documents the numerical method and benchmark setting.
