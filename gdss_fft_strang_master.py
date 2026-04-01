# -*- coding: utf-8 -*-
"""
GDSS (three coupled equations) — FFT pseudo-spectral + Strang splitting
MASTER SCRIPT (benchmark + initial-condition selector)

Includes:
  1) Initial-condition selector (gaussian, stripe, lump, ring, vortex, two pulses, etc.)
  2) Time-step sweep in dt (temporal benchmark)
  3) Invariants: M, Jx, Jy, Ediag, max|u|
  4) Final run with |u| snapshots
  5) 2D animation of |u|
  6) Fast x–t heatmap at the slice y=y0
  7) Final 3D plots of |u|, w, v (paper-style)
  8) Final 2D plots of |u|, w, v

Author: adapted master script for GDSS benchmarking
"""

import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Button, Slider
from tqdm import tqdm

try:
    from tabulate import tabulate

    HAVE_TABULATE = True
except Exception:
    HAVE_TABULATE = False


# ==============================================================================
# GRID SETUP + DEALIASING (2/3 rule)
# ==============================================================================
def setup_grid(Lx, Ly, Nx, Ny):
    """Build the physical and Fourier grids, together with the 2/3 dealiasing mask."""
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    Kx, Ky = np.meshgrid(kx, ky, indexing="xy")  # shape: (Ny, Nx)
    X, Y = np.meshgrid(x, y, indexing="xy")

    kmax_x, kmax_y = np.max(np.abs(kx)), np.max(np.abs(ky))
    dealias = (np.abs(Kx) <= (2 / 3) * kmax_x) & (np.abs(Ky) <= (2 / 3) * kmax_y)

    return x, y, X, Y, Kx, Ky, dx, dy, dealias


def apply_dealias(Uhat, mask):
    """Apply the spectral dealiasing mask if it is enabled."""
    return Uhat * mask if mask is not None else Uhat


# ==============================================================================
# POTENTIAL SOLVER (w_hat, v_hat) IN FOURIER SPACE
# ==============================================================================
def solve_potentials_hat(u, Kx, Ky, params, dealias_mask=None):
    """Solve the coupled elliptic subsystem for the long-wave potentials in Fourier space."""
    psi, eta = params["psi"], params["eta"]
    phi, chi = params["phi"], params["chi"]
    theta = params["theta"]

    M11 = -(psi * Kx**2 + eta * Ky**2)
    M12 = -(theta * Kx * Ky)
    M22 = -(phi * Kx**2 + chi * Ky**2)
    det = M11 * M22 - M12**2

    rho_hat = np.fft.fft2(np.abs(u) ** 2)
    rho_hat = apply_dealias(rho_hat, dealias_mask)

    R1 = 1j * Kx * rho_hat
    R2 = 1j * Ky * rho_hat
    R1 = apply_dealias(R1, dealias_mask)
    R2 = apply_dealias(R2, dealias_mask)

    # Numerical robustness for singular modes.
    mask_sing = np.abs(det) < 1e-30
    det_safe = det.copy()
    det_safe[mask_sing] = 1.0
    det_safe[0, 0] = 1.0

    w_hat = (R1 * M22 - R2 * M12) / det_safe
    v_hat = (R2 * M11 - R1 * M12) / det_safe

    w_hat[0, 0] = 0.0
    v_hat[0, 0] = 0.0
    w_hat[mask_sing] = 0.0
    v_hat[mask_sing] = 0.0

    w_hat = apply_dealias(w_hat, dealias_mask)
    v_hat = apply_dealias(v_hat, dealias_mask)

    return w_hat, v_hat


# ==============================================================================
# INVARIANTS / DIAGNOSTICS
# ==============================================================================
def invariants(u, w_hat, v_hat, Kx, Ky, params, dx, dy, dealias_mask=None):
    """
    Compute the diagnostic invariants.

    Returns
    -------
    M : float
        Mass.
    Jx, Jy : float
        Momentum components associated with u.
    Ediag : float
        Diagnostic energy (kinetic + local nonlinear + field contribution).
    umax : float
        Peak amplitude max|u|.
    """
    cell = dx * dy
    invN = 1.0 / u.size

    rho = np.abs(u) ** 2
    M = cell * np.sum(rho)

    # Momentum components.
    u_hat = np.fft.fft2(u)
    u_hat = apply_dealias(u_hat, dealias_mask)
    ux = np.fft.ifft2(1j * Kx * u_hat)
    uy = np.fft.ifft2(1j * Ky * u_hat)
    Jx = 2.0 * cell * np.sum(np.imag(np.conj(u) * ux))
    Jy = 2.0 * cell * np.sum(np.imag(np.conj(u) * uy))

    # Kinetic energy via Parseval.
    Ek = cell * invN * np.sum(
        (params["alpha"] * Kx**2 + params["beta"] * Ky**2) * np.abs(u_hat) ** 2
    )

    # Local nonlinear energy.
    Eq = 0.5 * params["gamma"] * cell * np.sum(rho**2)

    # Field energy.
    wx = np.fft.ifft2(1j * Kx * w_hat).real
    wy = np.fft.ifft2(1j * Ky * w_hat).real
    vx = np.fft.ifft2(1j * Kx * v_hat).real
    vy = np.fft.ifft2(1j * Ky * v_hat).real

    Q = (
        params["psi"] * wx**2
        + params["eta"] * wy**2
        + params["phi"] * vx**2
        + params["chi"] * vy**2
        + params["theta"] * (wx * vy + wy * vx)
    )

    Efield = 0.5 * params["xi"] * cell * np.sum(Q)

    Ediag = float(np.real(Ek + Eq + Efield))
    umax = float(np.max(np.abs(u)))

    return float(np.real(M)), float(np.real(Jx)), float(np.real(Jy)), Ediag, umax


def drift_metrics(arr, use_relative=True):
    """Compute final error and maximum drift with respect to the initial value."""
    a0 = arr[0]
    if (not use_relative) or (abs(a0) < 1e-30):
        err_final = abs(arr[-1] - a0)
        drift_max = np.max(np.abs(arr - a0))
        return err_final, drift_max
    err_final = abs(arr[-1] - a0) / abs(a0)
    drift_max = np.max(np.abs(arr - a0)) / abs(a0)
    return err_final, drift_max


def align_global_phase(u_ref, u):
    """Align the global phase of u with respect to the reference state u_ref."""
    inner = np.vdot(u_ref, u)
    if np.abs(inner) < 1e-30:
        return u
    phase = inner / np.abs(inner)
    return u / phase


# ==============================================================================
# INITIAL CONDITIONS (ICs) — soliton-like and benchmark test cases
# ==============================================================================
def ic_gaussian_pulse_2d(
    X,
    Y,
    A=1.0,
    sigma_x=2.5,
    sigma_y=2.5,
    kx0=2.0,
    ky0=0.0,
    x0=-8.0,
    y0=0.0,
):
    """Localized Gaussian pulse with carrier wavevector (kx0, ky0)."""
    env = A * np.exp(-((X - x0) ** 2) / (2 * sigma_x**2) - ((Y - y0) ** 2) / (2 * sigma_y**2))
    phase = np.exp(1j * (kx0 * X + ky0 * Y))
    return (env * phase).astype(np.complex128)


def ic_sech_stripe_2d(
    X,
    Y,
    A=1.2,
    width_x=3.0,
    kx0=1.2,
    ky0=0.0,
    x0=-15.0,
    y0=0.0,
    sigma_y=25.0,
):
    """
    Stripe / line-soliton-like profile:
      - sech profile in x,
      - wide Gaussian profile in y (nearly uniform for large sigma_y).
    """
    env_x = A / np.cosh((X - x0) / width_x)
    env_y = 1.0 if sigma_y is None else np.exp(-((Y - y0) ** 2) / (2.0 * sigma_y**2))
    phase = np.exp(1j * (kx0 * X + ky0 * Y))
    return (env_x * env_y * phase).astype(np.complex128)


def ic_sech_lump_2d(
    X,
    Y,
    A=1.0,
    width_x=3.0,
    width_y=3.0,
    kx0=0.8,
    ky0=0.4,
    x0=-10.0,
    y0=0.0,
):
    """Localized lump-like profile: sech(x) * sech(y)."""
    env = (A / np.cosh((X - x0) / width_x)) * (1.0 / np.cosh((Y - y0) / width_y))
    phase = np.exp(1j * (kx0 * X + ky0 * Y))
    return (env * phase).astype(np.complex128)


def ic_ring_gaussian_2d(X, Y, A=1.0, r0=10.0, sigma_r=2.0, x0=0.0, y0=0.0, k_r=0.0):
    """Radial Gaussian ring, useful for focusing/dispersion tests."""
    R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    env = A * np.exp(-((R - r0) ** 2) / (2 * sigma_r**2))
    phase = np.exp(1j * k_r * R)
    return (env * phase).astype(np.complex128)


def ic_vortex_gaussian_2d(
    X,
    Y,
    A=1.0,
    sigma=5.0,
    m=1,
    x0=0.0,
    y0=0.0,
    kx0=0.0,
    ky0=0.0,
):
    """Gaussian vortex with topological charge m."""
    Xc = X - x0
    Yc = Y - y0
    R = np.sqrt(Xc**2 + Yc**2) + 1e-14
    TH = np.arctan2(Yc, Xc)

    env = A * (R / sigma) ** abs(m) * np.exp(-(R**2) / (2 * sigma**2))
    phase = np.exp(1j * (m * TH + kx0 * X + ky0 * Y))
    return (env * phase).astype(np.complex128)


def ic_two_gaussians_2d(
    X,
    Y,
    A1=1.0,
    A2=1.0,
    sigma1=2.5,
    sigma2=2.5,
    x1=-10.0,
    y1=0.0,
    x2=10.0,
    y2=0.0,
    kx1=1.2,
    ky1=0.0,
    kx2=-1.2,
    ky2=0.0,
    rel_phase=0.0,
):
    """Two Gaussian pulses for interaction/collision studies."""
    u1 = A1 * np.exp(-((X - x1) ** 2 + (Y - y1) ** 2) / (2 * sigma1**2)) * np.exp(
        1j * (kx1 * X + ky1 * Y)
    )
    u2 = A2 * np.exp(-((X - x2) ** 2 + (Y - y2) ** 2) / (2 * sigma2**2)) * np.exp(
        1j * (kx2 * X + ky2 * Y + rel_phase)
    )
    return (u1 + u2).astype(np.complex128)


def ic_plane_wave_modulated_2d(
    X,
    Y,
    A=0.6,
    kx0=1.0,
    ky0=0.0,
    mod_amp=0.08,
    mod_kx=0.35,
    mod_ky=0.20,
):
    """Modulated plane wave, useful for stability/modulation tests."""
    amp = A * (1.0 + mod_amp * np.cos(mod_kx * X + mod_ky * Y))
    phase = np.exp(1j * (kx0 * X + ky0 * Y))
    return (amp * phase).astype(np.complex128)


def ic_super_gaussian_stripe_2d(
    X,
    Y,
    A=1.0,
    width_x=4.0,
    order=4,
    sigma_y=25.0,
    kx0=1.0,
    ky0=0.0,
    x0=-12.0,
    y0=0.0,
):
    """Stripe with a flatter core: super-Gaussian in x and Gaussian in y."""
    env_x = A * np.exp(-(np.abs(X - x0) / width_x) ** (2 * order))
    env_y = 1.0 if sigma_y is None else np.exp(-((Y - y0) ** 2) / (2 * sigma_y**2))
    phase = np.exp(1j * (kx0 * X + ky0 * Y))
    return (env_x * env_y * phase).astype(np.complex128)


# Master dictionary of initial-condition presets.
def get_ic_library(base_config):
    """Return the library of initial-condition presets based on the current domain size."""
    Lx = base_config["Lx"]
    Ly = base_config["Ly"]
    return {
        "gaussian": dict(
            fn=ic_gaussian_pulse_2d,
            kwargs=dict(A=1.0, sigma_x=3.0, sigma_y=3.0, kx0=1.8, ky0=0.0, x0=-Lx / 6, y0=0.0),
        ),
        "sech_stripe": dict(
            fn=ic_sech_stripe_2d,
            kwargs=dict(A=1.25, width_x=2.8, kx0=1.2, ky0=0.0, x0=-Lx / 3, y0=0.0, sigma_y=Ly / 3),
        ),
        "sech_lump": dict(
            fn=ic_sech_lump_2d,
            kwargs=dict(A=1.0, width_x=3.2, width_y=3.2, kx0=0.9, ky0=0.25, x0=-Lx / 5, y0=0.0),
        ),
        "ring": dict(
            fn=ic_ring_gaussian_2d,
            kwargs=dict(A=1.0, r0=8.0, sigma_r=1.8, x0=0.0, y0=0.0, k_r=0.2),
        ),
        "vortex_m1": dict(
            fn=ic_vortex_gaussian_2d,
            kwargs=dict(A=1.2, sigma=5.5, m=1, x0=0.0, y0=0.0, kx0=0.0, ky0=0.0),
        ),
        "two_gaussians_collision": dict(
            fn=ic_two_gaussians_2d,
            kwargs=dict(
                A1=1.0,
                A2=1.0,
                sigma1=2.8,
                sigma2=2.8,
                x1=-12.0,
                y1=0.0,
                x2=12.0,
                y2=0.0,
                kx1=1.3,
                ky1=0.0,
                kx2=-1.3,
                ky2=0.0,
                rel_phase=0.0,
            ),
        ),
        "plane_wave_modulated": dict(
            fn=ic_plane_wave_modulated_2d,
            kwargs=dict(A=0.55, kx0=1.0, ky0=0.2, mod_amp=0.12, mod_kx=0.35, mod_ky=0.18),
        ),
        "supergauss_stripe": dict(
            fn=ic_super_gaussian_stripe_2d,
            kwargs=dict(A=1.0, width_x=4.2, order=4, sigma_y=Ly / 3, kx0=1.0, ky0=0.0, x0=-Lx / 4, y0=0.0),
        ),
    }


def make_initial_condition(X, Y, ic_name, ic_library, ic_override_kwargs=None):
    """Build the selected initial condition, optionally overriding preset parameters."""
    if ic_name not in ic_library:
        raise ValueError(f"IC '{ic_name}' not found. Available options: {list(ic_library.keys())}")
    entry = ic_library[ic_name]
    kwargs = dict(entry["kwargs"])
    if ic_override_kwargs:
        kwargs.update(ic_override_kwargs)
    return entry["fn"](X, Y, **kwargs), kwargs


# ==============================================================================
# SIMULATION (Strang + FFT) — 2D
# ==============================================================================
def run_simulation_2d(
    dt,
    Tmax,
    Nx,
    Ny,
    Lx,
    Ly,
    params,
    ic_name,
    ic_library,
    ic_override_kwargs=None,
    dealias_on=True,
    monitor_points=60,
    store_snapshots=False,
    nframes=180,
):
    """Run the 2D GDSS simulation using FFT pseudo-spectral discretization and Strang splitting."""
    x, y, X, Y, Kx, Ky, dx, dy, dealias = setup_grid(Lx, Ly, Nx, Ny)
    dealias_mask = dealias if dealias_on else None

    # Initial condition.
    u0, used_ic_kwargs = make_initial_condition(X, Y, ic_name, ic_library, ic_override_kwargs)
    u_hat = np.fft.fft2(u0)
    u_hat = apply_dealias(u_hat, dealias_mask)
    u = np.fft.ifft2(u_hat)

    # Linear half-step propagator.
    lin_half = np.exp(-1j * 0.5 * dt * (params["alpha"] * Kx**2 + params["beta"] * Ky**2))

    Nt = int(round(Tmax / dt))
    stride_mon = max(1, Nt // max(1, monitor_points))

    # Snapshot storage for |u|.
    snaps = None
    snap_steps = None
    if store_snapshots:
        snap_steps = np.unique(np.round(np.linspace(0, Nt, nframes)).astype(int))
        snap_steps = snap_steps[(snap_steps >= 0) & (snap_steps <= Nt)]
        snaps = [None] * len(snap_steps)
        snap_pos = 0

    # Diagnostic histories.
    t_hist, M_hist, Jx_hist, Jy_hist, E_hist, umax_hist = [], [], [], [], [], []

    def sample(t, u_):
        w_hat, v_hat = solve_potentials_hat(u_, Kx, Ky, params, dealias_mask)
        M, Jx, Jy, E, umax = invariants(u_, w_hat, v_hat, Kx, Ky, params, dx, dy, dealias_mask)
        t_hist.append(t)
        M_hist.append(M)
        Jx_hist.append(Jx)
        Jy_hist.append(Jy)
        E_hist.append(E)
        umax_hist.append(umax)

    coupling_sign = params.get("coupling_sign", +1.0)

    # t = 0.
    sample(0.0, u)
    if store_snapshots:
        snaps[0] = np.abs(u).astype(np.float32)

    for n in range(Nt):
        # ---- Linear half step ----
        u_hat = np.fft.fft2(u)
        u_hat = apply_dealias(u_hat, dealias_mask)
        u = np.fft.ifft2(u_hat * lin_half)

        # ---- Nonlinear full step ----
        w_hat, v_hat = solve_potentials_hat(u, Kx, Ky, params, dealias_mask)
        wx = np.fft.ifft2(1j * Kx * w_hat).real
        vy = np.fft.ifft2(1j * Ky * v_hat).real

        V = params["gamma"] * np.abs(u) ** 2 + coupling_sign * params["xi"] * (wx + vy)
        u = u * np.exp(-1j * dt * V)

        # Dealias after the nonlinear stage.
        u_hat = np.fft.fft2(u)
        u_hat = apply_dealias(u_hat, dealias_mask)
        u = np.fft.ifft2(u_hat)

        # ---- Linear half step ----
        u_hat = np.fft.fft2(u)
        u_hat = apply_dealias(u_hat, dealias_mask)
        u = np.fft.ifft2(u_hat * lin_half)

        t = (n + 1) * dt

        if ((n + 1) % stride_mon == 0) or (n == Nt - 1):
            sample(t, u)

        if store_snapshots:
            while (snap_pos + 1) < len(snap_steps) and (n + 1) == snap_steps[snap_pos + 1]:
                snap_pos += 1
                snaps[snap_pos] = np.abs(u).astype(np.float32)

    # Fill any missing snapshots as a safety net.
    if store_snapshots:
        last = None
        for i in range(len(snaps)):
            if snaps[i] is None:
                snaps[i] = last
            else:
                last = snaps[i]
        if snaps[-1] is None:
            snaps[-1] = np.abs(u).astype(np.float32)

    # Final potentials (for plotting w and v).
    w_hat_fin, v_hat_fin = solve_potentials_hat(u, Kx, Ky, params, dealias_mask)
    w_fin = np.fft.ifft2(w_hat_fin).real
    v_fin = np.fft.ifft2(v_hat_fin).real

    return {
        "dt": dt,
        "Tmax": Tmax,
        "Nt": Nt,
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "Kx": Kx,
        "Ky": Ky,
        "dx": dx,
        "dy": dy,
        "u0": u0,
        "u_final": u,
        "w_final": w_fin,
        "v_final": v_fin,
        "t": np.array(t_hist),
        "M": np.array(M_hist),
        "Jx": np.array(Jx_hist),
        "Jy": np.array(Jy_hist),
        "E": np.array(E_hist),
        "umax": np.array(umax_hist),
        "snaps": snaps,
        "snap_steps": snap_steps,
        "params": params,
        "ic_name": ic_name,
        "ic_kwargs_used": used_ic_kwargs,
    }


# ==============================================================================
# dt SWEEP + SELF-ERROR
# ==============================================================================
def dt_sweep_2d(
    dt_list,
    base_config,
    params,
    ic_name,
    ic_library,
    ic_override_kwargs=None,
    compare_self_error=True,
    align_phase=True,
):
    """Run a time-step sweep and report drift/error diagnostics."""
    rows = []
    results = []

    u_prev = None
    prev_self = None

    for dt in tqdm(dt_list, desc=f"dt sweep ({ic_name})"):
        t0 = time.time()
        res = run_simulation_2d(
            dt=dt,
            params=params,
            ic_name=ic_name,
            ic_library=ic_library,
            ic_override_kwargs=ic_override_kwargs,
            **base_config,
        )
        cpu = time.time() - t0

        Merr, Mdr = drift_metrics(res["M"], use_relative=True)
        Jx_err, _ = drift_metrics(res["Jx"], use_relative=False)
        Jy_err, _ = drift_metrics(res["Jy"], use_relative=False)
        Eerr, Edr = drift_metrics(res["E"], use_relative=True)

        self_err_str = "-"
        rate_str = "-"

        if compare_self_error:
            if u_prev is not None:
                u_curr = res["u_final"]
                if align_phase:
                    u_curr = align_global_phase(u_prev, u_curr)
                dx, dy = res["dx"], res["dy"]
                self_err = np.sqrt(dx * dy * np.sum(np.abs(u_curr - u_prev) ** 2))
                self_err_str = f"{self_err:.2e}"
                if isinstance(prev_self, float) and prev_self > 0:
                    rate = np.log2(prev_self / self_err)
                    rate_str = f"{rate:.2f}"
                prev_self = float(self_err)
                u_prev = u_curr.copy()
            else:
                u_prev = res["u_final"].copy()

        rows.append(
            [
                f"{dt:.6e}",
                f"{cpu:.2f}",
                f"{Merr:.2e}",
                f"{Mdr:.2e}",
                f"{Jx_err:.2e}",
                f"{Jy_err:.2e}",
                f"{Eerr:.2e}",
                f"{Edr:.2e}",
                self_err_str,
                rate_str,
            ]
        )
        results.append(res)

    headers = [
        "dt",
        "CPU(s)",
        "M err(final)",
        "M drift(max)",
        "|Jx| err(abs)",
        "|Jy| err(abs)",
        "E err(final)",
        "E drift(max)",
        "self_err",
        "rate",
    ]

    print("\n" + "=" * 115)
    if HAVE_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print(headers)
        for r in rows:
            print(r)
    print("=" * 115 + "\n")

    return results


# ==============================================================================
# INVARIANT PLOTS
# ==============================================================================
def plot_invariants_2d(res):
    """Plot time histories of the main invariants and peak amplitude."""
    t = res["t"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)

    axs[0].plot(t, res["M"], lw=2)
    axs[0].set_title("Mass M")
    axs[0].grid(True, linestyle=":")

    axs[1].plot(t, res["E"], lw=2)
    axs[1].set_title("Energy Ediag")
    axs[1].grid(True, linestyle=":")

    axs[2].plot(t, res["Jx"], lw=2)
    axs[2].set_title("Momentum Jx")
    axs[2].grid(True, linestyle=":")

    axs[3].plot(t, res["Jy"], lw=2)
    axs[3].set_title("Momentum Jy")
    axs[3].grid(True, linestyle=":")

    for ax in axs:
        ax.set_xlabel("t")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2), useOffset=False)

    plt.show()

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(t, res["umax"], lw=2)
    plt.title("Peak amplitude max|u|")
    plt.xlabel("t")
    plt.grid(True, linestyle=":")
    plt.show()


# ==============================================================================
# 2D ANIMATION (imshow of |u|)
# ==============================================================================
def animate_snaps_2d(
    res,
    zoom=25.0,
    interval_ms=40,
    cmap="viridis",
    save_gif=False,
    gif_name="gdss_2d.gif",
    fps=20,
    dpi=160,
):
    """Animate the stored |u| snapshots as a 2D heatmap."""
    snaps = res["snaps"]
    if snaps is None:
        print("No snapshots available. Run with store_snapshots=True.")
        return None

    x, y = res["x"], res["y"]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Lx = np.ptp(x) + dx
    Ly = np.ptp(y) + dy
    extent = (-Lx / 2, Lx / 2, -Ly / 2, Ly / 2)

    vmax = float(np.max(np.stack(snaps)))

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(snaps[0], origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("|u| (snapshot 0)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(i):
        im.set_data(snaps[i])
        title.set_text(f"|u| (snapshot {i}/{len(snaps) - 1})")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(snaps), interval=interval_ms, blit=False)

    if save_gif:
        writer = animation.PillowWriter(fps=fps)
        ani.save(gif_name, writer=writer, dpi=dpi)
        print(f"GIF saved: {gif_name}")

    plt.show()
    return ani


# ==============================================================================
# 2D ANIMATION (NEW): animated 1D cut |u(x,y0,t)| or |u|^2
# ==============================================================================
def animate_linecut_2d(res, y_slice=0.0, xlim=None, quantity="abs", interval_ms=60, lw=2.0):
    """Animate a 1D slice of the solution at fixed y = y_slice."""
    snaps = res.get("snaps", None)
    snap_steps = res.get("snap_steps", None)
    if snaps is None or snap_steps is None:
        print("No snapshots available. Run with store_snapshots=True.")
        return None

    x = res["x"]
    y = res["y"]
    t_snaps = res["dt"] * snap_steps.astype(float)

    iy0 = int(np.argmin(np.abs(y - y_slice)))

    if xlim is None:
        ix = np.arange(len(x))
    else:
        xmin, xmax = xlim
        ix = np.where((x >= xmin) & (x <= xmax))[0]
        if ix.size == 0:
            raise ValueError("xlim does not intersect the domain.")

    # Determine a convenient y-axis scale.
    ymax = 0.0
    for S in snaps:
        line_abs = S[iy0, ix]
        yy = line_abs if quantity == "abs" else line_abs * line_abs
        ymax = max(ymax, float(np.max(yy)))
    if ymax <= 0:
        ymax = 1.0

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    line, = ax.plot(x[ix], np.zeros_like(x[ix]), lw=lw)
    ax.set_xlim(x[ix][0], x[ix][-1])
    ax.set_ylim(0.0, 1.05 * ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("|u|" if quantity == "abs" else r"$|u|^2$")
    title = ax.set_title(f"Animated slice at y={y[iy0]:.3g}")
    ax.grid(True, linestyle=":")

    def update(i):
        line_abs = snaps[i][iy0, ix]
        yy = line_abs if quantity == "abs" else line_abs * line_abs
        line.set_ydata(yy)
        title.set_text(f"Animated slice at y={y[iy0]:.3g} | t≈{t_snaps[i]:.3f}")
        return line, title

    ani = animation.FuncAnimation(fig, update, frames=len(snaps), interval=interval_ms, blit=False)
    plt.show()
    return ani


# ==============================================================================
# x–t HEATMAP (fast) at slice y=y0
# ==============================================================================
def plot_xt_heatmap_2d(
    res,
    y_slice=0.0,
    xlim=None,
    t_centered=False,
    quantity="abs2",
    cmap="viridis",
    t_stride=1,
    x_stride=1,
):
    """Plot an x–t heatmap at a fixed y-slice using the stored snapshots."""
    snaps = res.get("snaps", None)
    snap_steps = res.get("snap_steps", None)
    if snaps is None or snap_steps is None:
        raise ValueError("Snapshots are required. Run with store_snapshots=True.")

    x = res["x"]
    y = res["y"]
    dt = float(res["dt"])

    t = dt * snap_steps.astype(float)
    if t_centered:
        t = t - 0.5 * (t[0] + t[-1])

    iy0 = int(np.argmin(np.abs(y - y_slice)))

    if xlim is None:
        ix = np.arange(len(x))
    else:
        xmin, xmax = xlim
        ix = np.where((x >= xmin) & (x <= xmax))[0]
        if ix.size == 0:
            raise ValueError("xlim does not intersect the domain.")

    ix = ix[:: max(1, int(x_stride))]
    it = np.arange(0, len(snaps), max(1, int(t_stride)))
    if it[-1] != len(snaps) - 1:
        it = np.append(it, len(snaps) - 1)

    x_sel = x[ix]
    t_sel = t[it]

    Z = np.empty((len(t_sel), len(x_sel)), dtype=np.float32)
    for j, k in enumerate(it):
        line_abs = snaps[k][iy0, ix]
        Z[j, :] = line_abs if quantity == "abs" else line_abs * line_abs

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    extent = (x_sel[0], x_sel[-1], t_sel[0], t_sel[-1])
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(rf"Heatmap {('|u|' if quantity == 'abs' else '|u|^2')} (slice at y={y[iy0]:.3g})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.show()


# ==============================================================================
# 1D PROFILES at selected times (slice y=y0)
# ==============================================================================
def plot_line_profiles_2d(res, y_slice=0.0, xlim=None, times=(0, 1, 2, 3, 4), quantity="abs", lw=2.0):
    """Plot selected 1D profiles at fixed y = y_slice."""
    snaps = res.get("snaps", None)
    snap_steps = res.get("snap_steps", None)
    if snaps is None or snap_steps is None:
        raise ValueError("Snapshots are required. Run with store_snapshots=True.")

    x = res["x"]
    y = res["y"]
    t_snaps = res["dt"] * snap_steps.astype(float)
    iy0 = int(np.argmin(np.abs(y - y_slice)))

    if xlim is None:
        ix = np.arange(len(x))
    else:
        xmin, xmax = xlim
        ix = np.where((x >= xmin) & (x <= xmax))[0]
        if ix.size == 0:
            raise ValueError("xlim does not intersect the domain.")

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for tt in times:
        k = int(np.argmin(np.abs(t_snaps - tt)))
        line_abs = snaps[k][iy0, ix]
        yy = line_abs if quantity == "abs" else line_abs * line_abs
        ax.plot(x[ix], yy, lw=lw, label=f"t≈{t_snaps[k]:.3g}")

    ax.set_xlabel("x")
    ax.set_ylabel("|u|" if quantity == "abs" else r"$|u|^2$")
    ax.set_title(rf"1D profiles (slice at y={y[iy0]:.3g})")
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.show()


# ==============================================================================
# FINAL 3D PLOTS (|u|, w, v) — paper / screenshot style
# ==============================================================================
def plot_final_uvwv_3d(res, zoom=20.0, cmap_u="viridis", cmap_wv="plasma"):
    """Plot the final fields |u|, w, and v as 3D surfaces."""
    x, y = res["x"], res["y"]
    X, Y = res["X"], res["Y"]
    uabs = np.abs(res["u_final"])
    w = res["w_final"]
    v = res["v_final"]

    if zoom is not None:
        ix = np.where((x >= -zoom) & (x <= zoom))[0]
        iy = np.where((y >= -zoom) & (y <= zoom))[0]
        Xp = X[np.ix_(iy, ix)]
        Yp = Y[np.ix_(iy, ix)]
        Up = uabs[np.ix_(iy, ix)]
        Wp = w[np.ix_(iy, ix)]
        Vp = v[np.ix_(iy, ix)]
    else:
        Xp, Yp, Up, Wp, Vp = X, Y, uabs, w, v

    fig = plt.figure(figsize=(16, 5.6))
    axs = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]

    axs[0].plot_surface(Xp, Yp, Up, cmap=cmap_u, linewidth=0, antialiased=True)
    axs[0].set_title(r"$|u|$")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    axs[1].plot_surface(Xp, Yp, Wp, cmap=cmap_wv, linewidth=0, antialiased=True)
    axs[1].set_title(r"$w$")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    axs[2].plot_surface(Xp, Yp, Vp, cmap=cmap_wv, linewidth=0, antialiased=True)
    axs[2].set_title(r"$v$")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")

    for ax in axs:
        ax.view_init(elev=28, azim=45)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# FINAL 2D PLOTS (|u|, w, v) — fast overview
# ==============================================================================
def plot_final_uvwv_2d(res, zoom=20.0, cmap_u="viridis", cmap_wv="coolwarm"):
    """Plot the final fields |u|, w, and v as 2D heatmaps."""
    x, y = res["x"], res["y"]
    uabs = np.abs(res["u_final"])
    w = res["w_final"]
    v = res["v_final"]

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Lx = np.ptp(x) + dx
    Ly = np.ptp(y) + dy
    extent = (-Lx / 2, Lx / 2, -Ly / 2, Ly / 2)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    im0 = axs[0].imshow(uabs, origin="lower", extent=extent, cmap=cmap_u)
    axs[0].set_title(r"$|u|$")
    axs[0].set_xlim(-zoom, zoom)
    axs[0].set_ylim(-zoom, zoom)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(w, origin="lower", extent=extent, cmap=cmap_wv)
    axs[1].set_title(r"$w$")
    axs[1].set_xlim(-zoom, zoom)
    axs[1].set_ylim(-zoom, zoom)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(v, origin="lower", extent=extent, cmap=cmap_wv)
    axs[2].set_title(r"$v$")
    axs[2].set_xlim(-zoom, zoom)
    axs[2].set_ylim(-zoom, zoom)
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.show()


# ==============================================================================
# QUICK CONSERVATION SUMMARY (for paper text / console)
# ==============================================================================
def print_conservation_summary(res, label=""):
    """Print a compact conservation summary to the console."""
    Merr, Mdr = drift_metrics(res["M"], use_relative=True)
    Eerr, Edr = drift_metrics(res["E"], use_relative=True)
    Jx_err, Jx_dr = drift_metrics(res["Jx"], use_relative=False)
    Jy_err, Jy_dr = drift_metrics(res["Jy"], use_relative=False)

    print("\n" + "-" * 72)
    if label:
        print(f"Conservation summary — {label}")
    else:
        print("Conservation summary")
    print(f"  M   : err_final={Merr:.3e} | drift_max={Mdr:.3e} (rel.)")
    print(f"  E   : err_final={Eerr:.3e} | drift_max={Edr:.3e} (rel.)")
    print(f"  Jx  : err_final={Jx_err:.3e} | drift_max={Jx_dr:.3e} (abs.)")
    print(f"  Jy  : err_final={Jy_err:.3e} | drift_max={Jy_dr:.3e} (abs.)")
    print(f"  max|u| initial/final: {res['umax'][0]:.4e} -> {res['umax'][-1]:.4e}")
    print("-" * 72 + "\n")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # -----------------------------
    # GDSS parameters (adjustable)
    # -----------------------------
    params = {
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "xi": 1.0,
        "psi": 2.0,
        "eta": 2.0,
        "phi": 2.0,
        "chi": 1.0,
        "theta": 1.0,
        "coupling_sign": +1.0,  # If your formulation uses -xi(wx+vy), change this to -1.0.
    }

    # -----------------------------
    # Base configuration
    # -----------------------------
    base_config = dict(
        Nx=256,
        Ny=256,
        Lx=80.0,
        Ly=80.0,
        Tmax=2.0,
        dealias_on=True,
        monitor_points=80,
        store_snapshots=False,
        nframes=180,
    )

    # -----------------------------
    # Initial-condition library
    # -----------------------------
    ic_library = get_ic_library(base_config)

    # Choose the IC here:
    # "gaussian", "sech_stripe", "sech_lump", "ring", "vortex_m1",
    # "two_gaussians_collision", "plane_wave_modulated", "supergauss_stripe"
    IC_NAME = "ring"

    # If you want to override specific parameters of the chosen IC
    # without modifying the library, use a dictionary here.
    # Otherwise, leave it as None.
    IC_OVERRIDE = None
    # Example:
    # IC_OVERRIDE = dict(A=1.15, kx0=1.0, sigma_y=base_config["Ly"] / 2)

    # Show available initial conditions.
    print("\nAvailable ICs:")
    for k in ic_library.keys():
        print(f"  - {k}")

    # -----------------------------
    # dt sweep (temporal benchmark)
    # -----------------------------
    dt_list = [5e-3, 2.5e-3, 1.25e-3]

    print("\n" + "=" * 95)
    print("GDSS FFT+Strang — MASTER SCRIPT (benchmark + IC selector)")
    print(
        f"Grid: {base_config['Nx']}x{base_config['Ny']} | "
        f"Domain: [-{base_config['Lx'] / 2:.1f},{base_config['Lx'] / 2:.1f}] x "
        f"[-{base_config['Ly'] / 2:.1f},{base_config['Ly'] / 2:.1f}]"
    )
    print(f"Tmax = {base_config['Tmax']}")
    print(f"Selected IC = {IC_NAME}")
    print(f"coupling_sign = {params['coupling_sign']:+.0f}")
    print("=" * 95)

    results = dt_sweep_2d(
        dt_list,
        base_config,
        params,
        ic_name=IC_NAME,
        ic_library=ic_library,
        ic_override_kwargs=IC_OVERRIDE,
        compare_self_error=True,
        align_phase=True,
    )

    # Finest time step.
    best_dt = dt_list[-1]

    # -----------------------------
    # Final run with snapshots
    # -----------------------------
    cfg_final = base_config.copy()
    cfg_final["store_snapshots"] = True
    cfg_final["nframes"] = 160  # 120–180 usually provides a good speed/quality balance.

    print("Running final simulation with snapshots...")
    res = run_simulation_2d(
        dt=best_dt,
        params=params,
        ic_name=IC_NAME,
        ic_library=ic_library,
        ic_override_kwargs=IC_OVERRIDE,
        **cfg_final,
    )

    print(f"IC used (kwargs): {res['ic_kwargs_used']}")
    print_conservation_summary(res, label=f"IC={IC_NAME}, dt={best_dt:g}")

    # -----------------------------
    # Visualization switches
    # -----------------------------
    PLOT_INVARIANTS = True
    ANIMATE_2D_FIELD = True
    ANIMATE_2D_LINECUT = True
    PLOT_XT_HEATMAP = True
    PLOT_LINE_PROFILES = True

    PLOT_FINAL_2D_UWV = True  # Fast overview.
    PLOT_FINAL_3D_UWV = True  # Heavier, but useful for figures.

    # -----------------------------
    # Visualizations
    # -----------------------------
    if PLOT_INVARIANTS:
        plot_invariants_2d(res)

    if ANIMATE_2D_FIELD:
        animate_snaps_2d(
            res,
            zoom=25.0,
            interval_ms=35,
            cmap="viridis",
            save_gif=False,
        )

    if ANIMATE_2D_LINECUT:
        animate_linecut_2d(
            res,
            y_slice=0.0,
            xlim=(-20, 20),
            quantity="abs",  # "abs" or "abs2"
            interval_ms=60,
        )

    if PLOT_XT_HEATMAP:
        plot_xt_heatmap_2d(
            res,
            y_slice=0.0,
            xlim=(-20, 20),
            t_centered=False,
            quantity="abs2",  # "abs" or "abs2"
            cmap="viridis",
            t_stride=1,
            x_stride=1,
        )

    if PLOT_LINE_PROFILES:
        plot_line_profiles_2d(
            res,
            y_slice=0.0,
            xlim=(-20, 20),
            times=(0.0, 2.0, 4.0, 6.0, 8.0),
            quantity="abs",
        )

    if PLOT_FINAL_2D_UWV:
        plot_final_uvwv_2d(
            res,
            zoom=20.0,
            cmap_u="viridis",
            cmap_wv="coolwarm",
        )

    if PLOT_FINAL_3D_UWV:
        plot_final_uvwv_3d(
            res,
            zoom=20.0,
            cmap_u="viridis",
            cmap_wv="plasma",
        )
