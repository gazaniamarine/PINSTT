"""
Case Study 1: Omnidirectional Robot (2D)

Reproduces the first experiment from arXiv:2512.08248 Section V-A.

Setup:
  - State: [x, y] position in a 2D workspace [0,7] x [0,7]
  - Start:  S = B([1.5, 1.5], 0.25)
  - Target: T = B([5.5, 5.5], 0.25)
  - Prescribed time: t_c = 10 s
  - 4 circular obstacles between start and target
  - Dynamics treated as single integrator (direct velocity control)
  - External disturbance w(t) applied to show robustness

What we expect to see (matching paper):
  - PINN learns a time-varying tube that curves around obstacles
  - Formal verification passes (Theorem 3.4)
  - Controller keeps trajectory inside tube despite disturbances
  - Offline STT computation ~ a few seconds; online control ~ milliseconds
"""

import sys
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # headless backend -- change to 'TkAgg' if you want live windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# allow running directly from experiments/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.loss import TRASSpec
from src.trainer import train
from src.verifier import verify
from src.controller import simulate_single_integrator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')


# ---------------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------------

def make_spec() -> TRASSpec:
    dev = DEVICE
    return TRASSpec(
        c_S=torch.tensor([1.5, 1.5], dtype=torch.float32, device=dev),
        r_S=0.25,
        c_T=torch.tensor([5.5, 5.5], dtype=torch.float32, device=dev),
        r_T=0.25,
        # Workspace [0,7]x[0,7] approximated as inscribed ball
        c_Y=torch.tensor([3.5, 3.5], dtype=torch.float32, device=dev),
        r_Y=3.5,
        # Obstacles placed between start and target
        obstacles=[
            (torch.tensor([2.5, 3.5], dtype=torch.float32, device=dev), 0.55),
            (torch.tensor([4.0, 2.5], dtype=torch.float32, device=dev), 0.50),
            (torch.tensor([3.5, 4.8], dtype=torch.float32, device=dev), 0.45),
            (torch.tensor([5.0, 3.5], dtype=torch.float32, device=dev), 0.45),
        ],
        r_min=0.10,
        t_c=10.0,
        L_c=2.0,
        L_r=0.5,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_spatial(ax, pinn, spec, traj_xy=None):
    ax.set_xlim(-0.2, 7.2)
    ax.set_ylim(-0.2, 7.2)
    ax.set_aspect('equal')
    ax.set_title('Spatiotemporal Tube & Trajectory (2D Robot)', fontsize=11)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    # Workspace boundary
    ws = plt.Circle(spec.c_Y.cpu().numpy(), spec.r_Y,
                    fill=False, edgecolor='dimgray', lw=1.5, ls='--')
    ax.add_patch(ws)

    # Obstacles
    for i, (oc, or_) in enumerate(spec.obstacles):
        patch = plt.Circle(oc.cpu().numpy(), or_,
                           color='tomato', alpha=0.7, zorder=3,
                           label='Obstacle' if i == 0 else '')
        ax.add_patch(patch)

    # Tube snapshots at 20 time instants (blue gradient)
    t_snap = torch.linspace(0, spec.t_c, 20, device=DEVICE).unsqueeze(1)
    with torch.no_grad():
        c_s, r_s = pinn(t_snap)
    c_s = c_s.cpu().numpy()
    r_s = r_s.cpu().numpy().flatten()

    for i in range(len(t_snap)):
        alpha = 0.08 + 0.25 * (i / len(t_snap))
        tube = plt.Circle(c_s[i], r_s[i], color='steelblue',
                          alpha=alpha, zorder=2)
        ax.add_patch(tube)

    # Tube centre path
    ax.plot(c_s[:, 0], c_s[:, 1], 'b-', lw=2, label='Tube centre', zorder=4)

    # Start / target
    start = plt.Circle(spec.c_S.cpu().numpy(), spec.r_S,
                        color='limegreen', alpha=0.6, zorder=5, label='Start S')
    target = plt.Circle(spec.c_T.cpu().numpy(), spec.r_T,
                         color='gold', alpha=0.8, zorder=5, label='Target T')
    ax.add_patch(start)
    ax.add_patch(target)

    # Trajectory
    if traj_xy is not None:
        ax.plot(traj_xy[0], traj_xy[1], 'k-', lw=1.5, label='Trajectory', zorder=6)
        ax.plot(traj_xy[0, 0], traj_xy[1, 0], 'go', ms=8, zorder=7)
        ax.plot(traj_xy[0, -1], traj_xy[1, -1], 'r*', ms=10, zorder=7)

    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_time(ax, pinn, spec, traj_xy=None, t_traj=None):
    t_np = np.linspace(0, spec.t_c, 300)
    t_t  = torch.tensor(t_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    with torch.no_grad():
        c_v, r_v = pinn(t_t)
    c_v = c_v.cpu().numpy()
    r_v = r_v.cpu().numpy().flatten()

    ax.plot(t_np, r_v, 'b-', lw=2, label='Tube radius r(t)')

    if traj_xy is not None and t_traj is not None:
        errors = []
        for i, ti in enumerate(t_traj):
            idx = np.argmin(np.abs(t_np - ti))
            e = np.linalg.norm(traj_xy[:, i] - c_v[idx]) / (r_v[idx] + 1e-9)
            errors.append(e)
        ax.plot(t_traj, errors, 'k--', lw=1.5, label='Normalised error e₁(t)')
        ax.axhline(1.0, color='r', ls=':', lw=1.5, label='Safety boundary (e₁=1)')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Value')
    ax.set_title('Tube Radius and Trajectory Error vs Time', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_loss(ax, history):
    ax.semilogy(history['total'], label='Total', lw=2)
    ax.semilogy(history['L_p3'], label='Obstacle (L_p3)', lw=1.2)
    ax.semilogy(history['L_bc'], label='Boundary (L_bc)', lw=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss History', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("\n" + "=" * 60)
    print("  Case Study 1: Omnidirectional Robot (2D)")
    print("=" * 60)

    spec = make_spec()

    # ---- 1. Train PINSTT ----
    print("\n[1/4] Training PINSTT ...")
    t0 = time.time()
    pinn, history, meta = train(
        spec=spec,
        state_dim=2,
        hidden_dims=[128, 128, 128, 128],
        n_colloc=500,
        lr=1e-3,
        max_epochs=15000,
        convergence_tol=1e-4,
        warm_start=True,
        warm_start_epochs=2000,
        device=DEVICE,
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"  Offline STT computation: {train_time:.3f} s")

    # ---- 2. Formal verification ----
    print("\n[2/4] Running formal verification ...")
    ver = verify(pinn, spec, meta, n_check=1000, tol=1e-2, verbose=True)

    # ---- 3. Controller simulation ----
    print("\n[3/4] Simulating closed-loop trajectory ...")
    def disturbance(t, x):
        return 0.05 * np.array([np.sin(2.0 * t), np.cos(3.0 * t)])

    t1 = time.time()
    x0 = spec.c_S.cpu().numpy()
    sol = simulate_single_integrator(
        pinn, x0, spec.t_c,
        kappa=8.0, n_steps=1000,
        disturbance_fn=disturbance,
    )
    ctrl_time = time.time() - t1
    print(f"  Online control (total sim): {ctrl_time:.3f} s")
    print(f"  Per-step online time (approx): {ctrl_time/1000*1000:.3f} ms/step")
    if not sol.success:
        print(f"  WARNING: ODE solver did not succeed: {sol.message}")

    traj = sol.y            # (2, n_steps)
    t_traj = sol.t          # (n_steps,)

    # ---- 4. Plot ----
    print("\n[4/4] Saving plots ...")
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_spatial(axes[0], pinn, spec, traj_xy=traj)
    plot_time(axes[1], pinn, spec, traj_xy=traj, t_traj=t_traj)
    plot_loss(axes[2], history)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'robot_2d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {path}")

    # Summary
    print("\n--- Summary ---")
    print(f"  Offline (PINSTT training): {train_time:.2f} s")
    print(f"  Formal verification: PASSED" if ver['all_pass'] else "  Formal verification: (some checks failed -- see above)")
    print(f"  ODE sim success: {sol.success}")
    print(f"  Final position:  {traj[:, -1]}")
    print(f"  Target centre:   {spec.c_T.cpu().numpy()}")
    print(f"  Distance to target: {np.linalg.norm(traj[:, -1] - spec.c_T.cpu().numpy()):.4f} m")

    return pinn, history, meta, ver, sol


if __name__ == '__main__':
    run()
