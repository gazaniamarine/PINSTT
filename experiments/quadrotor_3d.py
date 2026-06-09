"""
Case Study 2: Quadrotor (3D)

Reproduces the second experiment from arXiv:2512.08248 Section V-B.

Setup:
  - Position state: [x, y, z] in workspace [0,10]^3
  - Full state for simulation: [x, y, z, vx, vy, vz] (double integrator)
  - Start:  S = B([1,1,1], 0.8)
  - Target: T = B([8,8,8], 0.8)
  - Prescribed time: t_c = 10 s
  - 3 spherical obstacles along the diagonal
  - Dynamics: double integrator  x1_dot = x2,  x2_dot = u + w(t)

What we expect:
  - PINN learns a 3D tube that weaves between spherical obstacles
  - Two-stage prescribed performance controller tracks the tube
  - Final position close to target ball centre
"""

import sys
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3d projection)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.loss import TRASSpec
from src.trainer import train
from src.verifier import verify
from src.controller import simulate_double_integrator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')


# ---------------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------------

def make_spec() -> TRASSpec:
    dev = DEVICE
    return TRASSpec(
        c_S=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=dev),
        r_S=0.8,
        c_T=torch.tensor([8.0, 8.0, 8.0], dtype=torch.float32, device=dev),
        r_T=0.8,
        # Workspace [0,10]^3.
        # Inscribed ball radius = 5 (faces only), circumscribed = 5*sqrt(3) ~ 8.66.
        # c_S=[1,1,1] is 6.93 m from centre -- outside inscribed ball.
        # Must use r_Y >= ||c_S - c_Y|| + r_S = 7.73, so use 8.5 with margin.
        c_Y=torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32, device=dev),
        r_Y=8.5,
        obstacles=[
            (torch.tensor([3.0, 3.2, 2.8], dtype=torch.float32, device=dev), 0.8),
            (torch.tensor([5.0, 4.5, 5.5], dtype=torch.float32, device=dev), 0.9),
            (torch.tensor([6.8, 6.5, 7.0], dtype=torch.float32, device=dev), 0.8),
        ],
        r_min=0.15,
        t_c=10.0,
        L_c=2.5,
        L_r=0.8,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_surface(center, radius, resolution=16):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def plot_3d(ax, pinn, spec, traj_xyz=None):
    ax.set_title('3D Quadrotor Navigation', fontsize=11)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # Obstacles
    for i, (oc, or_) in enumerate(spec.obstacles):
        xs, ys, zs = _sphere_surface(oc.cpu().numpy(), or_, resolution=14)
        ax.plot_surface(xs, ys, zs, color='tomato', alpha=0.35, zorder=1)

    # Start / target spheres
    xs, ys, zs = _sphere_surface(spec.c_S.cpu().numpy(), spec.r_S)
    ax.plot_surface(xs, ys, zs, color='limegreen', alpha=0.4, zorder=2)
    xs, ys, zs = _sphere_surface(spec.c_T.cpu().numpy(), spec.r_T)
    ax.plot_surface(xs, ys, zs, color='gold', alpha=0.5, zorder=2)

    # Tube snapshots
    t_snap = torch.linspace(0, spec.t_c, 14, device=DEVICE).unsqueeze(1)
    with torch.no_grad():
        c_s, r_s = pinn(t_snap)
    c_s = c_s.cpu().numpy()
    r_s = r_s.cpu().numpy().flatten()

    for i in range(len(t_snap)):
        alpha = 0.04 + 0.12 * (i / len(t_snap))
        xs, ys, zs = _sphere_surface(c_s[i], r_s[i], resolution=10)
        ax.plot_surface(xs, ys, zs, color='steelblue', alpha=alpha, zorder=3)

    # Tube centre path
    ax.plot(c_s[:, 0], c_s[:, 1], c_s[:, 2], 'b-', lw=2, label='Tube centre', zorder=5)

    # Trajectory
    if traj_xyz is not None:
        ax.plot(traj_xyz[0], traj_xyz[1], traj_xyz[2],
                'k-', lw=1.5, label='Trajectory', zorder=6)
        ax.scatter(*traj_xyz[:, 0], c='g', s=50, zorder=7)
        ax.scatter(*traj_xyz[:, -1], c='r', marker='*', s=100, zorder=7)

    ax.legend(fontsize=8)


def plot_time(ax, pinn, spec, traj_xyz=None, t_traj=None):
    t_np = np.linspace(0, spec.t_c, 300)
    t_t  = torch.tensor(t_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    with torch.no_grad():
        c_v, r_v = pinn(t_t)
    c_v = c_v.cpu().numpy()
    r_v = r_v.cpu().numpy().flatten()

    ax.plot(t_np, r_v, 'b-', lw=2, label='Tube radius r(t)')

    if traj_xyz is not None and t_traj is not None:
        errors = []
        for i, ti in enumerate(t_traj):
            idx = np.argmin(np.abs(t_np - ti))
            e = np.linalg.norm(traj_xyz[:, i] - c_v[idx]) / (r_v[idx] + 1e-9)
            errors.append(e)
        ax.plot(t_traj, errors, 'k--', lw=1.5, label='Normalised error e₁(t)')
        ax.axhline(1.0, color='r', ls=':', lw=1.5, label='Safety boundary')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Value')
    ax.set_title('Tube Radius and Error vs Time', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_loss(ax, history):
    ax.semilogy(history['total'], label='Total', lw=2)
    ax.semilogy(history['L_p3'], label='Obstacle (L_p3)', lw=1.2)
    ax.semilogy(history['L_bc'], label='Boundary (L_bc)', lw=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss (Quadrotor)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("\n" + "=" * 60)
    print("  Case Study 2: Quadrotor (3D)")
    print("=" * 60)

    spec = make_spec()

    # ---- 1. Train ----
    print("\n[1/4] Training PINSTT ...")
    t0 = time.time()
    pinn, history, meta = train(
        spec=spec,
        state_dim=3,
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

    # ---- 2. Verify ----
    print("\n[2/4] Formal verification ...")
    ver = verify(pinn, spec, meta, n_check=1000, tol=1e-2, verbose=True)

    # ---- 3. Simulate ----
    print("\n[3/4] Simulating closed-loop trajectory ...")
    def disturbance(t, state):
        w = np.zeros(6)
        w[:3] = 0.05 * np.array([np.sin(2*t), np.cos(3*t), np.sin(t + 1)])
        return w

    t1 = time.time()
    pos0 = spec.c_S.cpu().numpy()
    vel0 = np.zeros(3)
    x0   = np.concatenate([pos0, vel0])
    sol  = simulate_double_integrator(
        pinn, x0, spec.t_c,
        kappa1=3.0, kappa2=5.0, gamma2=2.0,
        n_steps=1000, disturbance_fn=disturbance,
    )
    ctrl_time = time.time() - t1
    print(f"  Online control (total sim): {ctrl_time:.3f} s")
    if not sol.success:
        print(f"  WARNING: {sol.message}")

    traj_xyz = sol.y[:3]    # position only
    t_traj   = sol.t

    # ---- 4. Plot ----
    print("\n[4/4] Saving plots ...")
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig = plt.figure(figsize=(16, 5))
    ax3d = fig.add_subplot(131, projection='3d')
    ax_t  = fig.add_subplot(132)
    ax_l  = fig.add_subplot(133)
    plot_3d(ax3d, pinn, spec, traj_xyz=traj_xyz)
    plot_time(ax_t, pinn, spec, traj_xyz=traj_xyz, t_traj=t_traj)
    plot_loss(ax_l, history)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'quadrotor_3d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {path}")

    # Summary
    print("\n--- Summary ---")
    print(f"  Offline (PINSTT training): {train_time:.2f} s")
    print(f"  Formal verification: {'PASSED' if ver['all_pass'] else 'some checks failed -- see above'}")
    print(f"  ODE sim success: {sol.success}")
    print(f"  Final position:  {traj_xyz[:, -1]}")
    print(f"  Target centre:   {spec.c_T.cpu().numpy()}")
    print(f"  Distance to target: {np.linalg.norm(traj_xyz[:, -1] - spec.c_T.cpu().numpy()):.4f} m")

    return pinn, history, meta, ver, sol


if __name__ == '__main__':
    run()
