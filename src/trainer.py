"""
Training algorithm for PINSTT (Algorithm 1 from the paper).

Two-phase approach:
  Phase 1 (warm-start): pre-train PINN to output a linear interpolation from
           c_S -> c_T and r_S -> r_T so the optimizer starts near a feasible
           solution rather than at a random initialization.
  Phase 2 (physics training): minimise the full T-RAS loss.

The Lipschitz-based validity parameter eta is set as:
    eta = -L * eps
where eps = t_c / (2*M) (half-spacing between M uniform collocation points)
and L = max(L_c, L_r).

With this choice, if all sampled-point losses are driven to zero, Theorem 3.4
guarantees the constraints hold over the ENTIRE continuous interval [0, t_c].
"""

import time
from typing import Optional

import torch
import torch.optim as optim
from tqdm import tqdm

from .pinn import PINSTT
from .loss import TRASSpec, compute_loss


# ---------------------------------------------------------------------------
# Default training weights
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    'w_p1': 10.0,   # workspace containment
    'w_p2': 10.0,   # min radius
    'w_p3': 15.0,   # obstacle avoidance (higher: safety critical)
    'w_p4':  5.0,   # Lipschitz on center
    'w_p5':  5.0,   # Lipschitz on radius
    'w_b1': 200.0,  # start center
    'w_b2': 200.0,  # start radius
    'w_b3': 200.0,  # target center
    'w_b4': 200.0,  # target radius
}


# ---------------------------------------------------------------------------
# Warm-start: fit a linear tube from S to T
# ---------------------------------------------------------------------------

def _warm_start(pinn: PINSTT, spec: TRASSpec, n_epochs: int = 2000,
                lr: float = 1e-3, device: str = 'cpu', verbose: bool = True):
    """Pre-train PINN to follow a linear interpolation from S to T."""
    t_vals = torch.linspace(0, spec.t_c, 300, device=device).unsqueeze(1)
    alpha = t_vals / spec.t_c                         # (300, 1) in [0,1]
    c_target = spec.c_S + alpha * (spec.c_T - spec.c_S)  # (300, state_dim)
    r_target = spec.r_S + alpha.squeeze() * (spec.r_T - spec.r_S)  # (300,)
    r_target = r_target.unsqueeze(1)                  # (300, 1)

    opt = optim.Adam(pinn.parameters(), lr=lr)
    if verbose:
        print("  [warm-start] pre-training linear tube...")
    for ep in range(n_epochs):
        opt.zero_grad()
        c, r = pinn(t_vals)
        loss = torch.nn.functional.mse_loss(c, c_target) \
             + torch.nn.functional.mse_loss(r, r_target)
        loss.backward()
        opt.step()
        if verbose and ep % 500 == 0:
            print(f"    ep {ep:5d}  mse={loss.item():.5f}")
    if verbose:
        print("  [warm-start] done.")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    spec: TRASSpec,
    state_dim: int,
    hidden_dims: Optional[list] = None,
    n_colloc: int = 500,
    lr: float = 1e-3,
    max_epochs: int = 15000,
    convergence_tol: float = 1e-4,
    weights: Optional[dict] = None,
    warm_start: bool = True,
    warm_start_epochs: int = 2000,
    device: str = 'cpu',
    verbose: bool = True,
) -> tuple:
    """
    Train a PINSTT network.

    Returns:
        pinn        -- trained PINSTT
        history     -- dict of lists, one value per epoch
        meta        -- dict with eta, eps, L used for verification
    """
    if hidden_dims is None:
        hidden_dims = [128, 128, 128, 128]
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # eta (called eta_hat in paper) = -L * eps  (Remark 3.3 / Theorem 3.4)
    eps = spec.t_c / (2.0 * n_colloc)
    L   = max(spec.L_c, spec.L_r)
    eta = -L * eps

    # uniform collocation points  (the M points t_r)
    t_colloc = torch.linspace(0, spec.t_c, n_colloc, device=device).unsqueeze(1)

    pinn = PINSTT(state_dim, hidden_dims).to(device)

    # ---- Phase 1: warm-start ----
    if warm_start:
        _warm_start(pinn, spec, n_epochs=warm_start_epochs,
                    lr=lr, device=device, verbose=verbose)

    # ---- Phase 2: physics training ----
    optimizer = optim.Adam(pinn.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )

    history = {k: [] for k in ('total', 'L_p1', 'L_p2', 'L_p3', 'L_p4', 'L_p5', 'L_bc')}

    t0 = time.time()
    pbar = tqdm(range(max_epochs), disable=not verbose, desc='training')
    for epoch in pbar:
        optimizer.zero_grad()
        loss, log = compute_loss(pinn, t_colloc, spec, eta, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        for k, v in log.items():
            if k in history:
                history[k].append(v)

        if verbose and epoch % 100 == 0:
            pbar.set_postfix(
                loss=f"{log['total']:.5f}",
                obs=f"{log['L_p3']:.4f}",
                bc=f"{log['L_bc']:.4f}",
            )

        if log['total'] < convergence_tol:
            if verbose:
                # update bar to show the actual final values before stopping
                pbar.set_postfix(
                    loss=f"{log['total']:.5f}",
                    obs=f"{log['L_p3']:.4f}",
                    bc=f"{log['L_bc']:.4f}",
                )
                pbar.close()
                print(f"  Converged at epoch {epoch}/{max_epochs}  "
                      f"loss={log['total']:.2e}  "
                      f"(used {100*epoch//max_epochs}% of budget)")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"  Training time: {elapsed:.2f} s")

    meta = {'eta': eta, 'eps': eps, 'L': L, 'n_colloc': n_colloc}
    return pinn, history, meta
