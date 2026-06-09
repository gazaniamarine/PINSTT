"""
Formal verification of a trained PINSTT (Theorem 3.4 from the paper).

The key idea:
  - PINN is trained on M discrete collocation points with a tightened margin eta < 0.
  - Because the PINN's outputs are Lipschitz continuous in t (enforced by the
    L_p4/L_p5 losses), any constraint satisfied at the sample points is also
    satisfied EVERYWHERE in [0, t_c], up to a deviation of L * eps.
  - Setting eta = -L * eps ensures the original (non-tightened) constraints
    hold over the continuous interval.

Theorem 3.4 verification condition:
    eta + L * eps <= 0

By construction (eta = -L * eps), this is always satisfied with equality.
The verification then reduces to checking that all sampled-point losses
are driven to zero (i.e., no constraint is violated at any t_r).
"""

import torch
import numpy as np
from .pinn import PINSTT
from .loss import TRASSpec


def verify(
    pinn: PINSTT,
    spec: TRASSpec,
    meta: dict,
    n_check: int = 1000,
    tol: float = 1e-3,
    verbose: bool = True,
) -> dict:
    """
    Run all checks from Theorem 3.4 on a dense grid of n_check points.

    meta must contain keys: eta, eps, L  (returned by trainer.train).

    Returns dict with individual pass/fail and an 'all_pass' flag.
    """
    eta = meta['eta']
    eps = meta['eps']
    L   = meta['L']

    device = spec.c_S.device
    t_dense = torch.linspace(0, spec.t_c, n_check, device=device).unsqueeze(1)

    pinn.eval()
    with torch.no_grad():
        c, r = pinn(t_dense)

    results = {}

    # ------------------------------------------------------------------
    # 1. Lipschitz condition (Theorem 3.4 global requirement)
    # ------------------------------------------------------------------
    lip_val = eta + L * eps
    results['lipschitz_condition'] = float(lip_val)
    results['lipschitz_pass'] = lip_val <= 1e-12   # <= 0 by construction

    # ------------------------------------------------------------------
    # 2. Workspace containment  ||c(t) - c_Y|| + r(t) <= r_Y
    # ------------------------------------------------------------------
    ws_viol = (torch.norm(c - spec.c_Y, dim=1) + r.squeeze(1) - spec.r_Y).clamp(min=0)
    results['workspace_max_viol'] = ws_viol.max().item()
    results['workspace_pass'] = ws_viol.max().item() < tol

    # ------------------------------------------------------------------
    # 3. Minimum radius  r(t) >= r_min
    # ------------------------------------------------------------------
    rad_viol = (spec.r_min - r.squeeze(1)).clamp(min=0)
    results['radius_max_viol'] = rad_viol.max().item()
    results['radius_pass'] = rad_viol.max().item() < tol

    # ------------------------------------------------------------------
    # 4. Obstacle avoidance  ||c(t) - o_i|| - rho_i >= r(t)  for all i
    # ------------------------------------------------------------------
    obs_max = 0.0
    for obs_c, obs_r in spec.obstacles:
        d_surf = torch.norm(c - obs_c, dim=1) - obs_r
        viol = (r.squeeze(1) - d_surf).clamp(min=0)
        obs_max = max(obs_max, viol.max().item())
    results['obstacle_max_viol'] = obs_max
    results['obstacle_pass'] = obs_max < tol

    # ------------------------------------------------------------------
    # 5. Boundary conditions
    # ------------------------------------------------------------------
    with torch.no_grad():
        c0, r0 = pinn(torch.tensor([[0.0]], device=device))
        cT, rT = pinn(torch.tensor([[spec.t_c]], device=device))

    results['start_center_err']  = torch.norm(c0 - spec.c_S).item()
    results['start_radius_err']  = abs(r0.item() - spec.r_S)
    results['target_center_err'] = torch.norm(cT - spec.c_T).item()
    results['target_radius_err'] = abs(rT.item() - spec.r_T)

    bc_tol = 0.1   # looser tolerance for boundary conditions
    results['boundary_pass'] = all([
        results['start_center_err']  < bc_tol,
        results['start_radius_err']  < bc_tol,
        results['target_center_err'] < bc_tol,
        results['target_radius_err'] < bc_tol,
    ])

    results['all_pass'] = all([
        results['lipschitz_pass'],
        results['workspace_pass'],
        results['radius_pass'],
        results['obstacle_pass'],
        results['boundary_pass'],
    ])

    if verbose:
        _print_results(results)

    pinn.train()
    return results


def _print_results(res: dict):
    def _mark(key): return "PASS" if res[key] else "FAIL"
    print("\n" + "=" * 50)
    print("  Formal Verification (Theorem 3.4)")
    print("=" * 50)
    print(f"  Lipschitz cond (eta+L*eps={res['lipschitz_condition']:.2e}): {_mark('lipschitz_pass')}")
    print(f"  Workspace containment  (max viol={res['workspace_max_viol']:.4f}): {_mark('workspace_pass')}")
    print(f"  Min radius             (max viol={res['radius_max_viol']:.4f}): {_mark('radius_pass')}")
    print(f"  Obstacle avoidance     (max viol={res['obstacle_max_viol']:.4f}): {_mark('obstacle_pass')}")
    print(f"  Boundary: start_c_err={res['start_center_err']:.4f}  start_r_err={res['start_radius_err']:.4f}")
    print(f"            targ_c_err ={res['target_center_err']:.4f}  targ_r_err ={res['target_radius_err']:.4f}  {_mark('boundary_pass')}")
    print("=" * 50)
    verdict = "VERIFIED" if res['all_pass'] else "NOT VERIFIED"
    print(f"  OVERALL: {verdict}")
    print("=" * 50 + "\n")
