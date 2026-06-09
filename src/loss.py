"""
Loss functions for PINSTT training.

Paper equations referenced:
  Eq. 8a  -> L_p1  workspace containment
  Eq. 8b  -> L_p2  minimum radius
  Eq. 8c  -> L_p3  obstacle avoidance
  Eq. 9a  -> L_p4  Lipschitz bound on center derivative
  Eq. 9b  -> L_p5  Lipschitz bound on radius derivative
  Eq. 10  -> L_phys = weighted sum of L_p1 .. L_p5
  Eq. 11  -> L_bc  boundary conditions at t=0 and t=t_c
  Eq. 7   -> L = L_phys + L_bc
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TRASSpec:
    """
    Temporal Reach-Avoid-Stay specification.

    All tensors should live on the same device as the PINN.

    The STT Gamma(t) = B(c(t), r(t)) must satisfy:
      1. Gamma(0)   subset S = B(c_S, r_S)           (start)
      2. Gamma(t_c) subset T = B(c_T, r_T)           (reach target by t_c)
      3. Gamma(t)   subset Y = B(c_Y, r_Y)  for all t  (stay in workspace)
      4. Gamma(t) disjoint from each obstacle          (avoid)
      5. r(t) >= r_min                                 (non-degenerate tube)
    """
    c_S: torch.Tensor          # initial set center, shape (state_dim,)
    r_S: float                 # initial set radius
    c_T: torch.Tensor          # target set center, shape (state_dim,)
    r_T: float                 # target set radius
    c_Y: torch.Tensor          # workspace ball center
    r_Y: float                 # workspace ball radius
    obstacles: List[Tuple[torch.Tensor, float]] = field(default_factory=list)
    r_min: float = 0.05        # minimum tube radius
    t_c: float = 10.0          # prescribed final time
    L_c: float = 2.0           # Lipschitz bound for ||dc/dt||
    L_r: float = 0.5           # Lipschitz bound for |dr/dt|


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

def _relu_mean(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).mean()


def loss_workspace(c, r, spec: TRASSpec, eta: float) -> torch.Tensor:
    """
    L_p1 (Eq. 8a): tube must stay inside workspace ball B(c_Y, r_Y).

    Constraint at sample point:  ||c(t_r) - c_Y|| + r(t_r)  <=  r_Y + eta
    (eta < 0, so this is tighter than the nominal constraint)
    """
    dist = torch.norm(c - spec.c_Y, dim=1, keepdim=True)
    return _relu_mean(dist + r - spec.r_Y - eta)


def loss_min_radius(r, spec: TRASSpec, eta: float) -> torch.Tensor:
    """
    L_p2 (Eq. 8b): radius must stay above r_min.

    Constraint:  r(t_r) >= r_min + eta   <=>  -r(t_r) + r_min - eta <= 0
    (eta < 0 so r_min - eta = r_min + |eta|, enforcing tighter lower bound)
    """
    return _relu_mean(-r + spec.r_min - eta)


def loss_obstacle(c, r, spec: TRASSpec, eta: float) -> torch.Tensor:
    """
    L_p3 (Eq. 8c): tube B(c,r) must not intersect any obstacle B(o_i, rho_i).

    Non-intersection condition:  ||c - o_i|| - rho_i  >=  r
    With Lipschitz margin:       r  <=  (||c - o_i|| - rho_i) + eta
    Loss:  ReLU( r - (||c - o_i|| - rho_i) - eta )
    """
    total = torch.zeros(c.shape[0], 1, device=c.device)
    for obs_c, obs_r in spec.obstacles:
        d_surf = torch.norm(c - obs_c, dim=1, keepdim=True) - obs_r
        total = total + F.relu(r - d_surf - eta)
    return total.mean()


def loss_lipschitz_c(dc_dt, spec: TRASSpec) -> torch.Tensor:
    """L_p4 (Eq. 9a): ||dc/dt|| <= L_c"""
    return _relu_mean(torch.norm(dc_dt, dim=1, keepdim=True) - spec.L_c)


def loss_lipschitz_r(dr_dt, spec: TRASSpec) -> torch.Tensor:
    """L_p5 (Eq. 9b): |dr/dt| <= L_r"""
    return _relu_mean(torch.abs(dr_dt) - spec.L_r)


def loss_boundary(pinn, spec: TRASSpec, weights: dict) -> torch.Tensor:
    """
    L_bc (Eq. 11): boundary conditions at t=0 (start) and t=t_c (target).

    L_bc = w_b1*MSE(c(0), c_S) + w_b2*MSE(r(0), r_S)
         + w_b3*MSE(c(tc), c_T) + w_b4*MSE(r(tc), r_T)
    """
    device = spec.c_S.device
    t0 = torch.tensor([[0.0]], device=device)
    tc = torch.tensor([[spec.t_c]], device=device)

    c0, r0 = pinn(t0)
    cc, rc = pinn(tc)

    rS = torch.tensor([[spec.r_S]], dtype=torch.float32, device=device)
    rT = torch.tensor([[spec.r_T]], dtype=torch.float32, device=device)

    bc = (weights.get('w_b1', 1.0) * F.mse_loss(c0, spec.c_S.unsqueeze(0))
        + weights.get('w_b2', 1.0) * F.mse_loss(r0, rS)
        + weights.get('w_b3', 1.0) * F.mse_loss(cc, spec.c_T.unsqueeze(0))
        + weights.get('w_b4', 1.0) * F.mse_loss(rc, rT))
    return bc


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def compute_loss(pinn, t_colloc: torch.Tensor, spec: TRASSpec,
                 eta: float, weights: dict) -> Tuple[torch.Tensor, dict]:
    """
    Full loss  L = L_phys + L_bc  (Eq. 7).

    Returns (total_loss, dict of individual components for logging).
    """
    c, r = pinn(t_colloc)
    dc_dt, dr_dt = pinn.derivatives(t_colloc)

    lp1 = loss_workspace(c, r, spec, eta)
    lp2 = loss_min_radius(r, spec, eta)
    lp3 = loss_obstacle(c, r, spec, eta)
    lp4 = loss_lipschitz_c(dc_dt, spec)
    lp5 = loss_lipschitz_r(dr_dt, spec)

    L_phys = (weights.get('w_p1', 10.0) * lp1
            + weights.get('w_p2', 10.0) * lp2
            + weights.get('w_p3', 15.0) * lp3
            + weights.get('w_p4',  5.0) * lp4
            + weights.get('w_p5',  5.0) * lp5)

    L_bc = loss_boundary(pinn, spec, weights)

    total = L_phys + L_bc

    log = {
        'total': total.item(),
        'L_p1': lp1.item(), 'L_p2': lp2.item(), 'L_p3': lp3.item(),
        'L_p4': lp4.item(), 'L_p5': lp5.item(), 'L_bc': L_bc.item(),
    }
    return total, log
