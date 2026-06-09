"""
Closed-form prescribed-performance controller (Theorem 4.1).

Core idea (single-integrator case):
  Define e1 = ||x - c(t)|| / r(t)  (normalised position error, in [0,1) if inside tube)
  Define barrier: epsilon_1 = ln((1 + e1) / (1 - e1))   -> +inf as e1 -> 1
  Control: u = -kappa * epsilon_1 * (x - c(t)) / r(t) + dc/dt

  The log-barrier makes the control grow without bound as the trajectory
  approaches the tube boundary, preventing escape. This is the "approximation-
  free" controller the paper calls (Eq. 14 / Theorem 4.1).

Double-integrator (quadrotor):
  Stage 1 maps position x1 to a desired velocity r2_des (same log-barrier).
  Stage 2 drives actual velocity x2 -> r2_des using a second log-barrier
  over a time-decaying error bound gamma2(t).

Both simulators use scipy solve_ivp (RK45).
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Utility: query PINN at a single scalar time (used inside ODE rhs)
# ---------------------------------------------------------------------------

def _query_pinn(pinn, t_scalar: float):
    """Return (c_np, r_scalar, dc_np) at time t as numpy arrays."""
    device = next(pinn.parameters()).device
    t = torch.tensor([[t_scalar]], dtype=torch.float32,
                     device=device, requires_grad=True)
    c, r = pinn(t)

    grads = []
    for i in range(pinn.state_dim):
        g = torch.autograd.grad(c[0, i], t, retain_graph=True)[0]
        grads.append(g[0, 0].item())

    return (c.detach().cpu().numpy()[0],
            r.detach().cpu().numpy()[0, 0],
            np.array(grads))


def _query_pinn_ddot(pinn, t_scalar: float):
    """Return c(t), r(t), dc/dt(t), d2c/dt2(t) as numpy arrays."""
    device = next(pinn.parameters()).device
    t = torch.tensor([[t_scalar]], dtype=torch.float32,
                     device=device, requires_grad=True)
    c, r = pinn(t)

    dc, d2c = [], []
    for i in range(pinn.state_dim):
        g1 = torch.autograd.grad(c[0, i], t, create_graph=True,
                                  retain_graph=True)[0]
        g2 = torch.autograd.grad(g1[0, 0], t, retain_graph=True)[0]
        dc.append(g1[0, 0].item())
        d2c.append(g2[0, 0].item())

    return (c.detach().cpu().numpy()[0],
            r.detach().cpu().numpy()[0, 0],
            np.array(dc), np.array(d2c))


# ---------------------------------------------------------------------------
# Log-barrier function
# ---------------------------------------------------------------------------

def _log_barrier(e: float, clip: float = 0.995) -> float:
    """epsilon(e) = ln((1+e)/(1-e)),  e clamped to [0, clip] for stability."""
    e = float(np.clip(e, 0.0, clip))
    return float(np.log((1.0 + e) / (1.0 - e)))


# ---------------------------------------------------------------------------
# Single-integrator controller  (2D robot)
# ---------------------------------------------------------------------------

def _ctrl_single(x: np.ndarray, t: float, pinn, kappa: float) -> np.ndarray:
    """
    u = -kappa * epsilon1(e1) * (x - c(t)) / r(t)  +  dc/dt
    Keeps state x inside the tube B(c(t), r(t)).
    """
    c, r, dc = _query_pinn(pinn, t)
    diff = x - c
    e1 = np.linalg.norm(diff) / r
    eps1 = _log_barrier(e1)
    return -kappa * eps1 * diff / r + dc


def simulate_single_integrator(
    pinn,
    x0: np.ndarray,
    t_c: float,
    kappa: float = 8.0,
    n_steps: int = 1000,
    disturbance_fn=None,
) -> object:
    """
    Simulate  dx/dt = u(x,t) + w(t)  under the single-integrator controller.

    disturbance_fn(t, x) -> np.ndarray of same shape as x0  (optional).
    Returns a scipy OdeSolution object.
    """
    pinn.eval()

    def rhs(t, x):
        u = _ctrl_single(x, t, pinn, kappa)
        w = disturbance_fn(t, x) if disturbance_fn else np.zeros_like(x0)
        return u + w

    t_eval = np.linspace(0, t_c, n_steps)
    return solve_ivp(rhs, [0.0, t_c], x0, t_eval=t_eval,
                     method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)


# ---------------------------------------------------------------------------
# Double-integrator controller  (quadrotor)
# ---------------------------------------------------------------------------

def _ctrl_double(state: np.ndarray, t: float, pinn,
                 kappa1: float, kappa2: float, gamma2: float) -> np.ndarray:
    """
    Two-stage controller for double-integrator  x1_dot = x2,  x2_dot = u.

    Stage 1: desired x2  r2_des = -kappa1*eps1*(x1-c)/r + dc/dt
    Stage 2: u = -kappa2*eps2*(x2 - r2_des)/gamma2  + d2c/dt2
    """
    n = pinn.state_dim
    x1, x2 = state[:n], state[n:]

    c, r, dc, d2c = _query_pinn_ddot(pinn, t)
    diff1 = x1 - c
    e1    = np.linalg.norm(diff1) / r
    eps1  = _log_barrier(e1)

    r2_des = -kappa1 * eps1 * diff1 / r + dc

    diff2 = x2 - r2_des
    e2    = np.linalg.norm(diff2) / (gamma2 + 1e-6)
    eps2  = _log_barrier(e2)

    u = -kappa2 * eps2 * diff2 / (gamma2 + 1e-6) + d2c
    return u


def simulate_double_integrator(
    pinn,
    x0: np.ndarray,       # full state [position (n,), velocity (n,)]
    t_c: float,
    kappa1: float = 3.0,
    kappa2: float = 5.0,
    gamma2: float = 2.0,
    n_steps: int = 1000,
    disturbance_fn=None,
) -> object:
    """
    Simulate  x1_dot = x2 + w1,  x2_dot = u + w2.

    disturbance_fn(t, state) -> np.ndarray, shape (2*n,)  (optional).
    """
    pinn.eval()
    n = pinn.state_dim

    def rhs(t, state):
        u = _ctrl_double(state, t, pinn, kappa1, kappa2, gamma2)
        w = disturbance_fn(t, state) if disturbance_fn else np.zeros(2 * n)
        dx1 = state[n:] + w[:n]
        dx2 = u + w[n:]
        return np.concatenate([dx1, dx2])

    t_eval = np.linspace(0, t_c, n_steps)
    return solve_ivp(rhs, [0.0, t_c], x0, t_eval=t_eval,
                     method='RK45', rtol=1e-6, atol=1e-8, max_step=0.02)
