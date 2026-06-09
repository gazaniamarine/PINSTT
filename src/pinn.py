"""
PINSTT: Physics-Informed Neural Spatiotemporal Tube

Maps time t -> (center c(t), radius r(t)) defining the tube Gamma(t) = B(c(t), r(t)).

Paper: "Learning Spatiotemporal Tubes for Temporal Reach-Avoid-Stay Tasks
        using Physics-Informed Neural Networks", Basu et al. 2025 (arXiv:2512.08248)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PINSTT(nn.Module):
    """
    Architecture: MLP with tanh activations.
    Input:  t in [0, t_c]          shape (N, 1)
    Output: c(t) in R^n            shape (N, state_dim)   -- tube center
            r(t) in R^+            shape (N, 1)           -- tube radius
    """

    def __init__(self, state_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128]

        self.state_dim = state_dim

        layers = []
        in_dim = 1
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.center_head = nn.Linear(in_dim, state_dim)
        self.radius_head = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor):
        """
        t: (N, 1) -- time values
        returns c: (N, state_dim), r: (N, 1) with r > 0 guaranteed by softplus
        """
        h = self.backbone(t)
        c = self.center_head(h)
        r = F.softplus(self.radius_head(h)) + 1e-3
        return c, r

    def derivatives(self, t: torch.Tensor):
        """
        Compute dc/dt and dr/dt via automatic differentiation.

        Uses create_graph=True so that the derivatives themselves are
        differentiable w.r.t. theta -- required for the Lipschitz loss terms
        (L_p4, L_p5) to properly update network weights during backprop.

        t: (N, 1)
        returns dc_dt: (N, state_dim), dr_dt: (N, 1)
        """
        t_in = t.clone().requires_grad_(True)
        c, r = self(t_in)

        # dc_i/dt for each state dimension i
        grads_c = []
        for i in range(self.state_dim):
            g = torch.autograd.grad(
                c[:, i].sum(), t_in,
                create_graph=True, retain_graph=True
            )[0]  # (N, 1)
            grads_c.append(g)
        dc_dt = torch.cat(grads_c, dim=1)  # (N, state_dim)

        # dr/dt
        dr_dt = torch.autograd.grad(
            r.sum(), t_in,
            create_graph=True, retain_graph=True
        )[0]  # (N, 1)

        return dc_dt, dr_dt

    def center_dot(self, t_scalar: float, device: str = 'cpu') -> torch.Tensor:
        """Convenience: dc/dt at a single time t_scalar (numpy-friendly for ODE sim)."""
        t = torch.tensor([[t_scalar]], dtype=torch.float32,
                         device=device, requires_grad=True)
        c, _ = self(t)
        grads = []
        for i in range(self.state_dim):
            g = torch.autograd.grad(c[0, i], t, retain_graph=True)[0]
            grads.append(g[0, 0].item())
        return torch.tensor(grads)
