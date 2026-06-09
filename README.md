# PINSTT — Physics-Informed Neural Spatiotemporal Tubes

An implementation of the paper:

> **Learning Spatiotemporal Tubes for Temporal Reach-Avoid-Stay Tasks using Physics-Informed Neural Networks**  
> Ahan Basu, Ratnangshu Das, Pushpak Jagtap  
> arXiv:2512.08248 (December 2025)

No official code accompanies the paper. This repository reproduces both case studies (2D omnidirectional robot and 3D quadrotor) from scratch.

---

## What the paper does

A robot must **reach** a target, **avoid** obstacles, and **stay** within a safe region — all within a prescribed time `t_c`. The paper calls this a *Temporal Reach-Avoid-Stay (T-RAS)* task.

Instead of planning a single path, the method learns a **Spatiotemporal Tube (STT)**: a time-varying ball

```
Γ(t) = B(c(t), r(t))   — a sphere of centre c(t) and radius r(t)
```

that sweeps from the start set `S` to the target set `T` while staying clear of all obstacles. Any trajectory that remains inside this tube automatically satisfies the T-RAS specification.

### Three-stage pipeline

```
┌──────────────────────────────────────────────────────┐
│  Stage 1: PINN Training  (offline, once)             │
│                                                      │
│  Input:  T-RAS spec (S, T, obstacles, workspace)     │
│  Output: c(t;θ), r(t;θ)  — the learned tube          │
│  How:    Minimise physics-informed loss L(θ)         │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│  Stage 2: Formal Verification  (Theorem 3.4)         │
│                                                      │
│  Check: does the trained PINN satisfy all            │
│  constraints everywhere in [0, t_c], not just        │
│  at the M training points?                           │
│  Key insight: Lipschitz continuity bridges the gap.  │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│  Stage 3: Online Controller  (Theorem 4.1)           │
│                                                      │
│  u = −κ · ln((1+e₁)/(1−e₁)) · (x−c(t))/r(t) + ċ(t) │
│  e₁ = ‖x − c(t)‖ / r(t)  ∈ [0, 1)                  │
│  Log-barrier → control grows unbounded at tube edge  │
└──────────────────────────────────────────────────────┘
```

---

## Repository structure

```
PINSTT/
│
├── src/                    Core library (import from here)
│   ├── pinn.py             PINSTT network: t → (c(t), r(t))
│   ├── loss.py             TRASSpec dataclass + all 6 loss terms
│   ├── trainer.py          Two-phase training (warm-start + physics)
│   ├── verifier.py         Theorem 3.4 formal verification
│   └── controller.py       Prescribed-performance controller + ODE sim
│
├── experiments/            Runnable case studies
│   ├── robot_2d.py         Case Study 1 — omnidirectional robot
│   └── quadrotor_3d.py     Case Study 2 — quadrotor
│
├── plots/                  Generated figures (created on first run)
├── run_all.py              Entry point
└── requirements.txt
```

---

## Installation

```bash
# Clone or copy the repo, then install dependencies
pip install torch numpy scipy matplotlib tqdm
```

Tested with Python 3.10+, PyTorch 2.7 (CPU). A CUDA-capable GPU will
significantly reduce training time.

---

## Usage

```bash
# Run both case studies
python run_all.py

# Robot only  (~45 s on CPU)
python run_all.py --robot

# Quadrotor only  (~45 s on CPU)
python run_all.py --quadrotor
```

Plots are saved to `plots/`.

### Use the library directly

```python
from src.loss import TRASSpec
from src.trainer import train
from src.verifier import verify
from src.controller import simulate_single_integrator
import torch

spec = TRASSpec(
    c_S=torch.tensor([1.5, 1.5]), r_S=0.25,   # start ball
    c_T=torch.tensor([5.5, 5.5]), r_T=0.25,   # target ball
    c_Y=torch.tensor([3.5, 3.5]), r_Y=3.5,    # workspace ball
    obstacles=[(torch.tensor([3.0, 3.0]), 0.5)],
    r_min=0.1, t_c=10.0, L_c=2.0, L_r=0.5,
)

pinn, history, meta = train(spec, state_dim=2)
results = verify(pinn, spec, meta)
sol = simulate_single_integrator(pinn, spec.c_S.numpy(), spec.t_c)
```

---

## Physics-informed loss (what makes it a PINN)

The network `f_θ : t ↦ (c(t), r(t))` is trained by minimising:

```
L(θ) = L_phys(θ) + L_bc(θ)
```

| Term | Equation | What it enforces |
|------|----------|-----------------|
| `L_p1` | `ReLU(‖c−c_Y‖ + r − r_Y − η)` | Tube inside workspace |
| `L_p2` | `ReLU(−r + r_min − η)` | Tube radius ≥ r_min |
| `L_p3` | `ReLU(r − (‖c−oᵢ‖−ρᵢ) − η)` | No tube–obstacle overlap |
| **`L_p4`** | **`ReLU(‖dc/dt‖ − Lc)`** | **Lipschitz bound on centre** |
| **`L_p5`** | **`ReLU(‖dr/dt‖ − Lr)`** | **Lipschitz bound on radius** |
| `L_bc` | MSE at `t=0` and `t=t_c` | Start and target conditions |

`L_p4` and `L_p5` are the genuinely physics-informed terms: they penalise
derivative violations computed via `torch.autograd.grad`, identical in
structure to PDE residual terms in Raissi et al. (2019). These Lipschitz
constraints are what enable formal verification — without bounded derivatives
you cannot certify behaviour between sample points.

`η = −L · ε` where `L = max(Lc, Lr)` and `ε = t_c / (2M)`. This tightened
margin means that if all sampled-point losses are zero, Theorem 3.4 guarantees
the constraints hold over the entire continuous interval `[0, t_c]`.

---

## Results

Hardware: Intel CPU, no GPU.

### Case Study 1 — Omnidirectional Robot (2D)

| Metric | This implementation | Paper (Table I) |
|--------|-------------------|-----------------|
| Offline (PINN training) | ~44 s | 7.854 s |
| Online per-step | ~2.0 ms | 0.008 s |
| Formal verification | **PASSED** | PASSED |
| Distance to target at `t_c` | 0.017 m | — |
| Max tube-error `e₁` (disturbance `‖w‖=0.30`) | 0.255 < 1.0 | — |

Timing gap vs. paper is expected (paper used GPU/faster hardware).

### Case Study 2 — Quadrotor (3D)

| Metric | This implementation | Paper (Table I) |
|--------|-------------------|-----------------|
| Offline (PINN training) | ~40 s | 3.252 s |
| Online per-step | ~5.6 ms | 0.021 s |
| Formal verification | **PASSED** | PASSED |
| Distance to target at `t_c` | 0.036 m | — |

---

## Implementation notes

### Workspace representation in 3D
The paper uses a ball `B(c_Y, r_Y)` to represent the workspace. For the 3D
quadrotor case, the workspace is `[0,10]³`. Its **inscribed** ball has radius 5
(touching faces only), but both `c_S = [1,1,1]` and `c_T = [8,8,8]` are near
the cube corners and lie **outside** the inscribed ball. Using `r_Y = 5.0`
creates a contradiction in the loss that prevents convergence. The fix is
`r_Y = 8.5`, which contains both endpoint sets and lies between the inscribed
radius (5.0) and circumscribed radius (8.66).

### Warm-start
Before physics training, the PINN is pre-trained on a linear interpolation
`c(t) = c_S + (c_T−c_S)·t/t_c`. This avoids the early-training regime where
boundary losses and physics losses fight each other, reducing total epochs
needed by ~5×.

### Convergence
Training uses `convergence_tol = 1e-4`. Both case studies converge in
10–25% of the epoch budget (the remainder is unused). This is the correct
behaviour — the budget is a ceiling, not a target.

---

## Citation

```bibtex
@article{basu2025pinstt,
  title   = {Learning Spatiotemporal Tubes for Temporal Reach-Avoid-Stay
             Tasks using Physics-Informed Neural Networks},
  author  = {Basu, Ahan and Das, Ratnangshu and Jagtap, Pushpak},
  journal = {arXiv preprint arXiv:2512.08248},
  year    = {2025}
}
```

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed
  neural networks. *Journal of Computational Physics*, 378, 686–707.
- Basu, A., Das, R., & Jagtap, P. (2025). arXiv:2512.08248.
