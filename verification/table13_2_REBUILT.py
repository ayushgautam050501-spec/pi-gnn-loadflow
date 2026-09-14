"""
REBUILT Table 13.2 -- IG baseline sensitivity analysis.

Fixes vs. both the original and the first patch attempt:
  1. Runs Integrated Gradients on the ACTUAL trained GCN (baseline, lambda=0, same
     architecture as the main verification experiment) instead of a hand-written toy
     linear function with an arbitrary 4x weight on one bus.
  2. Replaces the degenerate "Static Zero-Injection" input profile (all buses P=Q=0,
     which is mathematically guaranteed to give 0% attribution under any baseline --
     see prior analysis) with the REAL zero-injection buses that exist in case9's
     actual topology: buses 4, 6, 8 (1-indexed) = indices 3, 5, 7. These buses have
     exactly P=0, Q=0 by construction (transmission-only nodes), embedded within an
     otherwise realistic, non-degenerate varying-load dataset. This ties the test
     directly to the paper's own Bus 4 discussion (Section 15.6) instead of being a
     disconnected synthetic sanity check.
  3. Reports attribution share assigned to the zero-injection buses collectively,
     under a Zero Baseline vs. a Mean-Grid (in-distribution) Baseline, so the
     "baseline choice matters" claim is tested on buses that are actually
     zero-injection in the real network -- not an artificial all-zero dataset.
"""
import torch
import torch.nn as nn
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import json

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(0)
net0 = pn.case9()
nominal_p = net0.load['p_mw'].values.copy()
nominal_q = net0.load['q_mvar'].values.copy()
pp.runpp(net0, algorithm='nr')
Ybus = net0._ppc['internal']['Ybus'].toarray()
G, B = Ybus.real, Ybus.imag
Gt = torch.tensor(G, dtype=torch.float32)
Bt = torch.tensor(B, dtype=torch.float32)

Yabs = np.sqrt(G**2 + B**2)
np.fill_diagonal(Yabs, 0.0)
deg = Yabs.sum(axis=1, keepdims=True)
deg[deg == 0] = 1.0
A_norm = torch.tensor(Yabs / deg, dtype=torch.float32)

ZERO_INJECTION_BUSES = [3, 5, 7]  # Bus 4, 6, 8 (1-indexed) -- confirmed P=Q=0 exactly


def generate_dataset(n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_samples):
        net = pn.case9()
        scale = rng.uniform(0.7, 1.3)
        net.load['p_mw'] = nominal_p * scale
        net.load['q_mvar'] = nominal_q * scale
        try:
            pp.runpp(net, algorithm='nr')
        except Exception:
            continue
        p = net.res_bus['p_mw'].values / 100.0
        q = net.res_bus['q_mvar'].values / 100.0
        v = net.res_bus['vm_pu'].values
        va = np.radians(net.res_bus['va_degree'].values)
        X.append(np.stack([p, q], axis=-1))
        Y.append(np.stack([v, va], axis=-1))
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)


X, Y = generate_dataset(500, seed=0)
n_train = int(0.8 * len(X))
Xtr, Ytr = X[:n_train], Y[:n_train]
print(f"Converged samples: {len(X)} / 500. Verifying zero-injection buses stay at 0 across dataset:")
for b in ZERO_INJECTION_BUSES:
    print(f"  Bus index {b}: max |P|={X[:, b, 0].abs().max().item():.2e}, max |Q|={X[:, b, 1].abs().max().item():.2e}")


class YbusGCN(nn.Module):
    def __init__(self, in_dim=2, hidden=16, out_dim=2):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden)
        self.W2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = torch.einsum('ij,bjf->bif', A_norm, x)
        h = torch.relu(self.W1(h))
        h = torch.einsum('ij,bjf->bif', A_norm, h)
        return self.W2(h)


# Train a real baseline (lambda=0) GCN -- this is the model IG will actually explain
set_seed(0)
model = YbusGCN()
opt = torch.optim.Adam(model.parameters(), lr=0.02)
for _ in range(300):
    opt.zero_grad()
    pred = model(Xtr)
    loss = ((pred[:, :, 0] - Ytr[:, :, 0])**2).mean() + ((pred[:, :, 1] - Ytr[:, :, 1])**2).mean()
    loss.backward()
    opt.step()
model.eval()
print(f"Trained model final voltage MAE (test set): "
      f"{(model(X[n_train:])[:, :, 0] - Y[n_train:, :, 0]).abs().mean().item():.5f} p.u.")


def model_scalar_output(x):
    """Sum of predicted voltage magnitudes across all buses -- a real scalar output
    of the actual trained model, suitable for IG."""
    pred = model(x)
    return pred[:, :, 0].sum(dim=1)


def compute_ig(inputs, baseline, steps=50):
    scaled = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
    grads = []
    for x_step in scaled:
        x_step = x_step.clone().detach().requires_grad_(True)
        out = model_scalar_output(x_step)
        out.backward(torch.ones_like(out))
        grads.append(x_step.grad.data.numpy())
    avg_grads = np.mean(np.array(grads), axis=0)
    delta = (inputs - baseline).detach().numpy()
    return delta * avg_grads


def zero_injection_attribution_share(attributions, target_buses=ZERO_INJECTION_BUSES):
    total = np.sum(np.abs(attributions), axis=(1, 2))
    target = np.sum(np.abs(attributions[:, target_buses, :]), axis=(1, 2))
    valid = total > 1e-8
    return float(np.mean(target[valid] / total[valid]) * 100)


test_inputs = X[n_train:]
zero_baseline = torch.zeros_like(test_inputs)
mean_baseline = torch.mean(Xtr, dim=0, keepdim=True).expand_as(test_inputs)

ig_zero = compute_ig(test_inputs, zero_baseline)
ig_mean = compute_ig(test_inputs, mean_baseline)

share_zero = zero_injection_attribution_share(ig_zero)
share_mean = zero_injection_attribution_share(ig_mean)

print("\n--- REBUILT TABLE 13.2 (real trained GCN, real zero-injection buses 4/6/8) ---")
print(f"Attribution share on zero-injection buses, Zero Baseline    : {share_zero:.2f}%")
print(f"Attribution share on zero-injection buses, Mean-Grid Baseline: {share_mean:.2f}%")

results = {
    "zero_injection_buses_1indexed": [4, 6, 8],
    "zero_baseline_attribution_pct": share_zero,
    "mean_grid_baseline_attribution_pct": share_mean,
    "note": "Fraction of total |IG| attribution landing on the true zero-injection "
            "buses (which structurally should carry little causal weight), compared "
            "across baseline choices, on the actual trained PI-GNN-family model."
}
with open('table13_2_rebuilt.json', 'w') as f:
    json.dump(results, f, indent=2)
