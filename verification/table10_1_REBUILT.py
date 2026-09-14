"""
REBUILT Table 10.1 -- Ablation of feature tensors and physics-loss weighting (lambda)
Uses the SAME architecture family as the main verification experiment (Ybus-weighted
2-layer GCN), not a separate plain MLP, and the corrected loss/sampling from the
previous fix pass. This keeps the ablation consistent with the paper's headline result
instead of testing a different toy model.
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
G = Ybus.real
B = Ybus.imag
Gt = torch.tensor(G, dtype=torch.float32)
Bt = torch.tensor(B, dtype=torch.float32)

Yabs = np.sqrt(G**2 + B**2)
np.fill_diagonal(Yabs, 0.0)
deg = Yabs.sum(axis=1, keepdims=True)
deg[deg == 0] = 1.0
A_norm = torch.tensor(Yabs / deg, dtype=torch.float32)

N_BUS = 9
ZERO_INJECTION_BUSES = [3, 5, 7]  # Bus 4, 6, 8 (1-indexed) -- confirmed P=Q=0 exactly


def generate_dataset(n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    X_pq, X_pqi, Y = [], [], []
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
        i_avg = np.sqrt(p**2 + q**2) / (v + 1e-6)
        X_pq.append(np.stack([p, q], axis=-1))
        X_pqi.append(np.stack([p, q, i_avg], axis=-1))
        Y.append(np.stack([v, va], axis=-1))
    return (torch.tensor(np.array(X_pq), dtype=torch.float32),
            torch.tensor(np.array(X_pqi), dtype=torch.float32),
            torch.tensor(np.array(Y), dtype=torch.float32))


X_pq, X_pqi, Y = generate_dataset(n_samples=500, seed=0)
print(f"Converged samples: {len(X_pq)} / 500")
n_train = int(0.8 * len(X_pq))


class YbusGCN(nn.Module):
    """Same architecture family as the main verification experiment: 2-layer GCN,
    aggregation weighted by normalized |Ybus|, ReLU between layers."""
    def __init__(self, in_dim, hidden=16, out_dim=2):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden)
        self.W2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = torch.einsum('ij,bjf->bif', A_norm, x)
        h = torch.relu(self.W1(h))
        h = torch.einsum('ij,bjf->bif', A_norm, h)
        return self.W2(h)


def kcl_residual(v_pred, angle_pred, P_spec, Q_spec):
    vi, vj = v_pred.unsqueeze(2), v_pred.unsqueeze(1)
    dtheta = angle_pred.unsqueeze(2) - angle_pred.unsqueeze(1)
    Pcalc = (vi * vj * (Gt * torch.cos(dtheta) + Bt * torch.sin(dtheta))).sum(dim=2)
    Qcalc = (vi * vj * (Gt * torch.sin(dtheta) - Bt * torch.cos(dtheta))).sum(dim=2)
    return ((P_spec - Pcalc) ** 2 + (Q_spec - Qcalc) ** 2).mean()


def train_and_eval(X, Y, lam, seeds=5, hidden=16, epochs=300, lr=0.02):
    v_maes, a_maes = [], []
    Xtr, Xte = X[:n_train], X[n_train:]
    Ytr, Yte = Y[:n_train], Y[n_train:]
    P_spec_tr, Q_spec_tr = Xtr[:, :, 0], Xtr[:, :, 1]

    for seed in range(seeds):
        set_seed(seed)
        model = YbusGCN(in_dim=X.shape[-1], hidden=hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(Xtr)
            v_pred, a_pred = pred[:, :, 0], pred[:, :, 1]
            loss = ((v_pred - Ytr[:, :, 0])**2).mean() + ((a_pred - Ytr[:, :, 1])**2).mean()
            if lam > 0:
                loss = loss + lam * kcl_residual(v_pred, a_pred, P_spec_tr, Q_spec_tr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte)
            v_maes.append((pred[:, :, 0] - Yte[:, :, 0]).abs().mean().item())
            a_maes.append((pred[:, :, 1] - Yte[:, :, 1]).abs().mean().item())
    return float(np.mean(v_maes)), float(np.std(v_maes)), float(np.mean(a_maes)), float(np.std(a_maes))


configs = [
    ("Config A: [P,Q], lambda=0.0 (baseline)",   X_pq,  0.0),
    ("Config A: [P,Q], lambda=1.0 (PI-GNN)",     X_pq,  1.0),
    ("Config B: [P,Q,Iavg], lambda=0.0",         X_pqi, 0.0),
    ("Config B: [P,Q,Iavg], lambda=1.0",         X_pqi, 1.0),
    ("Config C: [P,Q], lambda=0.1",              X_pq,  0.1),
    ("Config D: [P,Q,Iavg], lambda=0.1",         X_pqi, 0.1),
]

results = {}
print("\n--- REBUILT TABLE 10.1 (Ybus-weighted GCN, real KCL loss, non-cumulative sampling) ---")
for name, X, lam in configs:
    v_mean, v_std, a_mean, a_std = train_and_eval(X, Y, lam)
    results[name] = {"voltage_mae_mean": v_mean, "voltage_mae_std": v_std,
                      "angle_mae_mean": a_mean, "angle_mae_std": a_std}
    print(f"{name:<42} | V-MAE: {v_mean:.5f}+/-{v_std:.5f} | Angle-MAE: {a_mean:.5f}+/-{a_std:.5f} rad")

with open('table10_1_rebuilt.json', 'w') as f:
    json.dump(results, f, indent=2)
