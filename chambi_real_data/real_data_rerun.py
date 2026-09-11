import sys
sys.path.insert(0, '/home/claude/ninebus')
sys.path.insert(0, '/home/claude/ninebus/chambi_real_data')
from powerflow9 import run_case, branch_data, bus_data, gen_data, build_ybus, N
from real_load_profile import real_loading_multipliers
import numpy as np
from scipy import stats

DATA_SEED = 42


def build_adjacency():
    A = np.eye(N)
    for (f, t, r, x, b) in branch_data:
        A[f-1, t-1] = 1
        A[t-1, f-1] = 1
    D = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
    return D @ A @ D


A_norm = build_adjacency()
Ybus_full = build_ybus(branch_data)
Bmat = Ybus_full.imag


def make_dataset_real(n_samples=800, seed=DATA_SEED):
    scales = real_loading_multipliers(n_samples, seed=seed)
    X_list, Y_list, Pnet_list = [], [], []
    n_failed = 0
    for scale in scales:
        Vm, Va, ok = run_case(float(scale))
        if not ok:
            n_failed += 1
            continue
        Pd = np.array([b[2] for b in bus_data]) * scale
        Qd = np.array([b[3] for b in bus_data]) * scale
        Pg = np.zeros(N)
        for (bus, pg, vg) in gen_data:
            Pg[bus-1] = pg
        P_inj = (Pg - Pd) / 100.0
        Q_inj = (-Qd) / 100.0
        X_list.append(np.stack([P_inj, Q_inj], axis=1))
        Y_list.append(np.stack([Vm - 1.0, Va/100.0], axis=1))
        Pnet_list.append(P_inj)
    print(f"  {n_failed} of {n_samples} real-loading samples failed to converge (dropped)")
    return np.array(X_list), np.array(Y_list), np.array(Pnet_list)


class BatchGCN:
    def __init__(self, in_dim=2, hidden=16, out_dim=2, lr=0.05, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_dim, hidden) * 0.5
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, out_dim) * 0.5
        self.b2 = np.zeros(out_dim)
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in [('W1', self.W1), ('W2', self.W2), ('b1', self.b1), ('b2', self.b2)]}
        self.v = {k: np.zeros_like(x) for k, x in self.m.items()}
        self.t = 0

    def forward(self, X):
        AX = np.einsum('ij,njk->nik', A_norm, X)
        self.X, self.AX = X, AX
        self.Z1 = AX @ self.W1 + self.b1
        self.H1 = np.maximum(self.Z1, 0)
        AH1 = np.einsum('ij,njk->nik', A_norm, self.H1)
        self.AH1 = AH1
        self.Z2 = AH1 @ self.W2 + self.b2
        return self.Z2

    def adam_step(self, grads):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = {'W1': self.W1, 'W2': self.W2, 'b1': self.b1, 'b2': self.b2}
        for k in params:
            g = grads[k]
            self.m[k] = beta1*self.m[k] + (1-beta1)*g
            self.v[k] = beta2*self.v[k] + (1-beta2)*(g**2)
            mhat = self.m[k] / (1 - beta1**self.t)
            vhat = self.v[k] / (1 - beta2**self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def backward(self, dZ2, n):
        dW2 = np.einsum('nik,nio->ko', self.AH1, dZ2) / n
        db2 = dZ2.sum(axis=(0, 1)) / n
        dAH1 = np.einsum('nio,ko->nik', dZ2, self.W2)
        dH1 = np.einsum('ij,njk->nik', A_norm.T, dAH1)
        dZ1 = dH1 * (self.Z1 > 0)
        dW1 = np.einsum('nik,nih->kh', self.AX, dZ1) / n
        db1 = dZ1.sum(axis=(0, 1)) / n
        return {'W1': dW1, 'W2': dW2, 'b1': db1, 'b2': db2}

    def train(self, X_train, Y_train, P_train, epochs=300, lam=0.0):
        n = len(X_train)
        for ep in range(epochs):
            pred = self.forward(X_train)
            reg_err = pred - Y_train
            dZ2 = 2 * reg_err / (N * 2)
            if lam > 0.0:
                theta = np.deg2rad(pred[:, :, 1] * 100.0)
                L = Bmat - np.diag(Bmat.sum(axis=1))
                P_calc = np.einsum('ij,nj->ni', L, theta)
                kcl_res = P_calc - P_train
                d_kcl_dtheta = np.einsum('ij,ni->nj', L, kcl_res) * 2 / (n * N)
                d_kcl_dva_pred = d_kcl_dtheta * (np.pi/180.0) * 100.0
                dZ2[:, :, 1] += lam * d_kcl_dva_pred
            grads = self.backward(dZ2, n)
            self.adam_step(grads)


def run_single_seed_real(seed, lam, epochs=300):
    X, Y, Pnet = make_dataset_real(800, seed=DATA_SEED)
    n = len(X)
    split = int(n * 0.8)
    X_train, Y_train, P_train = X[:split], Y[:split], Pnet[:split]
    X_test, Y_test = X[split:], Y[split:]
    model = BatchGCN(seed=seed, lr=0.05)
    model.train(X_train, Y_train, P_train, epochs=epochs, lam=lam)
    pred_test = model.forward(X_test)
    vm_err = np.abs(pred_test[:, :, 0] - Y_test[:, :, 0])
    return vm_err.mean()


if __name__ == "__main__":
    N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Real-load-derived dataset -- {N_SEEDS} seeds, baseline vs KCL-penalized (lam=1.0)")
    print("(loading multipliers drawn from the digitized Chambi TRAFO_LV2 curve, not synthetic Uniform(0.70,1.30))\n")

    baseline_maes, pignn_maes = [], []
    for s in range(N_SEEDS):
        b = run_single_seed_real(seed=s, lam=0.0)
        p = run_single_seed_real(seed=s, lam=1.0)
        baseline_maes.append(b)
        pignn_maes.append(p)
        print(f"seed {s:2d}: baseline={b:.5f}  pignn={p:.5f}")

    baseline_maes = np.array(baseline_maes)
    pignn_maes = np.array(pignn_maes)
    print(f"\nBaseline MAE: {baseline_maes.mean():.5f} +/- {baseline_maes.std():.5f}")
    print(f"PI-GNN  MAE: {pignn_maes.mean():.5f} +/- {pignn_maes.std():.5f}")
    t_stat, p_val = stats.ttest_rel(baseline_maes, pignn_maes)
    print(f"\nPaired t-test (n={N_SEEDS} seeds, REAL-load-derived data): t={t_stat:.3f}, p={p_val:.5f}")

Real Chambi 33kV load profile (TRAFO_LV2 feeder), manually digitized from a
photographed SCADA historian trend chart ("REAL POWER, Last 1440 minutes",
26-04-2026 15:41:41 to 27-04-2026 15:41:41).

IMPORTANT / HONESTY NOTE:
Treat these as an approximate real load *shape and range* (roughly 3-14.3 MW over 24h), not asprecise metering values. The two sharp dips visible late in the original chart
coincide in time with the protection-event window described in Section 11 ofthe paper and Section 10 of the dissertation; they are included here for thatreason, not because their exact depth is known precisely.

Axis calibration used (from the four labeled gridlines visible in the source
image): 14.30 MW (top), 9.13 MW, 3.95 MW, -1.22 MW (bottom), evenly spaced,
confirming a linear axis. Time axis: 24h span, per the chart's own "Last 1440
minutes" title.
"""
import numpy as np

# (hours_since_start, approx_MW) -- read by eye from the source chart
REAL_LOAD_CURVE_MW = [
    (0.0, 7.5), (1.5, 6.7), (3.0, 6.3), (4.5, 7.0), (6.0, 8.2),
    (7.5, 9.8), (9.0, 11.5), (10.5, 13.0), (12.0, 13.8), (13.5, 14.1),
    (15.0, 13.9), (16.0, 9.2), (17.0, 8.3), (18.5, 8.6),
    (19.5, 3.0),   # sharp dip -- coincides with protection-event window
    (20.0, 8.4),
    (21.0, 7.6),
    (21.5, 3.6),   # second sharp dip
    (22.5, 8.9), (24.0, 9.0),
]

# Nominal rating used to convert this feeder's absolute MW into a per-unit
# loading multiplier compatible with the synthetic IEEE 9-bus dataset's
# 70-130% convention. 9.13 MW (the chart's own mid-gridline value) is used as
# a stand-in "nominal" reference since no separate nameplate rating was found
# in the available documents.
NOMINAL_MW = 9.13


def real_loading_multipliers(n_samples=800, seed=42):
    """
    Sample loading multipliers from the REAL digitized curve's empirical
    distribution, instead of the synthetic dataset's flat Uniform(0.70, 1.30).
    Returned multipliers are clipped to [0.30, 1.60] to stay inside the
    Newton-Raphson solver's safe convergence band (see powerflow9.py).
    """
    times = np.array([t for t, _ in REAL_LOAD_CURVE_MW])
    values = np.array([v for _, v in REAL_LOAD_CURVE_MW])
    multipliers_at_points = values / NOMINAL_MW

    rng = np.random.RandomState(seed)
    # Interpolate a dense curve, then sample n_samples points from it with
    # added small noise, to approximate "many operating snapshots" from one
    # real 24h trace -- this is explicitly an approximation, not independent
    # real measurements.
    dense_t = np.linspace(0, 24, 500)
    dense_v = np.interp(dense_t, times, multipliers_at_points)
    idx = rng.randint(0, len(dense_t), size=n_samples)
    base = dense_v[idx]
    noise = rng.normal(0, 0.02, size=n_samples)
    out = np.clip(base + noise, 0.30, 1.60)
    return out


if __name__ == "__main__":
    m = real_loading_multipliers(800)
    print(f"Real-data-derived loading multipliers: n={len(m)}")
    print(f"  min={m.min():.3f}  max={m.max():.3f}  mean={m.mean():.3f}  std={m.std():.3f}")
    print(f"  (synthetic dataset used flat Uniform(0.70, 1.30) instead)")

