import numpy as np, json

data = json.load(open("dataset.json"))
Ybus = np.array(data["Ybus_real"]) + 1j*np.array(data["Ybus_imag"])
n_bus = Ybus.shape[0]
samples = data["samples"]

X = np.array([s["p_in"] + s["q_in"] for s in samples])   # (N, 2*n_bus) node features stacked
VM = np.array([s["vm"] for s in samples])                 # (N, n_bus)
VA = np.array([s["va"] for s in samples])                 # (N, n_bus) degrees

# Reshape per-node features: (N, n_bus, 2) -> [P_i, Q_i]
SN_MVA = 100.0  # pandapower default system base
P = np.array([s["p_in"] for s in samples]) / SN_MVA   # convert MW -> per-unit
Q = np.array([s["q_in"] for s in samples]) / SN_MVA   # convert MVAr -> per-unit
Xnode = np.stack([P, Q], axis=-1)  # (N, n_bus, 2), now in p.u. consistent with Ybus
Ytarget = np.stack([VM, np.deg2rad(VA)], axis=-1)  # (N, n_bus, 2) [vm_pu, va_rad]

# normalize inputs
Xmean, Xstd = Xnode.mean(axis=(0,1), keepdims=True), Xnode.std(axis=(0,1), keepdims=True) + 1e-8
Xn = (Xnode - Xmean) / Xstd

N = Xn.shape[0]
idx = np.arange(N)
rng0 = np.random.default_rng(42)
rng0.shuffle(idx)
n_train = int(0.8*N)
train_idx, test_idx = idx[:n_train], idx[n_train:]

# Adjacency weighting from |Ybus| (off-diagonal), row-normalized, plus self-loop (GCN-style)
A = np.abs(Ybus.copy())
np.fill_diagonal(A, 0.0)
A = A + np.eye(n_bus) * A.max()  # self-loop weight
D = A.sum(axis=1, keepdims=True)
A_norm = A / D  # (n_bus, n_bus)

def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(x.dtype)

class TwoLayerGCN:
    def __init__(self, in_dim, hid_dim, out_dim, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0/in_dim), size=(in_dim, hid_dim))
        self.b1 = np.zeros(hid_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0/hid_dim), size=(hid_dim, out_dim))
        self.b2 = np.zeros(out_dim)

    def forward(self, Xb):
        # Xb: (B, n_bus, in_dim)
        Z1 = np.einsum('ij,bjf->bif', A_norm, Xb) @ self.W1 + self.b1  # (B, n_bus, hid)
        H1 = relu(Z1)
        Z2 = np.einsum('ij,bjf->bif', A_norm, H1) @ self.W2 + self.b2  # (B, n_bus, out)
        cache = (Xb, Z1, H1, Z2)
        return Z2, cache

    def _adam_init(self):
        self.m = {k: np.zeros_like(v) for k, v in [('W1',self.W1),('b1',self.b1),('W2',self.W2),('b2',self.b2)]}
        self.v = {k: np.zeros_like(v) for k, v in [('W1',self.W1),('b1',self.b1),('W2',self.W2),('b2',self.b2)]}
        self.t = 0

    def backward(self, cache, dOut, lr, clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8):
        if not hasattr(self, 'm'):
            self._adam_init()
        Xb, Z1, H1, Z2 = cache
        B = Xb.shape[0]
        dOut = np.clip(dOut, -clip, clip)
        dW2 = np.einsum('bif,bio->fo', np.einsum('ij,bjf->bif', A_norm, H1), dOut) / B
        db2 = dOut.mean(axis=(0,1))
        dH1_agg = dOut @ self.W2.T
        dH1 = np.einsum('ij,bif->bjf', A_norm, dH1_agg)
        dZ1 = np.clip(dH1 * relu_grad(Z1), -clip, clip)
        dW1 = np.einsum('bif,bio->fo', np.einsum('ij,bjf->bif', A_norm, Xb), dZ1) / B
        db1 = dZ1.mean(axis=(0,1))

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        self.t += 1
        for k, g in grads.items():
            g = np.clip(g, -clip, clip)
            self.m[k] = beta1*self.m[k] + (1-beta1)*g
            self.v[k] = beta2*self.v[k] + (1-beta2)*(g*g)
            mhat = self.m[k] / (1 - beta1**self.t)
            vhat = self.v[k] / (1 - beta2**self.t)
            update = lr * mhat / (np.sqrt(vhat) + eps)
            setattr(self, k, getattr(self, k) - update)


def power_balance_residual(vm, va_rad):
    # vm, va_rad: (B, n_bus)
    B = vm.shape[0]
    Vc = vm * np.exp(1j*va_rad)  # complex voltage (B, n_bus)
    I = Vc @ Ybus.T  # (B, n_bus) injected current per bus (S = V * conj(I) convention: I = Ybus @ V)
    I = (Ybus @ Vc.T).T
    S = Vc * np.conj(I)  # complex power injected at each bus
    return S  # P = S.real, Q = S.imag


def train_model(lam_physics, seed, epochs=300, lr=0.01):
    model = TwoLayerGCN(in_dim=2, hid_dim=16, out_dim=2, seed=seed)
    Xtr, Ytr = Xn[train_idx], Ytarget[train_idx]
    P_target = Xnode[train_idx][...,0]
    Q_target = Xnode[train_idx][...,1]
    phys_scale = 1.0 / (np.mean(P_target**2) + np.mean(Q_target**2) + 1e-6)  # normalize physics loss magnitude
    for ep in range(epochs):
        pred, cache = model.forward(Xtr)
        pred_clip = pred.copy()
        pred_clip[...,0] = np.clip(pred_clip[...,0], 0.5, 1.5)   # vm physically bounded
        pred_clip[...,1] = np.clip(pred_clip[...,1], -np.pi, np.pi)
        diff = pred - Ytr
        dOut_reg = 2*diff/(diff.shape[0]*diff.shape[1])

        if lam_physics > 0:
            vm_pred, va_pred = pred_clip[...,0], pred_clip[...,1]
            S_pred = power_balance_residual(vm_pred, va_pred)
            res_p = S_pred.real - P_target
            res_q = S_pred.imag - Q_target
            # per-sample-per-bus squared residual (not yet averaged), used for per-node finite diff
            r2_base = res_p**2 + res_q**2  # (B, n_bus)

            eps = 1e-4
            dOut_phys = np.zeros_like(pred)
            for j in range(n_bus):
                for k in range(2):  # 0=vm, 1=va
                    pp = pred_clip.copy()
                    pp[:, j, k] += eps
                    vm_p, va_p = pp[...,0], pp[...,1]
                    Sp = power_balance_residual(vm_p, va_p)
                    resp_p = Sp.real - P_target; resp_q = Sp.imag - Q_target
                    r2_pert = resp_p**2 + resp_q**2
                    # only the change summed over ALL buses (since bus j affects neighbours too)
                    dloss = phys_scale * (r2_pert.sum(axis=1) - r2_base.sum(axis=1)) / n_bus
                    dOut_phys[:, j, k] = dloss / eps
            dOut_phys = np.nan_to_num(dOut_phys, nan=0.0, posinf=0.0, neginf=0.0)
            # gradient-norm balancing: rescale physics grad to match regression grad's RMS
            # (standard practice for combining losses of very different natural scale)
            reg_rms = np.sqrt(np.mean(dOut_reg**2)) + 1e-12
            phys_rms = np.sqrt(np.mean(dOut_phys**2)) + 1e-12
            dOut_phys = dOut_phys * (reg_rms / phys_rms)
        else:
            dOut_phys = 0.0

        dOut = dOut_reg + lam_physics * dOut_phys
        model.backward(cache, dOut, lr)
    return model

def evaluate(model, idxset):
    pred, _ = model.forward(Xn[idxset])
    true = Ytarget[idxset]
    mae_vm = np.mean(np.abs(pred[...,0]-true[...,0]))
    rmse_vm = np.sqrt(np.mean((pred[...,0]-true[...,0])**2))
    vm_pred, va_pred = pred[...,0], pred[...,1]
    S_pred = power_balance_residual(vm_pred, va_pred)
    P_target = Xnode[idxset][...,0]; Q_target = Xnode[idxset][...,1]
    kcl_p = np.mean(np.abs(S_pred.real - P_target))
    kcl_q = np.mean(np.abs(S_pred.imag - Q_target))
    return mae_vm, rmse_vm, kcl_p, kcl_q

# ---- Experiment 1: multiple seeds, baseline (lambda=0) vs PI-GNN (lambda=1.0) -> real error bars
seeds = [0,1,2,3,4]
results_baseline = []
results_pignn = []
for s in seeds:
    m0 = train_model(lam_physics=0.0, seed=s)
    m1 = train_model(lam_physics=1.0, seed=s)
    results_baseline.append(evaluate(m0, test_idx))
    results_pignn.append(evaluate(m1, test_idx))

results_baseline = np.array(results_baseline)  # (5,4): mae_vm, rmse_vm, kcl_p, kcl_q
results_pignn = np.array(results_pignn)

print("=== Baseline GCN (lambda=0), mean +/- std over 5 seeds ===")
print("MAE_vm: %.5f +/- %.5f" % (results_baseline[:,0].mean(), results_baseline[:,0].std()))
print("RMSE_vm: %.5f +/- %.5f" % (results_baseline[:,1].mean(), results_baseline[:,1].std()))
print("KCL_P resid: %.5f +/- %.5f" % (results_baseline[:,2].mean(), results_baseline[:,2].std()))
print("KCL_Q resid: %.5f +/- %.5f" % (results_baseline[:,3].mean(), results_baseline[:,3].std()))

print("=== PI-GNN (lambda=1.0), mean +/- std over 5 seeds ===")
print("MAE_vm: %.5f +/- %.5f" % (results_pignn[:,0].mean(), results_pignn[:,0].std()))
print("RMSE_vm: %.5f +/- %.5f" % (results_pignn[:,1].mean(), results_pignn[:,1].std()))
print("KCL_P resid: %.5f +/- %.5f" % (results_pignn[:,2].mean(), results_pignn[:,2].std()))
print("KCL_Q resid: %.5f +/- %.5f" % (results_pignn[:,3].mean(), results_pignn[:,3].std()))

pct_reduction_p = 100*(1 - results_pignn[:,2].mean()/results_baseline[:,2].mean())
pct_reduction_q = 100*(1 - results_pignn[:,3].mean()/results_baseline[:,3].mean())
print(f"Real measured KCL P-residual reduction: {pct_reduction_p:.1f}%")
print(f"Real measured KCL Q-residual reduction: {pct_reduction_q:.1f}%")

# paired t-test (real stat test) on MAE_vm across seeds
from math import sqrt
diffs = results_baseline[:,0] - results_pignn[:,0]
t_stat = diffs.mean() / (diffs.std(ddof=1)/sqrt(len(diffs)))
print(f"Paired t-stat (MAE_vm baseline vs PI-GNN, n={len(diffs)} seeds): t={t_stat:.3f}")

# ---- Experiment 2: ablation on lambda_physics
print("\n=== Ablation: physics-loss weight lambda ===")
lambdas = [0.0, 0.1, 0.5, 1.0, 5.0, 20.0]
ablation = {}
for lam in lambdas:
    reps = [evaluate(train_model(lam_physics=lam, seed=s), test_idx) for s in [0,1,2]]
    reps = np.array(reps)
    ablation[lam] = (reps[:,0].mean(), reps[:,0].std(), reps[:,2].mean(), reps[:,2].std())
    print(f"lambda={lam:6.2f}  MAE_vm={reps[:,0].mean():.5f}+/-{reps[:,0].std():.5f}  KCL_P={reps[:,2].mean():.5f}+/-{reps[:,2].std():.5f}")

json.dump({
    "baseline_mean_std": results_baseline.tolist(),
    "pignn_mean_std": results_pignn.tolist(),
    "pct_reduction_p": pct_reduction_p,
    "pct_reduction_q": pct_reduction_q,
    "t_stat": t_stat,
    "ablation": {str(k): v for k,v in ablation.items()}
}, open("experiment_results.json","w"), indent=2)
print("\nSaved experiment_results.json")
def calculate_stable_integrated_gradients(model, target_bus_idx, single_input_n, steps=50):
    """
    Computes standardized Integrated Gradients for a single test sample's prediction
    using a physically grounded zero-injection baseline to ensure reproducibility.
    """
    # 1. Establish a zero-injection baseline (P=0, Q=0) in normalized space
    # Since Xmean and Xstd are global constants from training data:
    zero_injection_raw = np.zeros((n_bus, 2)) # [P=0, Q=0]
    baseline_Xn = (zero_injection_raw - Xmean[0]) / Xstd[0] # Match (n_bus, 2) shape
    
    # 2. Linear interpolation path between baseline and current normalized input
    alphas = np.linspace(0.0, 1.0, num=steps)
    accumulated_gradients = np.zeros_like(single_input_n) # (n_bus, 2)
    
    eps = 1e-4 # Finite difference step size
    
    # 3. Path integration loop
    for alpha in alphas:
        # Generate the interpolated input snapshot
        path_input = baseline_Xn + alpha * (single_input_n - baseline_Xn)
        
        # Calculate finite difference gradients for the target bus's voltage magnitude (index 0)
        # Base forward pass
        pred_base, _ = model.forward(path_input[np.newaxis, ...]) # Add batch dim
        v_base = pred_base[0, target_bus_idx, 0]
        
        # Compute gradient for each feature (P and Q) across all nodes
        for n_idx in range(n_bus):
            for f_idx in range(2):
                perturbed_input = path_input.copy()
                perturbed_input[n_idx, f_idx] += eps
                
                pred_pert, _ = model.forward(perturbed_input[np.newaxis, ...])
                v_pert = pred_pert[0, target_bus_idx, 0]
                
                # Finite difference approximation: dV / dX
                grad_val = (v_pert - v_base) / eps
                accumulated_gradients[n_idx, f_idx] += grad_val
                
    # 4. Average gradients and calculate attribution profiles
    avg_gradients = accumulated_gradients / steps
    integrated_gradients = (single_input_n - baseline_Xn) * avg_gradients
    
    return integrated_gradients

# ---- Experiment 1: multiple seeds, baseline (lambda=0) vs PI-GNN (lambda=1.0) -> real error bars
seeds = [0,1,2,3,4]
results_baseline = []
results_pignn = []

# Tracker loops for XAI attribution consistency testing across causal nodes
xai_scores_baseline = []
xai_scores_pignn = []

# Assume Bus 4 and Bus 5 are indices 3 and 4 for the N-1 line outage test sample (e.g., test sample 0)
causal_buses = [3, 4] 
test_sample_idx = test_idx[0]
single_test_input = Xn[test_sample_idx]

for s in seeds:
    m0 = train_model(lam_physics=0.0, seed=s)
    m1 = train_model(lam_physics=1.0, seed=s)
    results_baseline.append(evaluate(m0, test_idx))
    results_pignn.append(evaluate(m1, test_idx))
    
    # Compute XAI attribution on causal nodes for target Bus 4 using both models
    ig_0 = calculate_stable_integrated_gradients(m0, target_bus_idx=3, single_input_n=single_test_input)
    ig_1 = calculate_stable_integrated_gradients(m1, target_bus_idx=3, single_input_n=single_test_input)
    
    # Calculate percentage share of total absolute attribution captured by causal nodes (Bus 4 & Bus 5)
    total_attr_0 = np.sum(np.abs(ig_0)) + 1e-12
    causal_attr_0 = np.sum(np.abs(ig_0[causal_buses, :]))
    xai_scores_baseline.append(causal_attr_0 / total_attr_0 * 100)
    
    total_attr_1 = np.sum(np.abs(ig_1)) + 1e-12
    causal_attr_1 = np.sum(np.abs(ig_1[causal_buses, :]))
    xai_scores_pignn.append(causal_attr_1 / total_attr_1 * 100)

results_baseline = np.array(results_baseline)  # (5,4): mae_vm, rmse_vm, kcl_p, kcl_q
results_pignn = np.array(results_pignn)

print("=== Baseline GCN (lambda=0), mean +/- std over 5 seeds ===")
print("MAE_vm: %.5f +/- %.5f" % (results_baseline[:,0].mean(), results_baseline[:,0].std()))
print("RMSE_vm: %.5f +/- %.5f" % (results_baseline[:,1].mean(), results_baseline[:,1].std()))
print("KCL_P resid: %.5f +/- %.5f" % (results_baseline[:,2].mean(), results_baseline[:,2].std()))
print("KCL_Q resid: %.5f +/- %.5f" % (results_baseline[:,3].mean(), results_baseline[:,3].std()))
print("XAI Causal Attribution Share: %.2f%% +/- %.2f%%" % (np.mean(xai_scores_baseline), np.std(xai_scores_baseline)))

print("\n=== PI-GNN (lambda=1.0), mean +/- std over 5 seeds ===")
print("MAE_vm: %.5f +/- %.5f" % (results_pignn[:,0].mean(), results_pignn[:,0].std()))
print("RMSE_vm: %.5f +/- %.5f" % (results_pignn[:,1].mean(), results_pignn[:,1].std()))
print("KCL_P resid: %.5f +/- %.5f" % (results_pignn[:,2].mean(), results_pignn[:,2].std()))
print("KCL_Q resid: %.5f +/- %.5f" % (results_pignn[:,3].mean(), results_pignn[:,3].std()))
print("XAI Causal Attribution Share: %.2f%% +/- %.2f%%" % (np.mean(xai_scores_pignn), np.std(xai_scores_pignn)))
