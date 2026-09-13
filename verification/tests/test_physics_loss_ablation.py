import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandapower as pp
import pandapower.networks as pn

# 1. Reproducibility & Data Generation
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)
net = pn.case9()
num_samples = 500
X_pq, X_pqi, Y_targets = [], [], []

for _ in range(num_samples):
    scale = np.random.uniform(0.7, 1.3)
    net.load['p_mw'] = net.load['p_mw'] * scale
    net.load['q_mvar'] = net.load['q_mvar'] * scale
    try:
        pp.runpp(net, algorithm='nr')
        p = net.res_bus['p_mw'].values / 100.0
        q = net.res_bus['q_mvar'].values / 100.0
        v = net.res_bus['vm_pu'].values
        va = np.radians(net.res_bus['va_degree'].values)
        i_avg = np.sqrt(p**2 + q**2) / (v + 1e-6)
        
        X_pq.append(np.stack([p, q], axis=-1))
        X_pqi.append(np.stack([p, q, i_avg], axis=-1))
        Y_targets.append(np.stack([v, va], axis=-1))
    except:
        continue

X_pq = torch.tensor(np.array(X_pq), dtype=torch.float32)
X_pqi = torch.tensor(np.array(X_pqi), dtype=torch.float32)
Y_targets = torch.tensor(np.array(Y_targets), dtype=torch.float32)

# 2. GCN Architecture
class SimpleGCN(nn.Module):
    def __init__(self, in_channels):
        super(SimpleGCN, self).__init__()
        self.fc1 = nn.Linear(in_channels, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.out(h)

# 3. Training & Evaluation Pipeline
def train_and_eval(X, Y, use_physics=False, lam=1.0, seeds=5):
    angle_maes, v_maes = [], []
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    for seed in range(seeds):
        torch.manual_seed(seed)
        model = SimpleGCN(in_channels=X.shape[-1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(250):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            loss_mse = F.mse_loss(pred, Y_train)
            
            if use_physics:
                v_pred, angle_pred = pred[:, :, 0], pred[:, :, 1]
                loss_kcl = torch.mean(torch.abs(v_pred - 1.0)**2) + torch.mean(torch.abs(angle_pred)**2) * 0.1
                total_loss = loss_mse + lam * loss_kcl
            else:
                total_loss = loss_mse
                
            total_loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            v_err = torch.abs(preds[:, :, 0] - Y_test[:, :, 0]).mean().item()
            angle_err = torch.abs(preds[:, :, 1] - Y_test[:, :, 1]).mean().item()
            v_maes.append(v_err)
            angle_maes.append(angle_err)
            
    return np.mean(v_maes), np.mean(angle_maes)

# 4. Run Benchmark
results = {
    'Config A (Blind: PQ, No Physics)': train_and_eval(X_pq, Y_targets, use_physics=False),
    'Config A (Blind: PQ, PI-GNN lam=1.0)': train_and_eval(X_pq, Y_targets, use_physics=True, lam=1.0),
    'Config B (Feature Test: PQI, No Physics)': train_and_eval(X_pqi, Y_targets, use_physics=False),
    'Config B (Feature Test: PQI, PI-GNN lam=1.0)': train_and_eval(X_pqi, Y_targets, use_physics=True, lam=1.0),
    'Config C (Loss Balancing: PQ, PI-GNN lam=0.1)': train_and_eval(X_pq, Y_targets, use_physics=True, lam=0.1),
    'Config D (Full Setup: PQI, PI-GNN lam=0.1)': train_and_eval(X_pqi, Y_targets, use_physics=True, lam=0.1),
}

print("\n--- ABLATION RESULTS SUMMARY ---")
for config, (v_err, a_err) in results.items():
    print(f"{config:<45} | Voltage MAE: {v_err:.5f} p.u. | Angle MAE: {a_err:.5f} rad")
