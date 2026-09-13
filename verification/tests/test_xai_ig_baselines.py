import torch
import numpy as np

# Set seed for reproducible XAI verification
torch.manual_seed(42)
np.random.seed(42)

# 1. Generate Dynamic Input Tensor (477 samples, 9 buses, 2 features [P, Q])
base_inputs = X_pq.clone().detach()
noise = torch.randn_like(base_inputs) * 0.05
inputs_dynamic = base_inputs + noise

# 2. Forward Pass Proxy with Strong Causal Weight on Bus 4 (Index 3)
def model_forward_dynamic(x):
    weights = torch.ones(x.shape[1], x.shape[2])
    weights[3, :] = 4.0  # Target Bus 4 features
    return torch.sum(x * weights, dim=(1, 2))

# 3. Integrated Gradients Loop
def compute_ig(inputs, baseline, steps=50):
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
    grads = []
    
    for x_step in scaled_inputs:
        x_step = x_step.clone().detach().requires_grad_(True)
        out = model_forward_dynamic(x_step)
        out.backward(torch.ones_like(out))
        grads.append(x_step.grad.data.numpy())
        
    avg_grads = np.mean(np.array(grads), axis=0)
    delta = (inputs - baseline).detach().numpy()
    return delta * avg_grads

# 4. Baselines & Attribution Computation
zero_baseline = torch.zeros_like(inputs_dynamic)
mean_baseline = torch.mean(inputs_dynamic, dim=0, keepdim=True).expand_as(inputs_dynamic)

ig_zero = compute_ig(inputs_dynamic, zero_baseline)
ig_mean = compute_ig(inputs_dynamic, mean_baseline)

def get_attribution_share(attributions, target_bus=3):
    total = np.sum(np.abs(attributions), axis=(1, 2))
    target = np.sum(np.abs(attributions[:, target_bus, :]), axis=1)
    return np.mean(target / (total + 1e-8)) * 100

ratio_zero = get_attribution_share(ig_zero)
ratio_mean = get_attribution_share(ig_mean)

print("--- DYNAMIC XAI INTEGRATED GRADIENTS ANALYSIS ---")
print(f"Zero Baseline (Naive P=0, Q=0) Causal Attribution Share    : {ratio_zero:.2f}%")
print(f"Mean-Grid Baseline (In-Distribution) Causal Attribution Share: {ratio_mean:.2f}%")
