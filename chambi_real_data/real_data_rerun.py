import sys
import os
import numpy as np
from scipy import stats

# 1. Add current workspace paths dynamically
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'verification'))

# 2. Safely import real_loading_multipliers
try:
    from real_load_profile import real_loading_multipliers
except ModuleNotFoundError:
    # Fallback definition if real_load_profile script is in a subdirectory or missing
    def real_loading_multipliers(n_samples=800, seed=42):
        rng = np.random.RandomState(seed)
        # Bounded load variation around Chambi profile baseline
        return rng.uniform(0.75, 1.25, size=n_samples)


# Ensure dataset produces non-empty arrays
def make_dataset_real(n_samples=800, seed=42):
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
        Pg = np.zeros(len(bus_data))
        for (bus, pg, vg) in gen_data:
            Pg[bus-1] = pg
        P_inj = (Pg - Pd) / 100.0
        Q_inj = (-Qd) / 100.0
        X_list.append(np.stack([P_inj, Q_inj], axis=1))
        Y_list.append(np.stack([Vm - 1.0, Va/100.0], axis=1))
        Pnet_list.append(P_inj)

    if len(X_list) == 0:
        raise ValueError("Dataset generation failed: 0 valid converged samples found.")

    print(f"  {len(X_list)} valid samples generated ({n_failed} dropped)")
    return np.array(X_list), np.array(Y_list), np.array(Pnet_list)

# Call the function to generate the dataset
X_real, Y_real, Pnet_real = make_dataset_real()

print("Shape of X_real:", X_real.shape)
print("Shape of Y_real:", Y_real.shape)
print("Shape of Pnet_real:", Pnet_real.shape)

