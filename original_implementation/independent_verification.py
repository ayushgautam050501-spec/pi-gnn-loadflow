"""
Independent NumPy Verification of Headline Performance Claims.
Reconstructed from dissertation Appendix, Item 8.

Verifies the inference-speed, voltage-error, and attribution claims
independently of the PyTorch Geometric training environment, using a
minimal two-layer GCN implemented from scratch in NumPy, on the same
IEEE 9-bus benchmark.

Requires: numpy. Depends on powerflow9.py (same folder).
"""
import time
import numpy as np

from powerflow9 import run_case, branch_data, bus_data, gen_data, N

np.random.seed(42)


def build_adjacency():
    """
    Edge weights are derived from |Ybus| magnitude (consistent with
    verification/train_and_ablate.py and Section 5 of the paper), not
    plain 0/1 topology -- so a bus connected via a low-impedance line
    gets more "say" in the aggregation than one connected via a
    high-impedance line, matching how KCL actually distributes current.
    """
    from powerflow9 import build_ybus, branch_data as _branch_data
    Ybus = build_ybus(_branch_data)
    A = np.abs(Ybus)
    np.fill_diagonal(A, 0.0)          # remove self-admittance, add explicit self-loop below
    A = A + np.eye(N) * A.max()       # self-loop weight (standard GCN trick)
    D = np.diag(1.0 / np.sqrt(A.sum(axis=1)))  # normalize so messages don't blow up
    return D @ A @ D


A_norm = build_adjacency()


def make_dataset(n_samples=800):
    X_list, Y_list = [], []
    for _ in range(n_samples):
        scale = np.random.uniform(0.70, 1.30)
        Vm, Va, ok = run_case(scale)
        if not ok:
            continue
        Pd = np.array([b[2] for b in bus_data]) * scale
        Qd = np.array([b[3] for b in bus_data]) * scale
        Pg = np.zeros(N)
        for (bus, pg, vg) in gen_data:
            Pg[bus-1] = pg
        P_inj = (Pg - Pd) / 100.0   # per-unit
        Q_inj = (-Qd) / 100.0
        X_list.append(np.stack([P_inj, Q_inj], axis=1))
        # Predict the DEVIATION from nominal (Vm-1.0, Va/100) rather than the raw
        # value -- easier to learn than predicting "~1.0" out of near-zero input.
        Y_list.append(np.stack([Vm - 1.0, Va/100.0], axis=1))
    return np.array(X_list), np.array(Y_list)


class SimpleGCN:
    def __init__(self, in_dim=2, hidden=16, out_dim=2, lr=0.05):
        self.W1 = np.random.randn(in_dim, hidden) * 0.5
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, out_dim) * 0.5
        self.b2 = np.zeros(out_dim)
        self.lr = lr
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.t = 0

    def forward(self, X):
        self.X = X
        self.Z1 = A_norm @ X @ self.W1 + self.b1
        self.H1 = np.maximum(self.Z1, 0)
        self.Z2 = A_norm @ self.H1 @ self.W2 + self.b2
        return self.Z2

    def backward(self, Y_pred, Y_true):
        N_, out_dim = Y_true.shape
        dZ2 = 2 * (Y_pred - Y_true) / (N_ * out_dim)
        dW2 = self.H1.T @ (A_norm.T @ dZ2)
        db2 = dZ2.sum(axis=0)
        dH1 = A_norm.T @ dZ2 @ self.W2.T
        dZ1 = dH1 * (self.Z1 > 0)
        dW1 = self.X.T @ (A_norm.T @ dZ1)
        db1 = dZ1.sum(axis=0)

        self.t += 1
        for p, dp, m, v in [(self.W1, dW1, 'mW1', 'vW1'), (self.W2, dW2, 'mW2', 'vW2')]:
            mom = getattr(self, m); var = getattr(self, v)
            mom[:] = 0.9*mom + 0.1*dp
            var[:] = 0.999*var + 0.001*(dp**2)
            p -= self.lr * mom / (np.sqrt(var) + 1e-8)
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2

    def train(self, X_train, Y_train, epochs=400):
        for ep in range(epochs):
            total_loss = 0
            idx = np.random.permutation(len(X_train))
            for i in idx:
                pred = self.forward(X_train[i])
                loss = np.mean((pred - Y_train[i])**2)
                total_loss += loss
                self.backward(pred, Y_train[i])
            if ep % 50 == 0:
                print(f" epoch {ep:3d} MSE={total_loss/len(X_train):.6f}")


def integrated_gradients(model, x_input, x_baseline, target_node, target_col, m_steps=50):
    total_grad = np.zeros_like(x_input)
    for k in range(1, m_steps + 1):
        alpha = k / m_steps
        x_interp = x_baseline + alpha * (x_input - x_baseline)
        eps = 1e-4
        grad = np.zeros_like(x_input)
        base_out = model.forward(x_interp)[target_node, target_col]
        for i in range(x_input.shape[0]):
            for j in range(x_input.shape[1]):
                x_perturbed = x_interp.copy()
                x_perturbed[i, j] += eps
                out_perturbed = model.forward(x_perturbed)[target_node, target_col]
                grad[i, j] = (out_perturbed - base_out) / eps
        total_grad += grad
    avg_grad = total_grad / m_steps
    return (x_input - x_baseline) * avg_grad


def main():
    X, Y = make_dataset(800)
    n = len(X)
    split = int(n * 0.8)
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]
    print(f"Dataset: {n} converged samples ({split} train / {n-split} test)")

    model = SimpleGCN()
    print("\nTraining...")
    model.train(X_train, Y_train, epochs=400)

    n_timing_runs = 100
    start = time.perf_counter()
    for i in range(n_timing_runs):
        _ = model.forward(X_test[i % len(X_test)])
    elapsed = time.perf_counter() - start
    per_sample_ms = (elapsed / n_timing_runs) * 1000
    print(f"\n[REAL NUMBER 1] Average inference time per sample: {per_sample_ms:.4f} ms "
          f"({per_sample_ms/1000:.6f} s)")

    errors = []
    for i in range(len(X_test)):
        pred = model.forward(X_test[i])
        vm_error = np.abs(pred[:, 0] - Y_test[i][:, 0])
        errors.append(vm_error)
    errors = np.concatenate(errors)
    print(f"[REAL NUMBER 2] Voltage magnitude error -- Mean: {errors.mean():.5f} p.u., "
          f"Max: {errors.max():.5f} p.u.")

    # Simulate a "fault-like" stressed scenario: line index 2 (bus 4-5) outaged, heavy load
    Vm_fault, Va_fault, ok = run_case(1.25, out_of_service={2})
    if ok:
        Pd = np.array([b[2] for b in bus_data]) * 1.25
        Qd = np.array([b[3] for b in bus_data]) * 1.25
        Pg = np.zeros(N)
        for (bus, pg, vg) in gen_data:
            Pg[bus-1] = pg
        x_fault = np.stack([(Pg - Pd)/100.0, (-Qd)/100.0], axis=1)
        x_baseline = np.zeros_like(x_fault)  # zero-injection reference

        target_node = int(np.argmax(np.abs(Vm_fault - 1.0)))
        ig_scores = integrated_gradients(model, x_fault, x_baseline, target_node,
                                          target_col=0, m_steps=30)
        abs_scores = np.abs(ig_scores).sum(axis=1)
        total_attribution = abs_scores.sum()

        faulted_line_buses = [3, 4]  # bus 4 and bus 5 (0-indexed)
        causal_attribution = abs_scores[faulted_line_buses].sum()
        pct_on_causal_buses = 100 * causal_attribution / total_attribution

        print(f"\n[REAL NUMBER 3] Simulated fault: line Bus4-Bus5 removed, 125% load")
        print(f" Most voltage-stressed bus: Bus {target_node+1} (Vm={Vm_fault[target_node]:.4f} p.u.)")
        print(f" Per-bus attribution (|IG| score): {np.round(abs_scores, 5)}")
        print(f" Attribution weight on physically-causal buses (4,5): {pct_on_causal_buses:.1f}%")
    else:
        print("Fault scenario did not converge -- try a different line/loading combination.")


if __name__ == "__main__":
    main()
