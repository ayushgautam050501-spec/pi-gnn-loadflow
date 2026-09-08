"""
N-1 Contingency Analysis (simple check) and Expanded N-1/N-2 Contingency
Dataset Generation with GroupKFold Cross-Validated Evaluation.
Reconstructed from dissertation Appendix, Items 4 and 7.

Requires: pandapower, numpy, pandas, scikit-learn
"""
import copy
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from powerflow9 import run_case, branch_data, bus_data, gen_data, N

np.random.seed(42)


def simple_n1_check(net=None):
    """Item 4: quick N-1 line-outage sweep on the base pandapower case, flags
    buses that leave the [0.95, 1.05] p.u. voltage band under each outage."""
    if net is None:
        net = nw.case9()
        pp.runpp(net)

    contingency_results = []
    for line_idx in net.line.index:
        copied_net = copy.deepcopy(net)
        copied_net.line.loc[line_idx, 'in_service'] = False
        try:
            pp.runpp(copied_net, enforce_q_lims=True)
            v_mag = copied_net.res_bus['vm_pu'].values
            critical_bus_indices = copied_net.res_bus.index[
                (v_mag < 0.95) | (v_mag > 1.05)
            ].tolist()
            contingency_results.append({'line_id': line_idx, 'critical_buses': critical_bus_indices})
        except pp.LoadflowNotConverged:
            continue
    return contingency_results


def expanded_n1_n2_dataset_and_cv():
    """Item 7: builds base + N-1 (9) + N-2 (36) = 46 topologies swept across
    15 loading steps (70-130%), producing node-level instances, then
    evaluates a RandomForest critical-bus classifier with GroupKFold
    (grouped by scenario, so no scenario leaks across train/test)."""
    n_lines = len(branch_data)
    topologies = [frozenset()]  # base case, no outage
    topologies += [frozenset([i]) for i in range(n_lines)]                  # N-1: 9 cases
    topologies += [frozenset(c) for c in combinations(range(n_lines), 2)]   # N-2: 36 cases
    print(f"Total topologies (base + N-1 + N-2): {len(topologies)}")

    load_scales = np.linspace(0.70, 1.30, 15)
    print(f"Loading steps: {len(load_scales)} -> total scenarios attempted: "
          f"{len(topologies) * len(load_scales)}")

    rows = []
    scenario_id = 0
    converged_scenarios = 0

    for topo in topologies:
        for scale in load_scales:
            Vm, Va, ok = run_case(scale, gen_scale=1.0, out_of_service=topo)
            if not ok:
                continue
            converged_scenarios += 1

            Pd = np.array([b[2] for b in bus_data]) * scale
            Qd = np.array([b[3] for b in bus_data]) * scale
            Pg = np.zeros(N)
            for (bus, pg, vg) in gen_data:
                Pg[bus-1] = pg
            Pnet = Pg - Pd
            Qnet = -Qd

            # crude line-connectivity degree + 1-hop neighbor P/Q aggregate
            # (message-passing-style feature)
            adj = np.zeros((N, N))
            for idx, (f, t, r, x, b) in enumerate(branch_data):
                if idx in topo:
                    continue
                adj[f-1, t-1] = 1
                adj[t-1, f-1] = 1
            degree = adj.sum(axis=1)
            neighbor_P = adj @ Pnet
            neighbor_Q = adj @ Qnet

            label = ((Vm < 0.95) | (Vm > 1.05)).astype(int)

            for i in range(N):
                rows.append({
                    "scenario_id": scenario_id,
                    "bus": i + 1,
                    "P": Pnet[i], "Q": Qnet[i],
                    "degree": degree[i],
                    "neighbor_P": neighbor_P[i], "neighbor_Q": neighbor_Q[i],
                    "load_scale": scale,
                    "n_lines_out": len(topo),
                    "label": label[i],
                })
            scenario_id += 1

    print(f"Converged scenarios: {converged_scenarios} / {len(topologies) * len(load_scales)}")
    print(f"Total node-level instances: {len(rows)}")

    df = pd.DataFrame(rows)
    print("\nClass balance (label=1 means critical / out-of-bounds voltage):")
    print(df['label'].value_counts(normalize=True).round(4))

    X = df[["P", "Q", "degree", "neighbor_P", "neighbor_Q", "load_scale", "n_lines_out"]].values
    y = df["label"].values
    groups = df["scenario_id"].values

    gkf = GroupKFold(n_splits=5)
    f1s, accs, precs, recs = [], [], [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                      class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        f1s.append(f1_score(y[test_idx], pred, zero_division=0))
        accs.append(accuracy_score(y[test_idx], pred))
        precs.append(precision_score(y[test_idx], pred, zero_division=0))
        recs.append(recall_score(y[test_idx], pred, zero_division=0))

    print("\n--- 5-fold GroupKFold cross-validation (grouped by scenario) ---")
    print(f"F1 (critical class): mean={np.mean(f1s):.4f} std={np.std(f1s):.4f} "
          f"per-fold={np.round(f1s, 4)}")
    print(f"Accuracy:             mean={np.mean(accs):.4f} std={np.std(accs):.4f}")
    print(f"Precision (critical): mean={np.mean(precs):.4f} std={np.std(precs):.4f}")
    print(f"Recall (critical):    mean={np.mean(recs):.4f} std={np.std(recs):.4f}")

    df.to_csv('expanded_scenarios.csv', index=False)
    return df


if __name__ == "__main__":
    print("=== Item 4: simple N-1 check ===")
    print(simple_n1_check())
    print("\n=== Item 7: expanded N-1/N-2 dataset + cross-validation ===")
    expanded_n1_n2_dataset_and_cv()
