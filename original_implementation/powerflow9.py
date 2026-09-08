"""
Standalone Newton-Raphson Solver (IEEE 9-Bus / WSCC system)
Reconstructed from dissertation Appendix, Item 6.
This is the core dependency imported by the other reconstructed scripts.
"""
import numpy as np

BASE_MVA = 100.0

# bus: [id, type(3=slack,2=PV,1=PQ), Pd(MW), Qd(MVAr)]
bus_data = [
    (1, 3, 0.0, 0.0),
    (2, 2, 0.0, 0.0),
    (3, 2, 0.0, 0.0),
    (4, 1, 0.0, 0.0),
    (5, 1, 90.0, 30.0),
    (6, 1, 0.0, 0.0),
    (7, 1, 100.0, 35.0),
    (8, 1, 0.0, 0.0),
    (9, 1, 125.0, 50.0),
]

# gen: bus, Pg(MW), Vg(pu)
gen_data = [
    (1, 0.0, 1.04),
    (2, 163.0, 1.025),
    (3, 85.0, 1.025),
]

# branch: from, to, r(pu), x(pu), b(pu)
branch_data = [
    (1, 4, 0.0,   0.0576, 0.0),
    (4, 5, 0.017, 0.092,  0.158),
    (5, 6, 0.039, 0.17,   0.358),
    (3, 6, 0.0,   0.0586, 0.0),
    (6, 7, 0.0119,0.1008, 0.209),
    (7, 8, 0.0085,0.072,  0.149),
    (8, 2, 0.0,   0.0625, 0.0),
    (8, 9, 0.032, 0.161,  0.306),
    (9, 4, 0.01,  0.085,  0.176),
]

N = 9


def build_ybus(branches, out_of_service=None):
    out_of_service = out_of_service or set()
    Y = np.zeros((N, N), dtype=complex)
    for idx, (f, t, r, x, b) in enumerate(branches):
        if idx in out_of_service:
            continue
        y = 1.0 / complex(r, x)
        Y[f-1, f-1] += y + 1j*b/2
        Y[t-1, t-1] += y + 1j*b/2
        Y[f-1, t-1] -= y
        Y[t-1, f-1] -= y
    return Y


def newton_raphson(Ybus, Pd, Qd, Pg, Vset, bus_types, tol=1e-8, max_iter=30):
    """bus_types: array len N, 3=slack, 2=PV, 1=PQ. Returns Vm, Va (deg), converged flag."""
    Vm = np.array([Vset[i] if bus_types[i] != 1 else 1.0 for i in range(N)])
    Va = np.zeros(N)
    P_spec = (Pg - Pd) / BASE_MVA
    Q_spec = (0.0 - Qd) / BASE_MVA  # generators assumed Q-free except handled implicitly

    pq = [i for i in range(N) if bus_types[i] == 1]
    pv = [i for i in range(N) if bus_types[i] == 2]
    non_slack = pv + pq

    G = Ybus.real
    B = Ybus.imag

    for it in range(max_iter):
        V = Vm * np.exp(1j*Va)
        S_calc = V * np.conj(Ybus @ V)
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        dP = P_spec[non_slack] - P_calc[non_slack]
        dQ = Q_spec[pq] - Q_calc[pq]
        mismatch = np.concatenate([dP, dQ])

        if np.max(np.abs(mismatch)) < tol:
            return Vm, np.degrees(Va), True

        n1, n2 = len(non_slack), len(pq)
        J = np.zeros((n1+n2, n1+n2))

        def dPi_dVaj(i, j):
            if i == j:
                return -Q_calc[i] - B[i, i]*Vm[i]**2
            return Vm[i]*Vm[j]*(G[i, j]*np.sin(Va[i]-Va[j]) - B[i, j]*np.cos(Va[i]-Va[j]))

        def dPi_dVmj(i, j):
            if i == j:
                return P_calc[i]/Vm[i] + G[i, i]*Vm[i]
            return Vm[i]*(G[i, j]*np.cos(Va[i]-Va[j]) + B[i, j]*np.sin(Va[i]-Va[j]))

        def dQi_dVaj(i, j):
            if i == j:
                return P_calc[i] - G[i, i]*Vm[i]**2
            return -Vm[i]*Vm[j]*(G[i, j]*np.cos(Va[i]-Va[j]) + B[i, j]*np.sin(Va[i]-Va[j]))

        def dQi_dVmj(i, j):
            if i == j:
                return Q_calc[i]/Vm[i] - B[i, i]*Vm[i]
            return Vm[i]*(G[i, j]*np.sin(Va[i]-Va[j]) - B[i, j]*np.cos(Va[i]-Va[j]))

        for a, i in enumerate(non_slack):
            for b_, j in enumerate(non_slack):
                J[a, b_] = dPi_dVaj(i, j)
            for b_, j in enumerate(pq):
                J[a, n1+b_] = dPi_dVmj(i, j)
        for a, i in enumerate(pq):
            for b_, j in enumerate(non_slack):
                J[n1+a, b_] = dQi_dVaj(i, j)
            for b_, j in enumerate(pq):
                J[n1+a, n1+b_] = dQi_dVmj(i, j)

        try:
            dx = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            return None, None, False

        for a, i in enumerate(non_slack):
            Va[i] += dx[a]
        for a, i in enumerate(pq):
            Vm[i] += dx[n1+a]

        if np.any(Vm < 0.5) or np.any(Vm > 1.6):
            return None, None, False

    return None, None, False


def run_case(load_scale, gen_scale=1.0, out_of_service=None):
    bus_types = np.array([b[1] for b in bus_data])
    Pd = np.array([b[2] for b in bus_data]) * load_scale
    Qd = np.array([b[3] for b in bus_data]) * load_scale
    Pg = np.zeros(N)
    Vset = np.ones(N)
    for (bus, pg, vg) in gen_data:
        Pg[bus-1] = pg * gen_scale
        Vset[bus-1] = vg
    Vset[0] = gen_data[0][2]
    Ybus = build_ybus(branch_data, out_of_service)
    Vm, Va, ok = newton_raphson(Ybus, Pd, Qd, Pg, Vset, bus_types)
    return Vm, Va, ok


if __name__ == "__main__":
    Vm, Va, ok = run_case(1.0)
    print("Base case converged:", ok)
    print("Vm:", np.round(Vm, 4))
    print("Va:", np.round(Va, 3))
