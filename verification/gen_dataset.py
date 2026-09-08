import numpy as np
import pandapower as pp
import pandapower.networks as nw
import json

np.random.seed(42)

net0 = nw.case9()
n_bus = len(net0.bus)

# Build Ybus (admittance matrix) once - topology doesn't change across loading scenarios
pp.runpp(net0)
Ybus = net0._ppc["internal"]["Ybus"].toarray()

samples = []
n_target = 800
attempts = 0
while len(samples) < n_target and attempts < 3000:
    attempts += 1
    net = nw.case9()
    scale = np.random.uniform(0.7, 1.3, size=len(net.load))
    net.load['p_mw'] = net0.load['p_mw'].values * scale
    net.load['q_mvar'] = net0.load['q_mvar'].values * scale
    try:
        pp.runpp(net, algorithm='nr', tolerance_mva=1e-8)
    except Exception:
        continue
    vm = net.res_bus['vm_pu'].values.copy()
    va = net.res_bus['va_degree'].values.copy()
    p_bus = np.zeros(n_bus)
    q_bus = np.zeros(n_bus)
    for i, row in net.load.iterrows():
        p_bus[row.bus] -= row.p_mw
        q_bus[row.bus] -= row.q_mvar
    for i, row in net.gen.iterrows():
        p_bus[row.bus] += net.res_gen.loc[i, 'p_mw']
    p_bus[net.ext_grid.bus.values[0]] += net.res_ext_grid['p_mw'].values[0]
    q_bus[net.ext_grid.bus.values[0]] += net.res_ext_grid['q_mvar'].values[0]

    samples.append({
        "p_in": p_bus.tolist(), "q_in": q_bus.tolist(),
        "vm": vm.tolist(), "va": va.tolist()
    })

print(f"Generated {len(samples)} converged samples out of {attempts} attempts")
with open("dataset.json", "w") as f:
    json.dump({"Ybus_real": Ybus.real.tolist(), "Ybus_imag": Ybus.imag.tolist(),
               "samples": samples}, f)
