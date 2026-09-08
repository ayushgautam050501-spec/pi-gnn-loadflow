"""
Physics-Bounded Dataset Generation.
Reconstructed from dissertation Appendix, Item 5.

Requires: pandapower, numpy, pandas
"""
import numpy as np
import pandapower as pp
import pandapower.networks as pn


def generate_physics_bounded_dataset(num_samples=5000):
    """
    Simulates randomized grid scenarios between 70% and 130% of nominal load,
    using Newton-Raphson to extract the true electrical state vectors.
    """
    base_net = pn.case9()
    num_buses = len(base_net.bus)

    input_load_profiles = []
    target_voltage_states = []

    print(f"Initializing simulation loop for {num_samples} grid profiles...")
    np.random.seed(42)

    nominal_p = base_net.load.p_mw.copy()
    nominal_q = base_net.load.q_mvar.copy()

    for i in range(num_samples):
        random_scale = np.random.uniform(0.70, 1.30, size=len(base_net.load))
        base_net.load.p_mw = nominal_p * random_scale
        base_net.load.q_mvar = nominal_q * random_scale

        try:
            pp.runpp(base_net, algorithm='nr', numba=False)

            v_mag = base_net.res_bus.vm_pu.values.copy()
            v_ang = base_net.res_bus.va_degree.values.copy()

            bus_p_injections = np.zeros(num_buses)
            bus_q_injections = np.zeros(num_buses)

            for idx, bus_id in enumerate(base_net.load.bus.values):
                bus_p_injections[bus_id] = base_net.load.p_mw.values[idx]
                bus_q_injections[bus_id] = base_net.load.q_mvar.values[idx]

            feature_row = np.concatenate([bus_p_injections, bus_q_injections])
            target_row = np.concatenate([v_mag, v_ang])

            input_load_profiles.append(feature_row)
            target_voltage_states.append(target_row)

        except pp.LoadflowNotConverged:
            continue

    print(f"Successfully converged and extracted {len(input_load_profiles)} scenarios.")
    return np.array(input_load_profiles), np.array(target_voltage_states)


if __name__ == "__main__":
    X, Y = generate_physics_bounded_dataset(num_samples=800)
    print("X shape:", X.shape, " Y shape:", Y.shape)
