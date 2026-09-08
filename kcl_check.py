"""
Supplementary KCL Violation Check.
Reconstructed from dissertation Appendix, Item 9.

Requires: pandapower, torch, numpy, scikit-learn
"""
import pandapower as pp
import pandapower.networks as pn
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

net = pn.case9()
pp.runpp(net)

V_mag = net.res_bus.vm_pu.values
V_ang = net.res_bus.va_degree.values


def calculate_kcl_violation(predicted_v, y_bus, injected_power):
    """S = V * conj(Y @ V); returns total |P_injected - P_calculated| violation."""
    calculated_s = predicted_v * (torch.matmul(y_bus, predicted_v).conj())
    violation = torch.abs(injected_power - calculated_s.real)
    return torch.sum(violation)


if __name__ == "__main__":
    # Example usage -- ground_truth_v and predicted_v would come from a trained
    # model's output compared against the Newton-Raphson solution above.
    # mae_vm = mean_absolute_error(ground_truth_v, predicted_v)
    # rmse_vm = np.sqrt(mean_squared_error(ground_truth_v, predicted_v))
    # print(f"MAE voltage magnitude: {mae_vm:.4f} p.u.")
    print("Ground-truth base case Vm:", np.round(V_mag, 4))
    print("Ground-truth base case Va:", np.round(V_ang, 3))
    print("Import calculate_kcl_violation(...) and call it with a trained model's "
          "predicted complex voltage tensor, Ybus tensor, and injected-power tensor.")
