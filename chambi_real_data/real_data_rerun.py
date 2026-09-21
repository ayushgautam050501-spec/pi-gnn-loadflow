Before running the file remember to install pandapower with the given code- !pip install pandapower in the terminal.WIthout adding the code its not possible for the viewer to run the given codes.
Also remember to  kindly run the realloadprofile.py and the powerflow9.py.Without running these given files the code in this section will show error.
Please install the codes given in the file gen_dataset.py before running the real_data_rerun.py code.
    
import sys
import os
import numpy as np
from scipy import stats

# 1. Add current workspace paths dynamically
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'verification'))        ( This commadn helps us get the current folder name,and also indicates Python "look inside this folder")

# 2. Safely import real_loading_multipliers
try:
    from real_load_profile import real_loading_multipliers
except ModuleNotFoundError:
    # Fallback definition if real_load_profile script is in a subdirectory or missing
    def real_loading_multipliers(n_samples=800, seed=42):
        rng = np.random.RandomState(seed)
        # Bounded load variation around Chambi profile baseline
        return rng.uniform(0.75, 1.25, size=n_samples)                                                                 (It helps us try to pull in the real chambi load data )   


# Ensure dataset produces non-empty arrays
def make_dataset_real(n_samples=800, seed=42):
    scales = real_loading_multipliers(n_samples, seed=seed)                                  (Gets the 800 loading multipliers from whichever function ended up defined above)
    X_list, Y_list, Pnet_list = [], [], []
    n_failed = 0
    for scale in scales:                                                                     (This sets up empty lists to collect the results,through each of 800 multipliers) 
        Vm, Va, ok = run_case(float(scale))
        if not ok:
            n_failed += 1
            continue                                                                                              (Runs our actual Newton-raphson solver at the loading level)
        Pd = np.array([b[2] for b in bus_data]) * scale
        Qd = np.array([b[3] for b in bus_data]) * scale                    (This helps us calculate how much active power(Pd) and reactive power(Qd) each of bus is demanding)
        Pg = np.zeros(len(bus_data))
        for (bus, pg, vg) in gen_data:
            Pg[bus-1] = pg                                                         (Builds array of how much power each generator is producing)(0 for buses with no generator)
        P_inj = (Pg - Pd) / 100.0
        Q_inj = (-Qd) / 100.0                                                                                                                (Net power injection at each bus)
        X_list.append(np.stack([P_inj, Q_inj], axis=1))
        Y_list.append(np.stack([Vm - 1.0, Va/100.0], axis=1))
        Pnet_list.append(P_inj)                                                                                                                   (Saves this scenarios input)

    if len(X_list) == 0:
        raise ValueError("Dataset generation failed: 0 valid converged samples found.")                                 ( Reports us how much of our samples actually worked )

    print(f"  {len(X_list)} valid samples generated ({n_failed} dropped)")
    return np.array(X_list), np.array(Y_list), np.array(Pnet_list)

# Call the function to generate the dataset
X_real, Y_real, Pnet_real = make_dataset_real()

print("Shape of X_real:", X_real.shape)                                                                    (Actually calls the function and prints the resulting array shapes)
print("Shape of Y_real:", Y_real.shape)
print("Shape of Pnet_real:", Pnet_real.shape)

Note for the vieweer- This experiment pipeline represented by the program real_data_rerun.py (which thus also performs the power-flow computations,helps us puting the dataset together) 
is also used for the loading scenarios,both real and synthetic.The given real loading data exists separately under the real_load_profile.py,but which contains the real loading data in 
the form of the Chambi TRAFO_LV2 SCADA data in the 24-hour digital form.The real_data_rerun.py code takes its loading factors from that program; sampling Uniform(0.75,1.25) 
acts only as a backup option in case of nonavailability of the real_load_profile.py program in some environment, and the program was never the source of results stated 
in Section 16.2.
