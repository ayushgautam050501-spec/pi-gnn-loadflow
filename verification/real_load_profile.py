"""
Real Chambi 33kV load profile (TRAFO_LV2 feeder), manually digitized from a
photographed SCADA historian trend chart ("REAL POWER, Last 1440 minutes",
26-04-2026 15:41:41 to 27-04-2026 15:41:41).

IMPORTANT / HONESTY NOTE:
These 20 points were read by eye off a phone photo of a monitor screen -- there
is no OCR or pixel-perfect curve extraction behind them (attempted; the JPEG
compression and screen glare made automated extraction unreliable, so it was
abandoned rather than reported as more precise than it is). Treat these as an
approximate real load *shape and range* (roughly 3-14.3 MW over 24h), not as
precise metering values. The two sharp dips visible late in the original chart
coincide in time with the protection-event window described in Section 11 of
the paper and Section 10 of the dissertation; they are included here for that
reason, not because their exact depth is known precisely.

Axis calibration used (from the four labeled gridlines visible in the source
image): 14.30 MW (top), 9.13 MW, 3.95 MW, -1.22 MW (bottom), evenly spaced,
confirming a linear axis. Time axis: 24h span, per the chart's own "Last 1440
minutes" title.
"""
import numpy as np

# (hours_since_start, approx_MW) -- read by eye from the source chart
REAL_LOAD_CURVE_MW = [
    (0.0, 7.5), (1.5, 6.7), (3.0, 6.3), (4.5, 7.0), (6.0, 8.2),
    (7.5, 9.8), (9.0, 11.5), (10.5, 13.0), (12.0, 13.8), (13.5, 14.1),
    (15.0, 13.9), (16.0, 9.2), (17.0, 8.3), (18.5, 8.6),
    (19.5, 3.0),   # sharp dip -- coincides with protection-event window
    (20.0, 8.4),
    (21.0, 7.6),
    (21.5, 3.6),   # second sharp dip
    (22.5, 8.9), (24.0, 9.0),
]

# Nominal rating used to convert this feeder's absolute MW into a per-unit
# loading multiplier compatible with the synthetic IEEE 9-bus dataset's
# 70-130% convention. 9.13 MW (the chart's own mid-gridline value) is used as
# a stand-in "nominal" reference since no separate nameplate rating was found
# in the available documents.
NOMINAL_MW = 9.13


def real_loading_multipliers(n_samples=800, seed=42):
    """
    Sample loading multipliers from the REAL digitized curve's empirical
    distribution, instead of the synthetic dataset's flat Uniform(0.70, 1.30).
    Returned multipliers are clipped to [0.30, 1.60] to stay inside the
    Newton-Raphson solver's safe convergence band (see powerflow9.py).
    """
    times = np.array([t for t, _ in REAL_LOAD_CURVE_MW])
    values = np.array([v for _, v in REAL_LOAD_CURVE_MW])
    multipliers_at_points = values / NOMINAL_MW

    rng = np.random.RandomState(seed)
    # Interpolate a dense curve, then sample n_samples points from it with
    # added small noise, to approximate "many operating snapshots" from one
    # real 24h trace -- this is explicitly an approximation, not independent
    # real measurements.
    dense_t = np.linspace(0, 24, 500)
    dense_v = np.interp(dense_t, times, multipliers_at_points)
    idx = rng.randint(0, len(dense_t), size=n_samples)
    base = dense_v[idx]
    noise = rng.normal(0, 0.02, size=n_samples)
    out = np.clip(base + noise, 0.30, 1.60)
    return out


if __name__ == "__main__":
    m = real_loading_multipliers(800)
    print(f"Real-data-derived loading multipliers: n={len(m)}")
    print(f"  min={m.min():.3f}  max={m.max():.3f}  mean={m.mean():.3f}  std={m.std():.3f}")
    print(f"  (synthetic dataset used flat Uniform(0.70, 1.30) instead)")
