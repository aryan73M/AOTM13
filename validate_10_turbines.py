
import numpy as np
import pandas as pd

from AEP_calculator import loadPowerCurve, searchSorted


# ============================================================
# 1. LOAD OPTIMIZED LAYOUT
# ============================================================

layout = pd.read_csv("10_turbine_layout.csv")

coords = layout[["x", "y"]].to_numpy(dtype=float)

N = len(coords)

print("Number of turbines:", N)


# ============================================================
# 2. LOAD OFFICIAL POWER CURVE
# ============================================================

power_curve = loadPowerCurve(
    "Shell_Hackathon_Dataset/power_curve.csv"
)


# ============================================================
# 3. CHECK FARM BOUNDARY
# ============================================================

boundary = 50.0
farm_size = 4000.0

boundary_ok = (
    (coords[:, 0] >= boundary).all()
    and (coords[:, 0] <= farm_size - boundary).all()
    and (coords[:, 1] >= boundary).all()
    and (coords[:, 1] <= farm_size - boundary).all()
)

print("\nBoundary constraints satisfied:", boundary_ok)


# ============================================================
# 4. CHECK MINIMUM DISTANCE
# ============================================================

min_distance = 400.0

distances = np.sqrt(
    (coords[:, 0, None] - coords[None, :, 0])**2
    +
    (coords[:, 1, None] - coords[None, :, 1])**2
)

np.fill_diagonal(distances, np.inf)

minimum_actual_distance = distances.min()

distance_ok = minimum_actual_distance >= min_distance - 1e-6

print("Minimum actual turbine distance:", minimum_actual_distance)
print("Proximity constraints satisfied:", distance_ok)


# ============================================================
# 5. WIND SCENARIO
# ============================================================

wind_speed = 15.0
wind_direction = 360.0

CT = 0.297620

rotor_radius = 50.0
kw = 0.05

wind_angle = np.radians(wind_direction - 90.0)

cos_dir = np.cos(wind_angle)
sin_dir = np.sin(wind_angle)


# ============================================================
# 6. ROTATE TURBINE COORDINATES
# ============================================================

x_rot = (
    coords[:, 0] * cos_dir
    -
    coords[:, 1] * sin_dir
)

y_rot = (
    coords[:, 0] * sin_dir
    +
    coords[:, 1] * cos_dir
)


# ============================================================
# 7. CALCULATE PAIRWISE WAKE DISTANCES
# ============================================================

x_dist = x_rot[:, None] - x_rot[None, :]
y_dist = np.abs(y_rot[:, None] - y_rot[None, :])


# ============================================================
# 8. CALCULATE OFFICIAL JENSEN WAKE DEFICIT
# ============================================================

wake_deficit = (
    1.0 - np.sqrt(1.0 - CT)
) * (
    rotor_radius /
    (rotor_radius + kw * x_dist)
)**2

wake_mask = (
    (x_dist <= 0)
    |
    (
        (x_dist > 0)
        &
        (y_dist > rotor_radius + kw * x_dist)
    )
)

wake_deficit[wake_mask] = 0.0

# No turbine creates a wake on itself
np.fill_diagonal(wake_deficit, 0.0)


# ============================================================
# 9. COMBINE MULTIPLE WAKES USING RSS
# ============================================================

combined_deficit = np.sqrt(
    np.sum(wake_deficit**2, axis=1)
)


# ============================================================
# 10. EFFECTIVE WIND SPEED
# ============================================================

# Match the evaluator's float32 wind-speed arithmetic
free_speed = np.float32(wind_speed)

effective_speed = (
    free_speed * (1.0 - combined_deficit.astype(np.float32))
).astype(np.float32)


# ============================================================
# 11. OFFICIAL POWER-CURVE LOOKUP
# ============================================================

indices = searchSorted(
    power_curve[:, 0],
    effective_speed
)

turbine_power = power_curve[indices, 2]

farm_power = np.sum(turbine_power)


# ============================================================
# 12. RESULTS
# ============================================================

print("\n========== VALIDATION RESULTS ==========")

print("Wind direction:", wind_direction)
print("Wind speed:", wind_speed)

print("\nMaximum combined wake deficit:", combined_deficit.max())
print("Average combined wake deficit:", combined_deficit.mean())

print("\nEffective wind speeds:")
print(effective_speed)

print("\nIndividual turbine powers (MW):")
print(turbine_power)

print("\nOfficial evaluator power:", farm_power, "MW")

print("\nBoundary valid:", boundary_ok)
print("Minimum distance valid:", distance_ok)

if boundary_ok and distance_ok:
    print("\nSUCCESS: Layout satisfies the geometric constraints.")
else:
    print("\nERROR: Layout violates one or more constraints.")
