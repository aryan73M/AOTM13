
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from AEP_calculator import (
    loadPowerCurve,
    binWindResourceData,
    searchSorted
)


# ============================================================
# 1. PROBLEM PARAMETERS
# ============================================================

N = 5

FARM_SIZE = 4000.0
BOUNDARY = 50.0
MIN_DISTANCE = 400.0

ROTOR_RADIUS = 50.0
KW = 0.05

N_DIRECTIONS = 36
N_SPEEDS = 15
N_SCENARIOS = N_DIRECTIONS * N_SPEEDS

EPS = 1e-5

TIME_LIMIT = 600
MIP_GAP = 0.01

POWER_CURVE_FILE = (
    "Shell_Hackathon_Dataset/power_curve.csv"
)

WIND_DATA_FILE = (
    "Shell_Hackathon_Dataset/Wind Data/wind_data_2007.csv"
)


# ============================================================
# 2. LOAD OFFICIAL DATA
# ============================================================

power_curve = loadPowerCurve(POWER_CURVE_FILE)

wind_probability = binWindResourceData(WIND_DATA_FILE)

assert len(wind_probability) == N_SCENARIOS
assert np.isclose(np.sum(wind_probability), 1.0)

power_speeds = power_curve[:, 0].astype(float)
power_values = power_curve[:, 2].astype(float)


# ============================================================
# 3. GENERATE THE 540 WIND SCENARIOS
# ============================================================

# Same direction order as the official evaluator
wind_directions = np.roll(
    np.arange(10, 361, 10, dtype=np.float32),
    1
)

# Midpoints of the 15 official wind-speed bins
wind_speeds = np.arange(
    1.0, 30.0, 2.0, dtype=np.float32
)

assert len(wind_directions) == 36
assert len(wind_speeds) == 15

# Official evaluator selects Ct using searchSorted
ct_indices = searchSorted(
    power_curve[:, 0],
    wind_speeds
)

ct_values = power_curve[ct_indices, 1].astype(float)

# Jensen coefficient for each wind-speed scenario
jensen_coefficients = (
    1.0 - np.sqrt(1.0 - ct_values)
)

print("Wind scenarios:", N_SCENARIOS)
print("Wind directions:", len(wind_directions))
print("Wind speeds:", len(wind_speeds))
print("Sum of probabilities:", wind_probability.sum())


# ============================================================
# 4. CREATE GUROBI MODEL
# ============================================================

model = gp.Model("Five_Turbine_Full_Wind_Problem")

model.Params.NonConvex = 2
model.Params.TimeLimit = TIME_LIMIT
model.Params.MIPGap = MIP_GAP
model.Params.MIPFocus = 1
model.Params.OutputFlag = 1


# ============================================================
# 5. DECISION VARIABLES: TURBINE LOCATIONS
# ============================================================

x = model.addVars(
    N,
    lb=BOUNDARY,
    ub=FARM_SIZE - BOUNDARY,
    vtype=GRB.CONTINUOUS,
    name="x"
)

y = model.addVars(
    N,
    lb=BOUNDARY,
    ub=FARM_SIZE - BOUNDARY,
    vtype=GRB.CONTINUOUS,
    name="y"
)


# ============================================================
# 6. INITIAL FEASIBLE LAYOUT
# ============================================================

# Five turbines in one row.
# This is only a MIP starting solution.

initial_x = [300, 1100, 1900, 2700, 3500]
initial_y = [300, 300, 300, 300, 300]

for i in range(N):
    x[i].Start = initial_x[i]
    y[i].Start = initial_y[i]


# ============================================================
# 7. BOUNDARY AND PROXIMITY CONSTRAINTS
# ============================================================

for i in range(N):
    for j in range(i + 1, N):

        model.addQConstr(
            (x[i] - x[j]) ** 2
            +
            (y[i] - y[j]) ** 2
            >= MIN_DISTANCE ** 2,
            name=f"proximity_{i}_{j}"
        )


# ============================================================
# 8. JENSEN PWL APPROXIMATION
# ============================================================

distance_points = np.linspace(
    0.0, 6000.0, 200
)

factor_points = (
    ROTOR_RADIUS
    /
    (ROTOR_RADIUS + KW * distance_points)
) ** 2


# ============================================================
# 9. VARIABLES FOR WAKE GEOMETRY
#    These depend on direction, not wind speed.
#    Therefore, we share them across all 15 speed bins.
# ============================================================

downstream = {}
inside_wake = {}
wake = {}

abs_crosswind = {}
downwind_distance = {}
wake_factor = {}


# ============================================================
# 10. SCENARIO-SPECIFIC VARIABLES
# ============================================================

deficit = {}
combined_deficit = {}
effective_speed = {}
turbine_power = {}


# ============================================================
# 11. BUILD WAKE GEOMETRY FOR 36 DIRECTIONS
# ============================================================

for d_idx, direction in enumerate(wind_directions):

    print(
        f"Building wake geometry: "
        f"direction {d_idx + 1}/36 ({direction} degrees)"
    )

    angle = np.radians(float(direction) - 90.0)

    cos_dir = round(float(np.cos(angle)), 12)
    sin_dir = round(float(np.sin(angle)), 12)

    # Rotate the shared turbine coordinates
    x_rot = {}
    y_rot = {}

    for i in range(N):

        x_rot[i] = cos_dir * x[i] - sin_dir * y[i]
        y_rot[i] = sin_dir * x[i] + cos_dir * y[i]

    # --------------------------------------------------------
    # Pairwise wake relationships
    # --------------------------------------------------------

    for i in range(N):

        for j in range(N):

            if i == j:
                continue

            dx = x_rot[i] - x_rot[j]
            dy = y_rot[i] - y_rot[j]

            # Intermediate signed variables
            dx_var = model.addVar(
                lb=-6000,
                ub=6000,
                name=f"dx_{d_idx}_{i}_{j}"
            )

            dy_var = model.addVar(
                lb=-6000,
                ub=6000,
                name=f"dy_{d_idx}_{i}_{j}"
            )

            model.addConstr(dx_var == dx)
            model.addConstr(dy_var == dy)

            # Absolute crosswind distance
            abs_crosswind[d_idx, i, j] = model.addVar(
                lb=0,
                ub=6000,
                name=f"abs_dy_{d_idx}_{i}_{j}"
            )

            model.addGenConstrAbs(
                abs_crosswind[d_idx, i, j],
                dy_var
            )

            # Downstream condition
            downstream[d_idx, i, j] = model.addVar(
                vtype=GRB.BINARY,
                name=f"downstream_{d_idx}_{i}_{j}"
            )

            model.addGenConstrIndicator(
                downstream[d_idx, i, j],
                1,
                dx_var >= EPS
            )

            model.addGenConstrIndicator(
                downstream[d_idx, i, j],
                0,
                dx_var <= 0
            )

            # Inside wake cone condition
            inside_wake[d_idx, i, j] = model.addVar(
                vtype=GRB.BINARY,
                name=f"inside_{d_idx}_{i}_{j}"
            )

            model.addGenConstrIndicator(
                inside_wake[d_idx, i, j],
                1,
                abs_crosswind[d_idx, i, j]
                <= ROTOR_RADIUS + KW * dx_var
            )

            model.addGenConstrIndicator(
                inside_wake[d_idx, i, j],
                0,
                abs_crosswind[d_idx, i, j]
                >= ROTOR_RADIUS + KW * dx_var + EPS
            )

            # Wake = downstream AND inside wake cone
            wake[d_idx, i, j] = model.addVar(
                vtype=GRB.BINARY,
                name=f"wake_{d_idx}_{i}_{j}"
            )

            model.addConstr(
                wake[d_idx, i, j]
                <= downstream[d_idx, i, j]
            )

            model.addConstr(
                wake[d_idx, i, j]
                <= inside_wake[d_idx, i, j]
            )

            model.addConstr(
                wake[d_idx, i, j]
                >= downstream[d_idx, i, j]
                + inside_wake[d_idx, i, j] - 1
            )

            # Positive downwind distance
            downwind_distance[d_idx, i, j] = model.addVar(
                lb=0,
                ub=6000,
                name=f"distance_{d_idx}_{i}_{j}"
            )

            model.addGenConstrMax(
                downwind_distance[d_idx, i, j],
                [dx_var],
                constant=0.0
            )

            # Jensen distance factor
            wake_factor[d_idx, i, j] = model.addVar(
                lb=0,
                ub=1,
                name=f"factor_{d_idx}_{i}_{j}"
            )

            model.addGenConstrPWL(
                downwind_distance[d_idx, i, j],
                wake_factor[d_idx, i, j],
                distance_points.tolist(),
                factor_points.tolist()
            )


# ============================================================
# 12. BUILD ALL 540 WIND SCENARIOS
# ============================================================

for d_idx in range(N_DIRECTIONS):

    for s_idx in range(N_SPEEDS):

        scenario = d_idx * N_SPEEDS + s_idx

        speed = float(wind_speeds[s_idx])

        coefficient = float(
            jensen_coefficients[s_idx]
        )

        probability = float(
            wind_probability[scenario]
        )

        print(
            f"Building scenario {scenario + 1}/540: "
            f"direction={wind_directions[d_idx]}, "
            f"speed={speed}"
        )

        # ----------------------------------------------------
        # 12A. Individual wake deficits
        # ----------------------------------------------------

        for i in range(N):

            for j in range(N):

                if i == j:
                    continue

                deficit[d_idx, s_idx, i, j] = model.addVar(
                    lb=0,
                    ub=coefficient,
                    name=f"deficit_{d_idx}_{s_idx}_{i}_{j}"
                )

                model.addGenConstrIndicator(
                    wake[d_idx, i, j],
                    1,
                    deficit[d_idx, s_idx, i, j]
                    ==
                    coefficient * wake_factor[d_idx, i, j]
                )

                model.addGenConstrIndicator(
                    wake[d_idx, i, j],
                    0,
                    deficit[d_idx, s_idx, i, j] == 0
                )

        # ----------------------------------------------------
        # 12B. RSS aggregation
        # ----------------------------------------------------

        for i in range(N):

            combined_deficit[d_idx, s_idx, i] = model.addVar(
                lb=0,
                ub=1,
                name=f"combined_{d_idx}_{s_idx}_{i}"
            )

            model.addQConstr(
                combined_deficit[d_idx, s_idx, i] ** 2
                ==
                gp.quicksum(
                    deficit[d_idx, s_idx, i, j] ** 2
                    for j in range(N)
                    if j != i
                ),
                name=f"rss_{d_idx}_{s_idx}_{i}"
            )

        # ----------------------------------------------------
        # 12C. Effective wind speed
        # ----------------------------------------------------

        for i in range(N):

            effective_speed[d_idx, s_idx, i] = model.addVar(
                lb=0,
                ub=speed,
                name=f"effective_speed_{d_idx}_{s_idx}_{i}"
            )

            model.addConstr(
                effective_speed[d_idx, s_idx, i]
                ==
                speed * (
                    1 - combined_deficit[d_idx, s_idx, i]
                )
            )

        # ----------------------------------------------------
        # 12D. Power curve approximation
        # ----------------------------------------------------

        for i in range(N):

            turbine_power[d_idx, s_idx, i] = model.addVar(
                lb=0,
                ub=float(np.max(power_values)),
                name=f"power_{d_idx}_{s_idx}_{i}"
            )

            model.addGenConstrPWL(
                effective_speed[d_idx, s_idx, i],
                turbine_power[d_idx, s_idx, i],
                power_speeds.tolist(),
                power_values.tolist()
            )


# ============================================================
# 13. OBJECTIVE: MAXIMIZE ANNUAL ENERGY PRODUCTION
# ============================================================

expected_power = gp.quicksum(
    float(wind_probability[d_idx * N_SPEEDS + s_idx])
    *
    gp.quicksum(
        turbine_power[d_idx, s_idx, i]
        for i in range(N)
    )
    for d_idx in range(N_DIRECTIONS)
    for s_idx in range(N_SPEEDS)
)

# Power is MW, time is 8760 hours/year.
# Divide by 1000 to convert MWh to GWh.

model.setObjective(
    (8760.0 / 1000.0) * expected_power,
    GRB.MAXIMIZE
)


# ============================================================
# 14. OPTIMIZATION
# ============================================================

print("\n========================================")
print("STARTING 5-TURBINE FULL-SCENARIO MODEL")
print("========================================")

print("Turbines:", N)
print("Directions:", N_DIRECTIONS)
print("Wind speeds:", N_SPEEDS)
print("Scenarios:", N_SCENARIOS)
print("Objective: Expected annual energy production (GWh)")

model.optimize()


# ============================================================
# 15. SAVE OPTIMIZED LAYOUT
# ============================================================

if model.SolCount > 0:

    optimized_layout = pd.DataFrame({
        "x": [x[i].X for i in range(N)],
        "y": [y[i].X for i in range(N)]
    })

    optimized_layout.to_csv(
        "5_turbine_full_scenarios_layout.csv",
        index=False
    )

    # Verify geometric constraints
    coordinates = optimized_layout[["x", "y"]].to_numpy()

    pair_distances = np.sqrt(
        (
            coordinates[:, 0, None]
            - coordinates[None, :, 0]
        ) ** 2
        +
        (
            coordinates[:, 1, None]
            - coordinates[None, :, 1]
        ) ** 2
    )

    np.fill_diagonal(pair_distances, np.inf)

    print("\n========================================")
    print("OPTIMIZATION RESULTS")
    print("========================================")

    print("Gurobi status:", model.Status)
    print("Best AEP:", model.ObjVal, "GWh")
    print("Best bound:", model.ObjBound, "GWh")
    print("MIP gap:", model.MIPGap)
    print("Runtime:", model.Runtime, "seconds")

    print("\nMinimum turbine separation:",
          pair_distances.min(), "m")

    print("\nOptimized turbine coordinates:")
    print(optimized_layout.to_string(index=False))

    print(
        "\nSaved: 5_turbine_full_scenarios_layout.csv"
    )

else:

    print("\nNo feasible solution found.")
    print("Gurobi status:", model.Status)
