
import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from AEP_calculator import loadPowerCurve


# ============================================================
# 1. PARAMETERS
# ============================================================

N = 10

FARM_SIZE = 4000.0
BOUNDARY = 50.0
MIN_DISTANCE = 400.0

ROTOR_RADIUS = 50.0
KW = 0.05

WIND_SPEED = 15.0

# Official evaluator direction convention
WIND_DIRECTIONS = [360.0] + [
    float(d) for d in range(10, 360, 10)
]

CT = 0.297620
EPS = 1e-5

WAKE_COEFFICIENT = 1.0 - np.sqrt(1.0 - CT)

TIME_LIMIT = 600
MIP_GAP = 0.001


# ============================================================
# 2. LOAD OFFICIAL POWER CURVE
# ============================================================

power_curve = loadPowerCurve(
    "Shell_Hackathon_Dataset/power_curve.csv"
)

power_speeds = power_curve[:, 0].astype(float)
power_values = power_curve[:, 2].astype(float)

assert np.all(np.diff(power_speeds) > 0)


# ============================================================
# 3. CREATE MODEL
# ============================================================

model = gp.Model("Wind_Farm_10_Turbines_36_Directions")

model.Params.NonConvex = 2
model.Params.TimeLimit = TIME_LIMIT
model.Params.MIPGap = MIP_GAP
model.Params.MIPFocus = 1
model.Params.OutputFlag = 1


# ============================================================
# 4. SHARED TURBINE COORDINATES
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
# 5. INITIAL SOLUTION
# ============================================================

# Reuse the optimized 10-turbine single-direction layout
# if it exists. Otherwise, use a 5 x 2 grid.

if os.path.exists("10_turbine_layout.csv"):

    initial_layout = pd.read_csv("10_turbine_layout.csv")

    if len(initial_layout) == N:
        initial_x = initial_layout["x"].to_numpy()
        initial_y = initial_layout["y"].to_numpy()
    else:
        raise ValueError(
            "10_turbine_layout.csv must contain exactly 10 turbines."
        )

else:

    initial_x = np.array([
        200 + (i % 5) * 800 for i in range(N)
    ], dtype=float)

    initial_y = np.array([
        200 + (i // 5) * 800 for i in range(N)
    ], dtype=float)


for i in range(N):
    x[i].Start = float(initial_x[i])
    y[i].Start = float(initial_y[i])


# ============================================================
# 6. MINIMUM DISTANCE CONSTRAINTS
# ============================================================

for i in range(N):
    for j in range(i + 1, N):

        model.addQConstr(
            (x[i] - x[j]) ** 2
            +
            (y[i] - y[j]) ** 2
            >= MIN_DISTANCE ** 2,
            name=f"minimum_distance_{i}_{j}"
        )


# ============================================================
# 7. JENSEN PWL DATA
# ============================================================

distance_points = np.linspace(0.0, 6000.0, 200)

factor_points = (
    ROTOR_RADIUS /
    (ROTOR_RADIUS + KW * distance_points)
) ** 2


# ============================================================
# 8. SCENARIO-SPECIFIC WAKE VARIABLES
# ============================================================

wake = {}
downstream = {}
inside_wake = {}

abs_crosswind = {}
downwind_distance = {}

wake_factor = {}
deficit = {}

combined_deficit = {}
effective_speed = {}
turbine_power = {}


for d_idx, wind_direction in enumerate(WIND_DIRECTIONS):

    print(
        f"Building scenario {d_idx + 1}/36: "
        f"direction = {wind_direction}"
    )

    # Round trigonometric coefficients to eliminate
    # insignificant floating-point coefficients.

    wind_angle = np.radians(wind_direction - 90.0)

    cos_dir = round(float(np.cos(wind_angle)), 12)
    sin_dir = round(float(np.sin(wind_angle)), 12)

    # Rotated coordinates for this direction
    x_rot = {}
    y_rot = {}

    for i in range(N):

        x_rot[i] = cos_dir * x[i] - sin_dir * y[i]
        y_rot[i] = sin_dir * x[i] + cos_dir * y[i]


    # --------------------------------------------------------
    # 8A. PAIRWISE WAKE INTERACTIONS
    # --------------------------------------------------------

    for i in range(N):

        for j in range(N):

            if i == j:
                continue

            dx = x_rot[i] - x_rot[j]
            dy = y_rot[i] - y_rot[j]

            # Intermediate variables
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

            # Downstream binary
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

            # Inside Jensen wake cone binary
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

            # Individual wake deficit
            deficit[d_idx, i, j] = model.addVar(
                lb=0,
                ub=WAKE_COEFFICIENT,
                name=f"deficit_{d_idx}_{i}_{j}"
            )

            model.addGenConstrIndicator(
                wake[d_idx, i, j],
                1,
                deficit[d_idx, i, j]
                == WAKE_COEFFICIENT
                * wake_factor[d_idx, i, j]
            )

            model.addGenConstrIndicator(
                wake[d_idx, i, j],
                0,
                deficit[d_idx, i, j] == 0
            )


    # --------------------------------------------------------
    # 8B. RSS AGGREGATION
    # --------------------------------------------------------

    for i in range(N):

        combined_deficit[d_idx, i] = model.addVar(
            lb=0,
            ub=1,
            name=f"combined_{d_idx}_{i}"
        )

        model.addQConstr(
            combined_deficit[d_idx, i] ** 2
            ==
            gp.quicksum(
                deficit[d_idx, i, j] ** 2
                for j in range(N)
                if i != j
            ),
            name=f"rss_{d_idx}_{i}"
        )


    # --------------------------------------------------------
    # 8C. EFFECTIVE WIND SPEED
    # --------------------------------------------------------

    for i in range(N):

        effective_speed[d_idx, i] = model.addVar(
            lb=0,
            ub=WIND_SPEED,
            name=f"effective_speed_{d_idx}_{i}"
        )

        model.addConstr(
            effective_speed[d_idx, i]
            ==
            WIND_SPEED
            * (1 - combined_deficit[d_idx, i])
        )


    # --------------------------------------------------------
    # 8D. POWER CURVE
    # --------------------------------------------------------

    for i in range(N):

        turbine_power[d_idx, i] = model.addVar(
            lb=0,
            ub=float(np.max(power_values)),
            name=f"power_{d_idx}_{i}"
        )

        model.addGenConstrPWL(
            effective_speed[d_idx, i],
            turbine_power[d_idx, i],
            power_speeds.tolist(),
            power_values.tolist()
        )


# ============================================================
# 9. OBJECTIVE: MAXIMIZE AVERAGE FARM POWER
# ============================================================

total_power = gp.quicksum(
    turbine_power[d_idx, i]
    for d_idx in range(len(WIND_DIRECTIONS))
    for i in range(N)
)

model.setObjective(
    total_power / len(WIND_DIRECTIONS),
    GRB.MAXIMIZE
)


# ============================================================
# 10. OPTIMIZE
# ============================================================

print("\nStarting 10-turbine, 36-direction optimization.")

print("Number of turbines:", N)
print("Number of directions:", len(WIND_DIRECTIONS))
print("Wind speed:", WIND_SPEED, "m/s")

model.optimize()


# ============================================================
# 11. SAVE RESULTS
# ============================================================

if model.SolCount > 0:

    optimized_layout = pd.DataFrame({
        "x": [x[i].X for i in range(N)],
        "y": [y[i].X for i in range(N)]
    })

    optimized_layout.to_csv(
        "10_turbine_36_direction_layout.csv",
        index=False
    )

    print("\n========== OPTIMIZATION RESULTS ==========")

    print("Gurobi status:", model.Status)
    print("Average farm power:", model.ObjVal, "MW")
    print("Best bound:", model.ObjBound, "MW")
    print("MIP gap:", model.MIPGap)
    print("Runtime:", model.Runtime, "seconds")

    print("\nOptimized coordinates:")
    print(optimized_layout.to_string(index=False))

    print("\nFarm power by wind direction:")

    for d_idx, direction in enumerate(WIND_DIRECTIONS):

        scenario_power = sum(
            turbine_power[d_idx, i].X
            for i in range(N)
        )

        print(
            f"{direction:6.1f} degrees: "
            f"{scenario_power:.6f} MW"
        )

    print(
        "\nLayout saved as "
        "10_turbine_36_direction_layout.csv"
    )

else:

    print("\nNo feasible solution found.")
    print("Gurobi status:", model.Status)
