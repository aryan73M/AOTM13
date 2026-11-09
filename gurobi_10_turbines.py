
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
WIND_DIRECTION = 360.0

CT = 0.297620
EPS = 1e-5

WAKE_COEFFICIENT = 1 - np.sqrt(1 - CT)


# ============================================================
# 2. POWER CURVE
# ============================================================

power_curve = loadPowerCurve(
    "Shell_Hackathon_Dataset/power_curve.csv"
)

power_speeds = power_curve[:, 0].astype(float)
power_values = power_curve[:, 2].astype(float)


# ============================================================
# 3. WIND ROTATION
# ============================================================

wind_angle = np.radians(WIND_DIRECTION - 90.0)

cos_dir = np.cos(wind_angle)
sin_dir = np.sin(wind_angle)


# ============================================================
# 4. MODEL
# ============================================================

model = gp.Model("Wind_Farm_10_Turbines")

model.Params.NonConvex = 2
model.Params.TimeLimit = 300
model.Params.MIPGap = 0.001
model.Params.MIPFocus = 1


# ============================================================
# 5. COORDINATE VARIABLES
# ============================================================

x = model.addVars(
    N, lb=BOUNDARY, ub=FARM_SIZE - BOUNDARY,
    name="x"
)

y = model.addVars(
    N, lb=BOUNDARY, ub=FARM_SIZE - BOUNDARY,
    name="y"
)


# ============================================================
# 6. INITIAL LAYOUT
# ============================================================

# A 5 x 2 initial grid

for i in range(N):
    row = i // 5
    col = i % 5

    x[i].Start = 200 + col * 800
    y[i].Start = 200 + row * 800


# ============================================================
# 7. MINIMUM DISTANCE
# ============================================================

for i in range(N):
    for j in range(i + 1, N):

        model.addQConstr(
            (x[i] - x[j])**2 +
            (y[i] - y[j])**2 >= MIN_DISTANCE**2
        )


# ============================================================
# 8. ROTATED COORDINATES
# ============================================================

x_rot = {}
y_rot = {}

for i in range(N):

    x_rot[i] = cos_dir*x[i] - sin_dir*y[i]
    y_rot[i] = sin_dir*x[i] + cos_dir*y[i]


# ============================================================
# 9. WAKE VARIABLES
# ============================================================

wake = {}
downstream = {}
inside_wake = {}

abs_crosswind = {}
downwind_distance = {}

wake_factor = {}
deficit = {}

distance_points = np.linspace(0, 6000, 200)

factor_points = (
    ROTOR_RADIUS /
    (ROTOR_RADIUS + KW*distance_points)
)**2


for i in range(N):

    for j in range(N):

        if i == j:
            continue

        dx = x_rot[i] - x_rot[j]
        dy = y_rot[i] - y_rot[j]

        # Signed crosswind variable
        dy_var = model.addVar(
            lb=-6000, ub=6000,
            name=f"dy_{i}_{j}"
        )

        model.addConstr(dy_var == dy)

        # Absolute crosswind distance
        abs_crosswind[i, j] = model.addVar(
            lb=0, ub=6000,
            name=f"abs_dy_{i}_{j}"
        )

        model.addGenConstrAbs(
            abs_crosswind[i, j],
            dy_var
        )

        # Signed downwind variable
        dx_var = model.addVar(
            lb=-6000, ub=6000,
            name=f"dx_{i}_{j}"
        )

        model.addConstr(dx_var == dx)

        # Downstream binary
        downstream[i, j] = model.addVar(
            vtype=GRB.BINARY,
            name=f"downstream_{i}_{j}"
        )

        model.addGenConstrIndicator(
            downstream[i, j], 1,
            dx_var >= EPS
        )

        model.addGenConstrIndicator(
            downstream[i, j], 0,
            dx_var <= 0
        )

        # Inside wake cone binary
        inside_wake[i, j] = model.addVar(
            vtype=GRB.BINARY,
            name=f"inside_{i}_{j}"
        )

        model.addGenConstrIndicator(
            inside_wake[i, j], 1,
            abs_crosswind[i, j]
            <= ROTOR_RADIUS + KW*dx_var
        )

        model.addGenConstrIndicator(
            inside_wake[i, j], 0,
            abs_crosswind[i, j]
            >= ROTOR_RADIUS + KW*dx_var + EPS
        )

        # Wake interaction = AND
        wake[i, j] = model.addVar(
            vtype=GRB.BINARY,
            name=f"wake_{i}_{j}"
        )

        model.addConstr(wake[i, j] <= downstream[i, j])
        model.addConstr(wake[i, j] <= inside_wake[i, j])

        model.addConstr(
            wake[i, j] >=
            downstream[i, j] + inside_wake[i, j] - 1
        )

        # Positive downwind distance
        downwind_distance[i, j] = model.addVar(
            lb=0, ub=6000,
            name=f"distance_{i}_{j}"
        )

        model.addGenConstrMax(
            downwind_distance[i, j],
            [dx_var],
            constant=0
        )

        # Jensen wake factor
        wake_factor[i, j] = model.addVar(
            lb=0, ub=1,
            name=f"factor_{i}_{j}"
        )

        model.addGenConstrPWL(
            downwind_distance[i, j],
            wake_factor[i, j],
            distance_points.tolist(),
            factor_points.tolist()
        )

        # Individual wake deficit
        deficit[i, j] = model.addVar(
            lb=0, ub=WAKE_COEFFICIENT,
            name=f"deficit_{i}_{j}"
        )

        model.addGenConstrIndicator(
            wake[i, j], 1,
            deficit[i, j] ==
            WAKE_COEFFICIENT*wake_factor[i, j]
        )

        model.addGenConstrIndicator(
            wake[i, j], 0,
            deficit[i, j] == 0
        )


# ============================================================
# 10. RSS WAKE AGGREGATION
# ============================================================

combined_deficit = model.addVars(
    N, lb=0, ub=1,
    name="combined_deficit"
)

for i in range(N):

    model.addQConstr(
        combined_deficit[i]**2 ==
        gp.quicksum(
            deficit[i, j]**2
            for j in range(N)
            if i != j
        )
    )


# ============================================================
# 11. EFFECTIVE WIND SPEED
# ============================================================

effective_speed = model.addVars(
    N, lb=0, ub=WIND_SPEED,
    name="effective_speed"
)

for i in range(N):

    model.addConstr(
        effective_speed[i] ==
        WIND_SPEED*(1 - combined_deficit[i])
    )


# ============================================================
# 12. POWER CURVE
# ============================================================

turbine_power = model.addVars(
    N,
    lb=0,
    ub=float(np.max(power_values)),
    name="power"
)

for i in range(N):

    model.addGenConstrPWL(
        effective_speed[i],
        turbine_power[i],
        power_speeds.tolist(),
        power_values.tolist()
    )


# ============================================================
# 13. OBJECTIVE
# ============================================================

model.setObjective(
    gp.quicksum(turbine_power[i] for i in range(N)),
    GRB.MAXIMIZE
)


# ============================================================
# 14. OPTIMIZATION
# ============================================================

print("\nOptimizing 10-turbine wind farm...")
model.optimize()


# ============================================================
# 15. RESULTS
# ============================================================

if model.SolCount > 0:

    layout = pd.DataFrame({
        "x": [x[i].X for i in range(N)],
        "y": [y[i].X for i in range(N)]
    })

    layout.to_csv(
        "10_turbine_layout.csv",
        index=False
    )

    print("\nOptimization completed.")
    print("Status:", model.Status)
    print("Farm power:", model.ObjVal, "MW")
    print("Best bound:", model.ObjBound)
    print("MIP gap:", model.MIPGap)

    print("\nOptimized coordinates:")
    print(layout.to_string(index=False))

    print("\nSaved: 10_turbine_layout.csv")

else:
    print("No feasible solution found.")
