"""
parameters.py
=============
All model inputs, corridor definitions, and assumptions for the SBB MehrSpur
Zürich–Winterthur infrastructure analysis.

HOW TO USE
----------
Import this module wherever you need access to assumptions or corridor specs:

    from parameters import CORRIDOR_REGIONS, N_YEARS, DISCOUNT_RATE, ...

Students modifying the case study should edit values in this file and stages.py.
"""

import pandas as pd

# =============================================================================
# 1. GENERAL PLANNING & UNCERTAINTY PARAMETERS
# =============================================================================

N_YEARS           = 40      # planning horizon (years)
DEMAND_GROWTH_Y40 = 0.3    # nominal total demand growth by Year 40 (+33%)

# Trajectory uncertainty sigmas
DG_SIGMA_1    = 0.005   # demand-growth trajectory: sigma in Year 1
DG_SIGMA_40   = 0.30    # demand-growth trajectory: sigma in Year 40

# Project Acceptability / Success Thresholds
MAX_AVG_TT      = 20   # max acceptable network average motorized travel time
PT_SHARE_TARGET = 0.35    # target PT mode share (35%)


# =============================================================================
# 2. CORRIDOR DEFINITIONS (Zürich–Winterthur / MehrSpur)
# =============================================================================

CORRIDOR_REGIONS = {
    "Zürich": ["Zürich"],
    "Winterthur": ["Winterthur"],
    "Airport / Glattal": [
        "Kloten", "Opfikon", "Wallisellen", "Dübendorf",
        "Dietlikon", "Wangen-Brüttisellen", "Bassersdorf", "Rümlang",
        "Illnau-Effretikon", "Lindau", "Nürensdorf"
    ],
    "Eastern Switzerland (Gateways)": [
        "Wiesendangen", "Elsau",
        "Elgg", "Hagenbuch"
    ]
}

# Flat list of all corridor municipalities
CORRIDOR_MUNICIPALITIES = [
    muni for munis in CORRIDOR_REGIONS.values() for muni in munis
]

# Municipalities defining the primary inner rail corridor (Zürich <-> Winterthur)
PRIMARY_RAIL_CORRIDOR = {"Zürich", "Winterthur"}


# =============================================================================
# 3. ECONOMIC & ENVIRONMENTAL VALUATION PARAMETERS (Uncertainties)
# =============================================================================
# ⚠️ IMPORTANT: Add all new economic or cost parameters to this dictionary!
# They will be automatically picked up by the Monte Carlo uncertainty model.



PERTURBABLE_PARAMS = {
    "DISCOUNT_RATE": 0.02,    # social discount rate (2.0%)
    
    # Car costs & emissions
    "F_FUEL": 0.04,     # fuel consumption            (kg fuel / km)
    "C_FUEL": 1.25,     # fuel price                  (CHF / kg)
    "P_CO2": 0.139,    # CO2 emission factor          (kg CO2 / km)
    "C_CO2": 0.0417,   # carbon cost, base year       (CHF / kg CO2)
    "C_CO2_GROWTH": 0.0026,   # carbon cost annual increase  (CHF / kg CO2 / year)
    "C_TT_CAR": 23.3,     # value of travel time – car   (CHF / h)
    
    # Public Transport (PT) valuation
    "C_FARE": 3.50,     # average single trip fare     (CHF / trip)
    "C_TT_PT": 14.4,     # value of in-vehicle PT TT    (CHF / h)

    # -------------------------------------------------------------------------
    # 4. INFRASTRUCTURE & INTERVENTION COSTS (MehrSpur)
    # -------------------------------------------------------------------------
    # Stage 1: Local Stations & Access Package (A3 Dietlikon + A4 Bassersdorf + A5 Wallisellen)
    # 444.3M (A3) + 201.4M (A4) + 279.3M (A5) = 925.0 MCHF
    "C_INV_STAGE1": 925_000_000,  # capital investment for stations A3+A4+A5 (CHF)
    "C_OP_STAGE1": 27_750_000,   # annual operating cost (3% of investment)

    # Stage 2: Core Tunnel & Winterthur Package (A0 Gesamt + A1 Winterthur + A2 Tunnel)
    # 145.6M (A0) + 846.3M (A1) + 1'310.7M (A2) = 2,302.6 MCHF
    "C_INV_STAGE2": 2_302_600_000, # capital investment cost (CHF, A0+A1+A2)
    "C_OP_STAGE2": 69_078_000,    # annual operating & maintenance cost (3% of investment)

    # Real Options / Flexibility Premium (for pre-engineering or reservation)
    "C_FLEX": 30_000_000,   # upfront flexibility option premium (CHF)
}

# =============================================================================
# 5. FIXED PHYSICAL / ENGINEERING DESIGN FACTS (Held constant)
# =============================================================================

FIXED_PARAMS = {
    "MAX_AVG_TT": MAX_AVG_TT,
    "PT_SHARE_TARGET": PT_SHARE_TARGET,
    "PEAK_TO_ANNUAL": 1200.0, # Multiplier to scale evening peak hour to annual (e.g., 300 days * 4 peak hours)
    
    # -------------------------------------------------------------------------
    # PT Capacity & Headways 
    # (Note: Defined here rather than in stages.py so students can easily 
    # move them to PERTURBABLE_PARAMS to test infrastructure underperformance!)
    # -------------------------------------------------------------------------
    "HEADWAY_STAGE0": 20.0,    # Baseline frequency (min)
    "HEADWAY_STAGE1": 18.0,    # Station Package frequency (min)
    "HEADWAY_STAGE2": 15.0,    # Core Tunnel enables 15-min rhythm
    "CAPACITY_STAGE0": 85_000, # Peak-hour trips capacity
    "CAPACITY_STAGE1": 100_000, # Marginal node relief
    "CAPACITY_STAGE2": 110_000, # +30% capacity

    # NIBA External Costs (Road)
    "C_NOISE_CAR": 0.015,   # CHF / Fzkm
    "C_AIR_CAR": 0.018,   # CHF / Fzkm
    "C_ACCIDENT_CAR": 0.084,   # CHF / Fzkm
    
    "STAGE2_YEAR": 15,           # trigger year in fixed staged strategy
}

# Dynamically unpack the dictionaries so they act like global variables
locals().update(PERTURBABLE_PARAMS)
locals().update(FIXED_PARAMS)


# =============================================================================
# 5. FIXED & UNCERTAIN PARAMETER REGISTRIES (for EMA Workbench / Modular CBA)
# =============================================================================

# 1. Structural Trajectory Uncertainties (Deep 40-year macro and behavioral trends)
STRUCTURAL_UNCERTAINTIES = {
    "u_demand": {
        "name": "Demand Growth Trajectory",
        "description": "Cumulative travel demand growth over 40 years",
        "nominal_start": 0.0,
        "nominal_end": DEMAND_GROWTH_Y40,
        "sigma_start": DG_SIGMA_1,
        "sigma_end": DG_SIGMA_40,
        "floor": -0.80,
    },
    "u_beta_pt": {
        "name": "PT Preference Trajectory",
        "description": "Societal & behavioral adoption of transit (multiplier)",
        "nominal_start": 1.00,
        "nominal_end": 1.20,
        "sigma_start": 0.01,
        "sigma_end": 0.25,
        "floor": 0.20,
    },
}

# (FIXED_PARAMS and PERTURBABLE_PARAMS are now defined in section 3 & 5 above)
PERTURBABLE_SD_FRACTION = 0.10   # 10% standard deviation for nuisance uncertainties

# Combined nominal parameters dictionary (deterministic baseline)
NOMINAL_PARAMS = {**FIXED_PARAMS, **PERTURBABLE_PARAMS}

# Module-level cost shortcuts
C_INV_STAGE1 = NOMINAL_PARAMS["C_INV_STAGE1"]
C_OP_STAGE1  = NOMINAL_PARAMS["C_OP_STAGE1"]
C_INV_STAGE2 = NOMINAL_PARAMS["C_INV_STAGE2"]
C_OP_STAGE2  = NOMINAL_PARAMS["C_OP_STAGE2"]
C_FLEX       = NOMINAL_PARAMS["C_FLEX"]

# Default base demand for reference
D0_TOTAL = 150_000 * 365  # nominal corridor annual passenger trips


def validate_params():
    """Ensure that parameters modified by students remain within physically and economically sound bounds."""
    if not isinstance(N_YEARS, int) or not (10 <= N_YEARS <= 100):
        raise ValueError(f"N_YEARS must be an integer between 10 and 100. Got {N_YEARS}.")
    
    if not isinstance(DEMAND_GROWTH_Y40, (int, float)) or not (-1.0 <= DEMAND_GROWTH_Y40 <= 5.0):
        raise ValueError(f"DEMAND_GROWTH_Y40 must be a float (e.g. 0.30 for +30%). Got {DEMAND_GROWTH_Y40}.")
        
    discount = PERTURBABLE_PARAMS.get("DISCOUNT_RATE", 0.0)
    if not isinstance(discount, (int, float)) or not (0.0 <= discount <= 0.20):
        raise ValueError(f"DISCOUNT_RATE must be between 0.0 and 0.20 (e.g. 0.02 for 2%). Got {discount}.")
        
    max_tt = MAX_AVG_TT
    if not isinstance(max_tt, (int, float)) or not (5.0 <= max_tt <= 60.0):
        raise ValueError(f"MAX_AVG_TT must be between 5.0 and 60.0 minutes. Got {max_tt}.")
        
    for k, v in PERTURBABLE_PARAMS.items():
        if not isinstance(v, (int, float)):
            raise ValueError(f"Parameter '{k}' must be a numeric value. Got {type(v).__name__}: {v}")
            
    for k, v in FIXED_PARAMS.items():
        if not isinstance(v, (int, float)):
            raise ValueError(f"Parameter '{k}' must be a numeric value. Got {type(v).__name__}: {v}")

validate_params()
