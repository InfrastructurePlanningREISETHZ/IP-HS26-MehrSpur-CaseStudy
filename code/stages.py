"""
stages.py
=========
Infrastructure stage specifications for the SBB MehrSpur Zürich–Winterthur project.

A 'stage' captures the physical infrastructure and operational interventions
that modify the multimodal transport network and skims in the Canton Zürich FSM model.

HOW TO USE
----------
This file is the single source of truth for all intervention parameters. You can customize the percentage improvements, add new mode choice parameters (ASCs/betas), or alter mode affinities directly inside each stage block below.

--- CHEAT SHEET: PARAMETERS & INTERVENTIONS ---

1. Global Mode Parameters (Utility Multipliers)
   Brief description: High-level multipliers to shift overall mode attractiveness.
   - car_affinity, pt_affinity, bike_affinity, walk_affinity (default 1.0): Scales the overall utility for a mode.
   - ebike_share (e.g. 0.1): Proportion of bicycles that are e-bikes.
   - EBIKE_SPEED_MULTIPLIER (e.g. 1.5): Speed advantage of e-bikes over acoustic bikes.

2. Mode Choice Constants (ASCs & Betas)
   Brief description: Core parameters for the discrete choice (Logit) model.
   - ASCs (ASC_CAR, ASC_WALK, ASC_BIKE, ASC_PT_WALK, ASC_PT_BIKE): Base preference for a mode (independent of time/cost).
   - Betas (Time) (B_CAR_TIME, B_WALK_TIME, B_BIKE_TIME, B_PT_IVT, B_PT_OVT, B_TRANSFER): Utility penalty per minute spent traveling.
   - Betas (Cost) (B_COST, B_PT_FARE): Utility penalty per CHF spent.
   - Fares (CAR_COST_CHF_PER_KM, PT_MIN_FARE_CHF, PT_FARE_CHF_PER_KM, PT_MAX_FARE_CHF): Monetary cost parameters.

3. Defining Spatial Scopes
   Interventions are targeted geographically using spatial filters:
   
   - "area_pairs": Used for corridor / link-based interventions between origins and destinations.
     Filter by "municipality_name" (e.g. "Dietikon") or "city_quartier" (e.g. "Altstetten").
     Example: Reducing rail travel time between Zürich and Winterthur:
       "railway_expansions": [
           {
               "name": "Corridor Rail Upgrade",
               "area_pairs": [
                   {"origin": {"municipality_name": "Zürich"}, "destination": {"municipality_name": "Winterthur"}}
               ],
               "both_directions": True,
               "effects": {
                   "travel_time_reduction_pct": 15.0
               }
           }
       ]
   
   - "zones": Used for area-wide or zone-based interventions (e.g., station access, local networks).
     Filter by specific zone IDs or whole municipality names.
     Example: Improving walk access & egress times in Dietlikon:
       "mobility_hubs": [
           {
               "name": "Dietlikon Hub Area",
               "zones": [{"municipality_name": "Dietlikon"}],  # or zone IDs: ["15401012"]
               "effects": {
                   "access_time_reduction_pct": 20.0,
                   "egress_time_reduction_pct": 20.0
               }
           }
       ]


4. Mode-Specific Physical Interventions
   Effects are defined under specific mode keys which target different transport networks:
   - "railway_expansions": Modifies the Public Transport (PT) network.
   - "bike_highways": Modifies the Bicycle network.
   - "road_capacity": Modifies the Car (road) network.
   - "mobility_hubs": Modifies access/transfer times at specific PT stations (impacts walking to/from PT).

   Within these keys, you can apply effects like:
   - "travel_time_reduction_pct" (Reduces in-vehicle travel time)
   - "speed_increase_pct" (Increases average speed)
   - "distance_reduction_pct" (Reduces trip distance, e.g., a new tunnel or bridge)
   - "access_time_reduction_pct", "egress_time_reduction_pct" (Reduces walk time to/from PT)
   
   EXAMPLE 1: Car Intervention (Increasing road speed by 10%)
   "road_capacity": [
       {
           "area_pairs": [{"origin": {"municipality_name": "Zürich"}, "destination": {"municipality_name": "Winterthur"}}],
           "effects": {"speed_increase_pct": 10.0}
       }
   ]

   EXAMPLE 2: Walk Intervention (Reducing walk time to PT stations by 20%)
   "mobility_hubs": [
       {
           "zones": [{"municipality_name": "Dietlikon"}],
           "effects": {"access_time_reduction_pct": 20.0, "egress_time_reduction_pct": 20.0}
       }
   ]

   EXAMPLE 3: Bike Intervention (Shortening bike travel distance by 15% via a direct cycle path/bridge)
   "bike_highways": [
       {
           "area_pairs": [{"origin": {"city_quartier": "Altstetten"}, "destination": {"municipality_name": "Dietikon"}}],
           "both_directions": True,
           "effects": {
               "distance_reduction_pct": 15.0,
               "speed_increase_pct": 10.0
           }
       }
   ]
"""

from __future__ import annotations
import itertools
import parameters as p

def get_stages(params: dict | None = None) -> dict[int, dict]:
    """
    Build the STAGES specification dictionary from a parameter dict.
    
    Keys: integer stage numbers (0, 1, 2, ...)
    Values: dict defining the native FSM scenario interventions and mode choice overrides.
    """
    params = params or {}

    # --- DEFINE SPATIAL BOUNDARIES ---
    # We generate area pairs for all combinations of municipalities in the MehrSpur corridor
    mehrspur_corridor = [
        {"origin": {"municipality_name": o}, "destination": {"municipality_name": d}}
        for o, d in itertools.combinations(p.CORRIDOR_MUNICIPALITIES, 2)
    ]
    min_dist_km = 5.0

    stages = {

        # ── Stage 0 ───────────────────────────────────────────────────────────
        # Baseline network (no new infrastructure)
        0: {
            "name": "Stage 0 – Baseline Network",
            
            # --- 1. Global Affinities & Technology Shares ---
            "car_affinity": 1.0,
            "pt_affinity": 1.0,
            "bike_affinity": 1.0,
            "walk_affinity": 1.0,
            "ebike_share": 0.10,
            "EBIKE_SPEED_MULTIPLIER": 1.5,
        },

        # ── Stage 1 ───────────────────────────────────────────────────────────
   

        #Official SBB MehrSpur Scope (Opening 2033 to 2035 | CAPEX: CHF 925.0 M):

        #       A3 Dietlikon (CHF 444.3 M): 4-track expansion & grade-separated flyover (Entflechtung), 
        #       eliminating cross-track bottlenecks towards the airport and Winterthur.

        #       A4 Bassersdorf (CHF 201.4 M): Station modernization, portal approach tracks, and multimodal 
        #       hub integration (Velostation & P+R).

        #       A5 Wallisellen (CHF 279.3 M): Grade-separated junction towards Zurich HB/Airport, 
        #       barrier-free platform access, and improved bus feeder connectivity.
                
        # Transport & Economic Impact:
        #       Relieves surface bottlenecks before tunnel completion, stabilizing network timetables.
        #       Reduces station access/egress times by 20 to 30% via upgraded pedestrian and cycling infrastructure.
        #       Reduces passenger waiting & transfer times by unlocking 15-minute suburban feeder slots.


        1: {
            "name": "Stage 1 – Local Stations & Access Package",
            
            # --- 1. Global Affinities & Technology Shares ---
            "car_affinity": 1.0,
            "pt_affinity": 1.02,
            "bike_affinity": 1.0,  # Improved active mobility safety (Accident Benefit proxy)
            "walk_affinity": 1.0,  # Improved local road crossings
            "ebike_share": 0.11,
            "EBIKE_SPEED_MULTIPLIER": 1.5,

            # --- 2. Discrete Choice Overrides (ASCs & Betas) ---
            "ASC_PT_BIKE": -1.0,  # Secure Velostationen at hubs (up from -1.5)
            "B_TRANSFER": -0.32,  # Seamless step-free transfers (up from -0.43)
            # Optional overrides (uncomment to activate):
            # "ASC_CAR": 0.5,
            # "ASC_WALK": 2.4,
            # "ASC_BIKE": -0.1,
            # "ASC_PT_WALK": 0.0,
            # "B_CAR_TIME": -0.075,
            # "B_WALK_TIME": -0.10,
            # "B_BIKE_TIME": -0.10,
            # "B_PT_IVT": -0.045,
            # "B_PT_OVT": -0.08,
            # "B_COST": -0.18,
            # "B_PT_FARE": -0.18,
            # "CAR_COST_CHF_PER_KM": 0.27,
            # "PT_MIN_FARE_CHF": 3.0,
            # "PT_FARE_CHF_PER_KM": 0.08,
            # "PT_MAX_FARE_CHF": 8.0,

            # --- 3. Physical Network / Link Interventions ---
            "railway_expansions": [
                {
                    "name": "Zurich-Winterthur rail expansion - level 1",
                    "area_pairs": mehrspur_corridor,
                    "both_directions": True,
                    "effects": {
                        # In-Vehicle & Speed (Initial baseline improvements)
                        "travel_time_reduction_pct": 5.0,
                        "speed_increase_pct": 0.0,
                        "distance_reduction_pct": 0.0,
                        
                        # Public Transport Service & Stations
                        "initial_wait_reduction_pct": 20.0,
                        "transfer_wait_reduction_pct": 15.0,
                        "transfer_time_reduction_pct": 10.0,
                        "access_time_reduction_pct": 0.0,
                        "egress_time_reduction_pct": 0.0,
                    }
                }
            ],
            
            # --- 4. Hub & Node Interventions ---
            "mobility_hubs": [
                {
                    "name": "Station Upgrades (Dietlikon, Bassersdorf, Wallisellen)",
                    "zones": [
                        {"municipality_name": "Dietlikon"},
                        {"municipality_name": "Bassersdorf"},
                        {"municipality_name": "Wallisellen"}
                    ],
                    "effects": {
                        "access_time_reduction_pct": 25.0,  # Improved pedestrian ramps & bus loop access
                        "transfer_time_reduction_pct": 10.0, # Shorter platform transfer paths
                        "initial_wait_reduction_pct": 0.0,
                        "transfer_wait_reduction_pct": 0.0,
                        "egress_time_reduction_pct": 20.0,
                    }
                }
            ]
        },

        # ── Stage 2 ───────────────────────────────────────────────────────────
        # Core Tunnel & Winterthur Hub Package (A0 Gesamt, A1 Winterthur, A2 Tunnel)
        #
        # Official SBB MehrSpur Scope (Opening 2037 | Inv Cost: CHF 2,302.6 M):
        #
        #       A0 Gesamtprojekt (CHF 145.6 M): System-wide technical integration, 
        #       ETCS Level 2 signaling, traction power supply, and overall project management.
        #
        #       A1 Winterthur (CHF 846.3 M): Major track layout reconfiguration at Winterthur HB,
        #       adding grade-separated flyovers, extended platforms, and conflict-free routing.
        #
        #       A2 Tunnel (CHF 1,310.7 M): 8.3 km twin-tube Brüttenertunnel cutting directly between
        #       Dietlikon/Bassersdorf and Winterthur, bypassing the curvy Effretikon bottleneck.
        #
        # Transport & Economic Impact:
        #       Delivers major in-vehicle travel time savings (25 to 30%) on the core Zurich–Winterthur axis.
        #       Creates a continuous 4-track corridor, doubling line capacity and resolving bottlenecks.
        #       Unlocks integral 15-minute clock-face (Viertelstundentakt) across regional & long-distance lines.
        #       Carries over all Stage 1 hub improvements and adds Winterthur HB as a premier multimodal hub.
        2: {
            "name": "Stage 2 – Core Tunnel & Winterthur Hub (A0, A1, A2)",
            
            # --- 1. Global Affinities & Technology Shares ---
            "car_affinity": 1.0,
            "pt_affinity": 1.04,  # Flagship infrastructure service quality boost
            "bike_affinity": 1.0, 
            "walk_affinity": 1.0, 
            "ebike_share": 0.11,
            "EBIKE_SPEED_MULTIPLIER": 1.5,

            # --- 2. Discrete Choice Overrides (ASCs & Betas) ---
            "ASC_PT_BIKE": -1.0,   # Velostationen carried over from Stage 1 (up from -1.5)
            # "B_PT_IVT": -0.045,  
            "B_PT_OVT": -0.070,    # Upgraded station quality (from -0.08)
            "B_TRANSFER": -0.32,   # Seamless step-free transfers (up from -0.43)
            # Optional overrides (uncomment to activate):
            # "ASC_CAR": 0.5,
            # "ASC_WALK": 2.4,
            # "ASC_BIKE": -0.1,
            # "ASC_PT_WALK": 0.0,
            # "B_CAR_TIME": -0.075,
            # "B_WALK_TIME": -0.10,
            # "B_BIKE_TIME": -0.10,
            # "B_COST": -0.18,
            # "B_PT_FARE": -0.18,
            # "CAR_COST_CHF_PER_KM": 0.27,
            # "PT_MIN_FARE_CHF": 3.0,
            # "PT_FARE_CHF_PER_KM": 0.08,
            # "PT_MAX_FARE_CHF": 8.0,

            # --- 3. Physical Network / Link Interventions ---
            "railway_expansions": [
                {
                    "name": "Brüttenertunnel & Winterthur Node Core Link (A0, A1, A2)",
                    "area_pairs": mehrspur_corridor,
                    "both_directions": True,
                    "effects": {
                        # In-Vehicle & Speed (Direct 160 km/h tunnel bypass)
                        "travel_time_reduction_pct": 30.0, # 30% in-vehicle time reduction Zurich-Winterthur
                        "speed_increase_pct": 0.0,
                        "distance_reduction_pct": 0.0,
                        
                        # Full Viertelstundentakt network coordination
                        "initial_wait_reduction_pct": 40.0,  # 15-min headways on core and feeders
                        "transfer_wait_reduction_pct": 30.0, # Synchronized connections at Winterthur HB
                        "transfer_time_reduction_pct": 10.0, 
                        "access_time_reduction_pct": 0.0,
                        "egress_time_reduction_pct": 0.0,
                    }
                }
            ],
            
            # --- 4. Hub & Node Interventions ---
            "mobility_hubs": [
                {
                    "name": "Corridor Multimodal Hubs (A1 Winterthur, A3 Dietlikon, A4 Bassersdorf, A5 Wallisellen)",
                    "zones": [
                        {"municipality_name": "Dietlikon"},
                        {"municipality_name": "Bassersdorf"},
                        {"municipality_name": "Wallisellen"},
                        {"municipality_name": "Winterthur"}
                    ],
                    "effects": {
                        "access_time_reduction_pct": 25.0,   # Carry over Stage 1 & Winterthur feeder access
                        "transfer_time_reduction_pct": 15.0, # Optimized platform connections at Winterthur HB
                        "initial_wait_reduction_pct": 0.0,
                        "transfer_wait_reduction_pct": 0.0,
                        "egress_time_reduction_pct": 20.0,   # Carry over from Stage 1
                    }
                }
            ]
        },
    }
    
    for s in stages.values():
        s["_min_distance_km"] = min_dist_km
        
    return stages







