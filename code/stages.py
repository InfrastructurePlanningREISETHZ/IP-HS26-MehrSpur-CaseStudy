"""
stages.py
=========
Infrastructure stage specifications for the SBB MehrSpur Zürich–Winterthur project.

A 'stage' captures the physical infrastructure and operational interventions
that modify the multimodal transport network and skims in the Canton Zürich FSM model.

HOW TO USE
----------
Edit the two named packages below to define physical interventions and their
spatial scope. Each package groups railway improvements, mobility hubs and
appraisal assumptions (CAPEX, construction emissions and asset lifetime).
Baseline headway, technology and capacity assumptions stay in parameters.py;
deployment timing and triggers stay in adaptive_planning.py.
ASCs/betas describe common behavior, not automatic benefits of individual hubs.
Asset lifetimes below affect final-year appraisal only, not transport.

SURROGATE CONSEQUENCE
---------------------
The two packages can operate independently. Transport states are 0 (baseline),
1 (stations), 2 (tunnel), and 3 (both). The combined state contains the two
packages' benefits once each plus COMBINED_EFFECTS. One GP learns all four configurations. Changes
to their physical effects require rebuilding the surrogate and response table.
Changing only plan timing, costs, comfort capacity, construction emissions or
asset lifetimes does not require a surrogate rebuild. Rerun appraisal to include them.

--- CHEAT SHEET: PARAMETERS & INTERVENTIONS ---

1. Defining Spatial Scopes
   Interventions are targeted geographically using spatial filters:
   
   - "area_pairs": Selects OD skim cells between origins and destinations, not physical links.
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
     Filter by specific zone IDs or whole municipality names. The model selects
     the first two candidate stops per zone for walk and bicycle access, then
     applies effects to journeys using those stops.
     Example: Improving PT access & egress times in Dietlikon:
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


2. Mode-Specific Physical Interventions
   Effects are defined under specific mode keys which target different transport networks:
   - "railway_expansions": A list of railway interventions. The section/service
     entry sets minutes saved, headway reduction and added comfort capacity.
     Additional entries can use OD filters and percentage "effects" as above.
   - "bike_highways": Modifies the Bicycle network.
   - "road_capacity": Modifies selected directed links in the local MSA road network.
   - "mobility_hubs": Modifies PT access, egress and transfer times for walk and bicycle access.
   - "section_time_saving_min" in a "railway_expansions" entry, or directly in a
     package: Minutes saved on the selected parameters.SECTION. For bike_route,
     enter the saving for the complete route; each journey receives the fraction
     corresponding to the improved route length it uses. A pt_stop instead uses
     station-specific mobility_hubs effects below.
     Savings are relative to baseline: stations save 1 minute, the tunnel 4,
     and both packages 6, including a 1-minute combined bonus. The optional external cohort receives the same saving
     in appraisal, without entering mode choice or assignment.
     Section settings follow parameters.SECTION; headway settings apply between
     corridor municipalities. OD filters on additional entries scope only their
     percentage "effects". Comfort capacity is used in crowding appraisal.

   Within these keys, you can apply effects like:
   - "travel_time_reduction_pct" (Reduces in-vehicle travel time)
   - "speed_increase_pct" (Increases average speed)
   - "distance_reduction_pct" (Reduces trip distance, e.g., a new tunnel or bridge)
   - "access_time_reduction_pct", "egress_time_reduction_pct" (Reduces walk time to/from PT)
   
   EXAMPLE 1: Road intervention (increasing selected link capacity by 25%)
   "road_capacity": [
       {
           # Find these directional IDs with the link editor in Notebook 02.
           "edge_ids": ["OSM_123456_789012_0", "OSM_789012_123456_0"],
           "effects": {"capacity_increase_pct": 25.0}
       }
   ]

   Road capacity is link-based: ``area_pairs`` select OD skim cells and cannot
   identify physical road links. The transport interface applies these edits
   to a copy of the corridor immediately before MSA assignment.

   EXAMPLE 2: Station Access Intervention (Reducing PT access/egress times by 20%)
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

   For a counted cycling route configured in parameters.SECTION, a package can
   instead set "section_time_saving_min": 2.0 directly. Do not also apply an OD
   time reduction for the same improvement.

   EXAMPLE 4: A specific station (rather than a municipality-wide access package)
   "mobility_hubs": [
       {"station": "Wallisellen", "effects": {
           "access_time_reduction_pct": 25.0,
           "egress_time_reduction_pct": 20.0,
           "transfer_time_reduction_pct": 15.0
       }}
   ]
   Access and egress affect their respective station users. Transfer reductions
   affect only represented physical walking at this station, not through users.

-----------------------------------Forch Mobility Hub Case Study-----------------------------------
   
code/stages.py defines the infrastructure intervention packages (stages) for the Forch Mobility Hub
case study based on the 'Leitbild Fokus Forch 2045' planning vision.

Each stage applies specific percentage changes to:
- PT transfer/interchange times
- PT waiting times / headways
- PT in-vehicle travel times
- Active mobility / Walk access times

"""

from __future__ import annotations

from typing import Any, Dict
from copy import deepcopy
from math import isfinite

# =============================================================================
# 1. PACKAGES: ROAD- & BIKEWAYS IMPROVEMENTS, MOBILITY HUBS AND APPRAISAL
# =============================================================================

PACKAGES = {
    # --- Package 0: Baseline Condition ---
    "package_0": 
    {
        "name": "Stage 0: Baseline Network Condition",
        "description": "Current baseline network without infrastructure or service modifications.",
        "mobility_hubs": 
        [
            {
                "name": "Baseline Condition",
                "zones": [
                    {"municipality_name": "Küsnacht (ZH)"},
                    {"municipality_name": "Maur"}
                ],
                "effects": {
                    "access_time_reduction_pct": 0.0,
                    "transfer_time_reduction_pct": 0.0,
                    "egress_time_reduction_pct": 0.0,
                    "wait_time_reduction_pct": 0.0,
                }
            }
        ],
        "appraisal": {
            "capital_cost_chf": 0,
            "lifetime_years": 80, #set to 80 years for consistency, but no investment is made in the baseline
            "capital_share": 0.0,
            "construction_co2_tonnes": 0.0,
        },
    },

    # --- Package 1: Central Mobility Hub ---
    "package_1": 
    {
        "name": "Package 1: Forch Central Mobility Hub Transformation",
        "description": "Underground garage, new station square, bus terminal, and micro-mobility hub at Bahnhofsraum Mitte.",
        
        # Hub & Node Interventions
        "mobility_hubs": [
            {
                "name": "Forch Central Hub (Bahnhofsraum Mitte)",
                "zones": [
                    {"municipality_name": "Küsnacht (ZH)"},
                    {"municipality_name": "Maur"}
                ],
                "effects": {
                    "access_time_reduction_pct": 0.0,
                    "transfer_time_reduction_pct": 20.0,    # Seamless transfers at new central square (compact cross-platform terminal)
                    "egress_time_reduction_pct": 0.0,
                    "wait_time_reduction_pct": 5.0,         # Synchronized bus & rail timetables
                }
            }
        ],
    
        # Appraisal considerations for Cost-Benefit Analysis (CBA)
        "appraisal": 
        {
            "capital_cost_chf": 23_745_000,     # Refined CAPEX: Garage (13.6M) + Buses (5.915M) + Chargers (1.35M) + Bays (2.88M)
            "lifetime_years": 80,               # Asset service life (Bundesamt für Verkehr, 2025)
            "capital_share": 0.60,              # Eligible residual value share (60%) - Bus charging stations will not last as long
            "construction_co2_tonnes": 5_000.0, # Estimated construction emissions according to KBOB / Ökobilanzdaten im Baubereich
        },

        "effects_outside_model": [
            "Tailpipe NOx and PM10 elimination at bus terminal from e-buses",
            "Urban quality upgrade and pedestrianization of Bahnhofsraum Mitte",
            "Local decibel noise reduction from electric buses"
        ]
    },

    # --- Package 2: First Mile/Last Mile Access Network ---
    "package_2": 
    {
        "name": "Package 2: First-Mile / Last-Mile Active Feeder Network",
        "description": "200-spot hybrid bike facility, 2.5 km cycleway network, 2 grade-separated bridges, and 8 district Tempo 30 zones.",
        
        # 1. Station Hub Terminal Access & Egress
    "mobility_hubs": [
        {
            "name": "Forch Catchment Feeder Access (Hub & Interface)",
            "zones": [
                {"municipality_name": "Küsnacht (ZH)"},
                {"municipality_name": "Maur"},
                {"municipality_name": "Zumikon"}
            ],
            "effects": {
                # Direct platform-level bike parking & curbless plaza access cuts terminal delay
                "access_time_reduction_pct": 12.0,
                "egress_time_reduction_pct": 12.0
            }
        }
    ],

    # 2. Dedicated Active Infrastructure (Option A: High-Impact Grade-Separated Corridors)
    "bike_highways": 
    [
        {
            "name": "Segment A: Aesch/Scheuren (Maur) Overpass Link",
            "description": "0.9 km cycleway + Overpass Bridge crossing Maurstrasse",
            "area_pairs": [
                {"origin": {"grid_id": "19501001"}, "destination": {"grid_id": "19501002"}}
            ],
            "both_directions": True,
            "effects": {
                "speed_increase_pct": 18.0,            # High speed gain from grade-separated overpass
                "travel_time_reduction_pct": 15.0      # Eliminates signal/traffic wait time at Maurstrasse
            }
        },
        {
            "name": "Segment B: Kaltenstein Underpass Link",
            "description": "1.0 km cycleway + Underpass Link under A52 highway",
            "area_pairs": [
                {"origin": {"grid_id": "15401001"}, "destination": {"grid_id": "15401002"}}
            ],
            "both_directions": True,
            "effects": {
                "speed_increase_pct": 15.0,            # Bypasses highway ramp conflicts
                "travel_time_reduction_pct": 12.0      # Direct continuous routing under A52
            }
        },
        {
            "name": "Segment C: Zumikon Feeder Corridor (Alte Forchstrasse)",
            "description": "0.6 km dedicated cycleway along Alte Forchstrasse corridor",
            "area_pairs": [
                {"origin": {"grid_id": "16001001"}, "destination": {"grid_id": "19501002"}}
            ],
            "both_directions": True,
            "effects": {
                "speed_increase_pct": 12.0,            # Dedicated right-of-way speed gain
                "travel_time_reduction_pct": 10.0      # Uninterrupted feeder flow past local traffic
            }
        }
    ],

        # 3. Directed Road Network Calming (8 Tempo 30 Entry Portals)
        "road_capacity": 
        [
            {
                "name": "Portal Group 1: Bundtstrasse Feeder Access",
                "description": "Traffic calming on the mapped Bundtstrasse entry links toward Forch Station.",
                "edge_ids": [
                    "OSM_206187046_206187043_0",
                    "OSM_206187043_206187046_0",
                ],
                "effects": {
                    "speed_increase_pct": -20.0,       # 50 km/h -> 30 km/h speed limit reduction
                    "capacity_increase_pct": -5.0
                }
            },
            {
                "name": "Portal Group 2: Alte Forchstrasse Feeder Access",
                "description": "Traffic calming on the mapped western Alte Forchstrasse entry links toward Forch Station.",
                "edge_ids": [
                    "OSM_80014366_469826369_0",
                    "OSM_469826369_80014366_0",
                ],
                "effects": {
                    "speed_increase_pct": -20.0,
                    "capacity_increase_pct": -5.0
                }
            },
            {
                "name": "Portal Group 3: Kaltensteinstrasse / Im Grossacher Feeder Access",
                "description": "Traffic calming on the mapped Kaltensteinstrasse and Im Grossacher entry links toward Forch Station.",
                "edge_ids": [
                    "OSM_1490926804_7126219560_0",
                    "OSM_7126219560_1490926804_0",
                    "OSM_4071360703_217302787_0",
                    "OSM_217302787_4071360703_0",
                ],
                "effects": {
                    "speed_increase_pct": -20.0,
                    "capacity_increase_pct": -5.0
                }
            }
        ],
    
        # Appraisal considerations for Cost-Benefit Analysis (CBA)
        "appraisal": 
        {
            "capital_cost_chf": 15_044_800,     # Standalone CAPEX: Bike hub (302.3k) + Cycleways (4M) + Bridges (6.6975M) + Tempo 30 (4M)
            "lifetime_years": 50,               # Blended asset life (Civil bridges: 80 yrs, Cycleways: 30 yrs, Bike racks: 20 yrs)
            "capital_share": 0.85,              # High civil share (85% long-term structural earthworks/bridges vs 15% equipment)
            "construction_co2_tonnes": 2_200.0, # Embodied carbon (Concrete bridges, asphalt cycleways, traffic calming)
        },

        "effects_outside_model": 
        [ 
            "VRU Safety: Substantial drop in pedestrian collision risk (stopping distance halved at 30 km/h)",
            "Modal Shift Synergy: Improved active safety converts short car access trips to walking/e-biking",
            "Neighborhood Livability: Elimination of high-speed cut-through traffic on local residential streets"
        ]
    },  
}

# =============================================================================
# 2. COMBINED-ONLY BENEFITS & SYNERGIES
# =============================================================================
# Synergistic effects & CAPEX savings active only when BOTH Package 1 and Package 2 are combined
COMBINED_EFFECTS = {
    "mobility_hubs": [
        {
            "name": "Forch Integrated Hub Synergies",
            "zones": [
                {"municipality_name": "Küsnacht (ZH)"},
                {"municipality_name": "Maur"},
                {"municipality_name": "Zumikon"}  # Added Zumikon catchment zone
            ],
            "effects": {
                "transfer_time_reduction_pct": 5.0,  # Total transfer = 25% (20% Pkg 1 + 5% Synergy)
                "access_time_reduction_pct": 2.0,    # Total access = 12% (10% Pkg 2 + 2% Synergy)
                "egress_time_reduction_pct": 2.0,    # Total egress = 12% (10% Pkg 2 + 2% Synergy)
                "wait_time_reduction_pct": 0.0,
            }
        }
    ],
    
    # Financial & physical synergy deltas applied to combined appraisal
    "synergies": 
    {
        "underground_shell_capex_savings_chf": 72_800,  # Reduced shell rate (1.65k -> 1.0k/m2 for 112 m2)
        "combined_capital_costs_chf": 38_717_000,       # CHF 23.745M (Pkg 1) + CHF 14.972M (Pkg 2 Synergy)
        "blended_lifetime_years": 70,                   # Weighted lifetime horizon
        "blended_capital_share": 0.70,                  # Weighted civil capital share (70%)
        "total_construction_co2_tonnes": 7_200.0        # Combined Scope 3 construction carbon (5,000t + 2,200t)
    },
    "effects_outside_model": [
        "Land-use efficiency: freeing surface land for high-value Transit-Oriented Development by subterranean parking co-location",
        "Regional multi-modal image elevation enhancing transit competitiveness against private cars"
    ],
}


    # Capacity increases are additive: +10% stations +15% tunnel +5% combined = +30% of baseline.
    # Additional physical interventions can use the same dictionaries as above:
    # "mobility_hubs": [{
    #     "name": "Shared interchange improvement",
    #     "zones": [{"municipality_name": "Dietlikon"}],
    #     "effects": {"transfer_time_reduction_pct": 10.0},
    # }],
    # Add OD-specific PT effects as further railway_expansions entries.
    # Other supported keys: bike_highways, road_capacity.
    # Specify frequency changes above, to apply their waiting benefit only once.

# =============================================================================
# INTERNAL ASSEMBLY AND APPRAISAL HELPERS  (normally leave unchanged)
# =============================================================================

STATE_IDS = (0, 1, 2, 3)
STATE_LABELS = {
    0: "Baseline", 
    1: "Package 1 (Central Hub)", 
    2: "Package 2 (Access Network)", 
    3: "Combined Hub Strategy"
}
STATE_COLORS = {0: "#9E9E9E", 1: "#FFC107", 2: "#2196F3", 3: "#4CAF50"}
_STATE_COMPONENTS = {0: (False, False), 1: (True, False), 2: (False, True), 3: (True, True)}


def _railway_entries(package: dict) -> list[dict]:
    """Use the same list (or single dictionary) form as native interventions."""
    entries = package.get("railway_expansions") or []
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("railway_expansions must be a list of intervention dictionaries.")
    return entries


def _package_railway_effect(package: dict, key: str) -> float:
    """Sum an additive section/service assumption across railway entries."""
    total = 0.0
    entries = _railway_entries(package)
    if key == "section_time_saving_min":
        entries = [package, *entries]
    for entry in entries:
        try:
            value = float(entry.get(key, 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be a finite nonnegative number.") from error
        if not isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number.")
        total += value
    return total


def _railway_od_interventions(package: dict) -> list[dict]:
    """Keep OD interventions after extracting section/service assumptions."""
    settings = {"section_time_saving_min", "headway_reduction_min", "capacity_increase"}
    interventions = []
    for entry in _railway_entries(package):
        intervention = {key: deepcopy(value) for key, value in entry.items() if key not in settings}
        if intervention.keys() - {"name", "description"}:
            interventions.append(intervention)
    return interventions


def package_parameter_defaults() -> dict:
    """Expose package inputs under the shared appraisal/uncertainty parameter keys."""
    values = {}
    for number, name in enumerate(("package_1", "package_2"), 1):
        package = PACKAGES[name]
        values[f"C_INV_STAGE{number}"] = package["appraisal"]["capital_cost_chf"]
        values[f"CAPACITY_INCREASE_STAGE{number}"] = _package_railway_effect(package, "capacity_increase")
    values["CAPACITY_INCREASE_COMBINED"] = _package_railway_effect(COMBINED_EFFECTS, "capacity_increase")
    return values


def package_appraisal_settings() -> dict:
    """Appraisal cache inputs, read from the single package configuration."""
    return {
        "asset_lifetimes": {
            name: {key: package["appraisal"][key] for key in ("lifetime_years", "capital_share")}
            for name, package in PACKAGES.items()
        },
        "construction_emissions": {
            name: package["appraisal"]["construction_co2_tonnes"]
            for name, package in PACKAGES.items()
            if "construction_co2_tonnes" in package["appraisal"]
        },
    }


def get_stages(params: dict | None = None, *, packages: dict | None = None,
               combined_effects: dict | None = None) -> dict[int, dict]:
    """Assemble baseline (0), Package 1 (1), Package 2 (2), and both packages (3)."""
    import parameters as p
    params = {**p.NOMINAL_PARAMS, **(params or {})}

   # Waiting effects cover different corridor municipalities in both directions.
    # Section in-vehicle savings instead follow parameters.SECTION route coverage.

    return _assemble_stages(
        PACKAGES if packages is None else packages, params,
        combined_effects=combined_effects,
    )


def stage_components(stage: int) -> tuple[bool, bool]:
    """Return whether package 1 and package 2 are operating."""
    return _STATE_COMPONENTS[int(stage)]


def state_for_components(pkg1_active: bool, pkg2_active: bool) -> int:
    """Select a transport state from two independently operating packages."""
    components = (bool(pkg1_active), bool(pkg2_active))
    return next(state for state, active in _STATE_COMPONENTS.items() if active == components)


def stage_headway(stage: int, params: dict | None = None) -> float:
    """Minutes between services after additive package and combined-only savings."""
    import parameters as p
    params = {**p.NOMINAL_PARAMS, **(params or {})}
    headway = float(params["PT_HEADWAY_BASELINE"])
    if not isfinite(headway) or headway <= 0:
        raise ValueError("PT_HEADWAY_BASELINE must be finite and positive.")
    reduction = _stage_minute_effect(stage, "headway_reduction_min")
    if reduction >= headway:
        raise ValueError("Total headway reduction must be smaller than PT_HEADWAY_BASELINE.")
    return headway - reduction


def _stage_minute_effect(stage: int, key: str) -> float:
    """Sum configured package minutes, adding the bonus only when both operate."""
    active_packages = stage_components(stage)
    packages = [PACKAGES[name]
                for name, active in zip(("package_1", "package_2"), active_packages) if active]
    if all(active_packages):
        packages.append(COMBINED_EFFECTS)
    return sum(_package_railway_effect(package, key) for package in packages)


def stage_capacity(stage: int, params: dict | None = None) -> float:
    """Peak-hour comfort threshold with package increases and a combined-only bonus."""
    import parameters as p
    params = {**p.NOMINAL_PARAMS, **(params or {})}
    baseline = float(params["PT_CAPACITY_BASELINE"])
    if not isfinite(baseline) or baseline <= 0:
        raise ValueError("PT_CAPACITY_BASELINE must be finite and positive.")
    increase = 0.0
    for package, active in enumerate(stage_components(stage), 1):
        value = float(params[f"CAPACITY_INCREASE_STAGE{package}"])
        if not isfinite(value) or value < 0:
            raise ValueError("Capacity increases must be finite nonnegative fractions.")
        if active:
            increase += value
    bonus = float(params["CAPACITY_INCREASE_COMBINED"])
    if not isfinite(bonus) or bonus < 0:
        raise ValueError("Combined capacity increase must be a finite nonnegative fraction.")
    if all(stage_components(stage)):
        increase += bonus
    return baseline * (1.0 + increase)


def _service_wait_effects(stage: int, params: dict) -> dict:
    """Convert the final state headway to one initial/transfer waiting reduction."""
    reduction = 100.0 * (1.0 - stage_headway(stage, params) / stage_headway(0, params))
    return {"initial_wait_reduction_pct": reduction, "transfer_wait_reduction_pct": reduction}


def _assemble_stages(packages: dict, params: dict, *,
                      combined_effects: dict | None = None) -> dict[int, dict]:
    """Expand concise inputs to the existing native schema and combine packages."""
    combined_effects = COMBINED_EFFECTS if combined_effects is None else combined_effects

    def minute_effect(stage: int, key: str) -> float:
        active = stage_components(stage)
        selected = [packages[name] for name, enabled in zip(("package_1", "package_2"), active) if enabled]
        if all(active):
            selected.append(combined_effects)
        return sum(_package_railway_effect(package, key) for package in selected)

    technology = {
        "ebike_share": float(params["EBIKE_SHARE"]),
        "EBIKE_SPEED_MULTIPLIER": float(params["EBIKE_SPEED_MULTIPLIER"]),
    }

    # Native intervention defaults, not additional student assumptions.

    effect_defaults = {
        "railway_expansions": dict.fromkeys((
            "travel_time_reduction_pct", "speed_increase_pct", "distance_reduction_pct",
            "initial_wait_reduction_pct", "transfer_wait_reduction_pct",
            "transfer_time_reduction_pct", "access_time_reduction_pct", "egress_time_reduction_pct",
        ), 0.0),
        "mobility_hubs": dict.fromkeys((
            "access_time_reduction_pct", "transfer_time_reduction_pct",
            "initial_wait_reduction_pct", "transfer_wait_reduction_pct", "egress_time_reduction_pct",
        ), 0.0),
    }
    stages = {0: {"name": "Stage 0 – Baseline Network", "section_time_saving_min": 0.0, **technology}}
    for stage, package in ((1, "package_1"), (2, "package_2")):
        inputs = deepcopy(packages[package])
        inputs.pop("appraisal", None)
        inputs.pop("section_time_saving_min", None)
        inputs.pop("railway_expansions", None)
        for key in ("mobility_hubs", "bike_highways", "road_capacity"):
            if isinstance(inputs.get(key), dict):
                inputs[key] = [inputs[key]]
        railway_od = _railway_od_interventions(packages[package])
        if railway_od:
            inputs["railway_expansions"] = railway_od
        specification = {
            "name": inputs.pop("name"),
            "section_time_saving_min": minute_effect(stage, "section_time_saving_min"),
            **technology, **inputs,
        }
        stages[stage] = specification
    combined = deepcopy(stages[2])
    combined["name"] = "Both stages - Forch Integrated Mobility Hub"
    combined["section_time_saving_min"] = minute_effect(3, "section_time_saving_min")
    for key in ("railway_expansions", "mobility_hubs", "bike_highways", "road_capacity"):
        if key in stages[1] or key in stages[2] or key in combined_effects:
            bonus = (_railway_od_interventions(combined_effects) if key == "railway_expansions"
                     else combined_effects.get(key, []))
            if isinstance(bonus, dict):
                bonus = [bonus]
            combined[key] = deepcopy(stages[1].get(key, []) + stages[2].get(key, []) + bonus)
    stages[3] = combined
    for stage, specification in stages.items():
        for key, defaults in effect_defaults.items():
            for intervention in specification.get(key, []):
                intervention["effects"] = {**defaults, **intervention.get("effects", {})}
    return stages


def construction_emissions_tonnes(package: int | str) -> float:
    """Total construction tonnes CO2e for package 1 or package 2."""
    package_names = {1: "package_1", 2: "package_2", "package_1": "package_1", "package_2": "package_2"}
    if isinstance(package, bool) or package not in package_names:
        raise ValueError("Construction emissions package must be 1/'package_1' or 2/'package_2'.")
    try:
        value = float(PACKAGES[package_names[package]]["appraisal"].get("construction_co2_tonnes", 0.0))
    except (TypeError, ValueError) as error:
        raise ValueError("Construction emissions must be finite nonnegative tonnes CO2e.") from error
    if not isfinite(value) or value < 0:
        raise ValueError("Construction emissions must be finite nonnegative tonnes CO2e.")
    return value


def asset_residual_value(package: int | str, investment_chf: float, operated_years: float) -> float:
    """Undiscounted CHF remaining at horizon end for one built package."""
    package_names = {1: "package_1", 2: "package_2", "package_1": "package_1", "package_2": "package_2"}
    if isinstance(package, bool) or package not in package_names:
        raise ValueError("Residual valuation package must be 1/'package_1' or 2/'package_2'.")
    settings = PACKAGES[package_names[package]]["appraisal"]
    try:
        investment = float(investment_chf)
        age = float(operated_years)
        share = float(settings["capital_share"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Asset investment_chf, operated_years and capital_share must be numeric.") from exc
    if not isfinite(investment) or investment < 0:
        raise ValueError("Asset investment_chf must be finite and nonnegative.")
    if not isfinite(age) or age < 0:
        raise ValueError("Asset operated_years must be finite and nonnegative.")
    if not isfinite(share) or not 0 <= share <= 1:
        raise ValueError("Asset capital_share must be a finite fraction between zero and one.")
    if share == 0:
        return 0.0
    try:
        lifetime = float(settings["lifetime_years"])
    except (TypeError, ValueError) as exc:
        raise ValueError("An enabled asset lifetime_years must be finite and positive.") from exc
    if not isfinite(lifetime) or lifetime <= 0:
        raise ValueError("An enabled asset lifetime_years must be finite and positive.")
    return investment * share * max(0.0, 1.0 - age / lifetime)

