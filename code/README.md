# How the MehrSpur transport simulation works

The pipeline has two layers. The **spatial layer** runs the Canton Zürich four-step FSM (1,223 zones) once per infrastructure stage to compute OD matrices, modal split, travel times, distances, and corridor indicators. The **40-year pathway layer** then reuses those stage results, scaling demand and switching between stages year by year without re-running the FSM. This makes large uncertainty sweeps computationally feasible, at the cost of the long-term layer being an aggregate approximation.

```text
parameters.py ──► stages.py, pathways.py, simulation_engine.py

stages.py ─┐
           ├──► transport_model_interface.py ──► stage metrics
IP_course_FSM-main ─┘

stage metrics ──► pathways.py ⇄ simulation_engine.py (40-yr loop)
                       ▼
        40-year DataFrame ──► discounted NPC & robustness
```

## Files you edit

### [`parameters.py`](parameters.py)

Central registry for every numeric assumption. All other modules import from here. Two dictionaries control how the uncertainty engine behaves:

- [`PERTURBABLE_PARAMS`](parameters.py#L69-L99) — economic and cost parameters that EMA Workbench perturbs during Monte Carlo runs: social discount rate, fuel consumption and price, CO₂ emission factor and carbon price (with annual escalation via `C_CO2_GROWTH`), values of travel time for car and PT, stage CapEx, annual OpEx, and the flexibility option premium. Any new economic parameter you add here is automatically picked up by [`sample_perturbed_params()`](simulation_engine.py#L201-L212), which maps a uniform draw $u \in [0,1]$ to a Normal realization around the nominal with a 10% standard deviation (`PERTURBABLE_SD_FRACTION = 0.10`, [parameters.py#L162](parameters.py#L162)).

- [`FIXED_PARAMS`](parameters.py#L105-L128) — engineering facts held constant across all runs: `PEAK_TO_ANNUAL = 1200.0` (300 operating days × 4 peak hours, used to annualize all peak-hour metrics), PT headways and onboard capacities per stage (Stage 0: 20 min / 85k trips; Stage 1: 18 min / 100k trips; Stage 2: 15 min / 110k trips), and NIBA unit external costs for noise, air quality, and accidents.

- [`STRUCTURAL_UNCERTAINTIES`](parameters.py#L140-L159) — two deep trajectory uncertainties: cumulative demand growth (`u_demand`, nominal +30% by Year 40) and PT preference (`u_beta_pt`, multiplier starting at 1.0, nominal end 1.2). A uniform EMA sample $u$ maps to $z = \Phi^{-1}(u)$ and produces a 40-year trajectory via $x(t) = \mu(t) + z \cdot \sigma(t)$, where $\mu(t)$ and $\sigma(t)$ both evolve linearly. Holding $z$ fixed over time produces coherent future states rather than independent annual noise.

- [`CORRIDOR_REGIONS`](parameters.py#L38-L50) — municipality names defining the Zürich–Winterthur corridor, mapped to FSM zone IDs at runtime. A trip is counted as in-corridor if its origin **or** destination zone belongs to the corridor.

- Scalar planning thresholds: `N_YEARS = 40`, `MAX_AVG_TT = 20` (minutes), `PT_SHARE_TARGET = 0.35`.

[`validate_params()`](parameters.py#L178-L202) runs at import time and raises `ValueError` on physically implausible values.

---

### [`stages.py`](stages.py)

Defines what each infrastructure stage physically does. [`get_stages()`](stages.py#L108-L331) returns a dictionary keyed by stage index (0, 1, 2). Each entry specifies a list of intervention dictionaries that `transport_model_interface.py` applies to the FSM skims and mode-choice coefficients.

Interventions operate in two ways:
- **Physical** — direct edits to travel-time or distance skims (e.g., in-vehicle time reduction, access/egress time reduction, transfer wait reduction). Specified as percentage reductions or absolute overrides; applied via [`apply_interventions()`](../IP_course_FSM-main/interventions.py#L402-L429) using an OD zone mask.
- **Behavioural** — adjustments to MNL Alternative-Specific Constants via `*_affinity` multipliers. An affinity is an odds multiplier: `adjusted ASC = ASC + ln(affinity)` ([mode_choice_zurich.py#L84-L91](../IP_course_FSM-main/mode_choice_zurich.py#L84-L91)). A value of 1.0 leaves the baseline unchanged; values above 1.0 increase mode attractiveness; values below 1.0 reduce it.

Stage 0 ([stages.py#L127-L139](stages.py#L127-L139)) is the no-build baseline. Stage 1 ([stages.py#L161-L232](stages.py#L161-L232)) implements the local station and access package. Stage 2 ([stages.py#L253-L325](stages.py#L253-L325)) adds the Brüttenertunnel and the 15-minute service rhythm.

The `_min_distance_km` field (set per stage and read by [`extract_corridor_metrics()`](transport_model_interface.py#L635-L744)) controls the PKM distance filter for modal split reporting. In the MehrSpur reference case it is set to 5 km, excluding local walking trips from the headline `car_share` and `pt_share` figures. `*_share_trips` are unfiltered raw trip counts. Set to 0.0 to disable filtering.

---

### [`pathways.py`](pathways.py)

Defines the nine planning strategies and runs the 40-year simulation loop.

**Strategy definitions** — [`get_pathways()`](pathways.py#L98-L202) returns a dictionary with nine entries: `baseline`, `static1`, `static2`, `staged1`, `staged2`, `staged3`, `flexible1`, `flexible2`, `flexible3`. Each specifies an `initial_stage` (0, 1, or 2) and optional transitions `to1` and `to2`. Transitions are either `fixed` (a deterministic calendar year) or `trigger` (adaptive threshold rule). The function computes CapEx, OpEx, and option premium assignments from `parameters.py` automatically.

**Adaptive triggers** — [`DEFAULT_TRIGGERS`](pathways.py#L46-L69) configures the two signpost-based rules. Each trigger specifies:
- `signpost`: the monitored metric — `"pt_trips"`, `"annual_pt_trips"`, `"pt_share"`, `"avg_tt_min"`, or `"congestion_delay_hours"`.
- `threshold`: the value that, when exceeded, increments the persistence counter.
- `persistence`: consecutive years the threshold must be exceeded before a decision is made.
- `lead_time`: years from decision to stage opening (construction duration).
- `stage_activated`: which stage (1 or 2) the trigger activates.

The annual simulation loop in [`run_pathway_from_trajectories()`](pathways.py#L303-L461) processes each year as follows:

```text
each year:
   fixed transition scheduled? ──yes──► activate stage
        ▼
   read previous year's signpost (t−1)
        ▼
   signpost > threshold?
        yes ──► persistence += 1 ──► reached? ──yes──► schedule decision
        no  ──► persistence = 0            (opening = decision year + lead time)
        ▼
   opening year reached? ──yes──► activate new stage
        ▼
   simulate year ──► add CapEx / OpEx ──► store row + signposts
        ▼
   more years? ──yes──► repeat     no ──► return 40 rows + decision metadata
```

Key rules: triggers read the *previous* year's signpost to avoid look-ahead bias ([pathways.py#L340-L385](pathways.py#L340-L385)); threshold comparisons are strict (`>`); a single year at or below threshold resets persistence to zero; Stage 2 triggers only evaluate once Stage 1 is active (`stage >= 1`). Fixed transitions fire at the start of the target year without lag ([pathways.py#L332-L338](pathways.py#L332-L338)).

*Example:* persistence = 2, lead time = 5. Threshold exceeded in Years 1–2 → decision in Year 3 → Stage 2 opens Year 8.

**Trajectory shapes** — [`demand_growth_trajectory()`](pathways.py#L269-L283) and [`pt_affinity_trajectory()`](pathways.py#L253-L267) accept a `shape` argument from `SHAPES = ("linear", "early", "late", "logistic", "random_walk", "almost_flat")` ([pathways.py#L214](pathways.py#L214)). The `random_walk` shape uses an Ornstein-Uhlenbeck mean-reverting process ([pathways.py#L239-L251](pathways.py#L239-L251)); all others interpolate $\mu(t)$ and $\sigma(t)$ via a shape fraction $f(t)$ before applying the fixed $z$ score.

**Financial accounting:** flexibility option premiums (`C_FLEX`, one per trigger in the pathway) and initial-stage CapEx are booked in Year 1. Later-stage CapEx is paid as a lump sum in the activation year. OpEx is charged annually for all active stages; Stage 2 OpEx is cumulative (`op_stage1 + op_stage2`, [pathways.py#L419-L423](pathways.py#L419-L423)).

[`run_pathway()`](pathways.py#L289-L300) returns `(results, metadata)`:
- `results`: 40-row DataFrame with annual stage, scaled demand, modal split, vehicle trips, `avg_tt_min`, `co2_tonnes`, `congestion_delay_hours`, disaggregated societal costs, CapEx, OpEx, and total annual cost.
- `metadata`: `decision1_year`, `activation1_year`, `decision2_year`, `activation2_year`.

---

### [`simulation_engine.py`](simulation_engine.py)

Annual cost formulas, physical indicators, uncertainty sampling, and NPC appraisal. Called from `pathways.py` year by year; also exposes the uncertainty interface for EMA Workbench.

**Annual costs** — [`annual_mode_costs()`](simulation_engine.py#L51-L114) computes societal costs in CHF, annualized via `PEAK_TO_ANNUAL`:
- Car: travel time + fuel + CO₂ + noise (NIBA 2.1) + air quality (NIBA 1.1) + accidents (NIBA 20.1). Travel time scales with demand growth plus road congestion delay; VKT-based externalities scale with $D_\text{car} \times (1 + g_t) \times \text{PEAK\_TO\_ANNUAL}$. Carbon price increases annually by `C_CO2_GROWTH`.
- PT: in-vehicle time + wait (half the stage headway, capped at 5 min) × a quadratic crowding multiplier ([simulation_engine.py#L85-L101](simulation_engine.py#L85-L101)):

```text
crowding multiplier = max[1.0, (peak PT trips / capacity)²]
PT time cost         = (in-vehicle time + wait) × C_TT_PT × crowding multiplier
```

**Physical indicators** — [`annual_physical_indicators()`](simulation_engine.py#L116-L143): `co2_tonnes` from car VKT × emission factor; `avg_tt_min` as $(T_\text{car} + T_\text{pt}) / (Q_\text{car} + Q_\text{pt}) \times 60$ (motorized trips only); `congestion_delay_hours` = extra delay × `PEAK_TO_ANNUAL`.

**NPC appraisal** — [`npc_by_component()`](simulation_engine.py#L355-L412) discounts all cash flows at $d(t) = (1+r)^{-t}$ and returns MCHF totals for car subcomponents (fuel, CO₂ NIBA 6.1, noise, air, accidents), PT time (NIBA 11.1), CapEx (NIBA 10.6), OpEx (NIBA 10.5), and total NPC. Against a baseline, travel-time differences apply the Rule of Half ([simulation_engine.py#L359-L388](simulation_engine.py#L359-L388)):
$$\text{Benefit} = \tfrac{1}{2}(Q_\text{base} + Q_\text{proj}) \times (C_\text{base} - C_\text{proj})$$
Consumer surplus gains enter NPC as negative costs.

**Uncertainty sampling** — [`get_ema_uncertainties()`](simulation_engine.py#L183-L198) dynamically builds EMA Workbench `RealParameter` objects from `STRUCTURAL_UNCERTAINTIES` and `PERTURBABLE_PARAMS`. [`sample_perturbed_params()`](simulation_engine.py#L201-L212) converts uniform draws to parameter realizations:

```text
uniform EMA sample u ──► z = Φ⁻¹(u)
   ▼                              ▼
demand/PT trajectory        perturbed parameter
(μ_t + z·σ_t)               (nominal + z·10%)
   └──────────────┬──────────────┘
                   ▼
         40-year pathway run ──► NPC, timing, robustness
```

## Files you don't touch

### [`transport_model_interface.py`](transport_model_interface.py)

Connects `stages.py` to the FSM and extracts corridor metrics. You call its functions from notebooks; you do not edit it unless implementing an advanced intervention type not covered by the existing skim-edit and affinity mechanisms.

[`load_transport_context()`](transport_model_interface.py#L255-L326) loads zones, baseline OD demand, skims, and the road network into a [`TransportContext`](transport_model_interface.py#L88-L125), cached at `cache/tmi_context.pkl`. [`run_simulation()`](transport_model_interface.py#L747-L815) is the main entry point: it applies a stage, runs MNL mode choice and optional road assignment, and returns a [`ModeChoiceResult`](transport_model_interface.py#L133-L175) plus a corridor metrics dict.

Road congestion is configured via [`ASSIGNMENT_SETTINGS`](transport_model_interface.py#L49-L61):

- `"method": "LUT"` (default) — [`get_lut_delay()`](simulation_engine.py#L215-L239) interpolates corridor delay from `data/processed/delay_lut_stage_<stage>.json`. Missing file → zero delay; demand beyond the precomputed range clamps to the nearest endpoint. Required for 40-year pathway runs and uncertainty sweeps.
- `"method": "MSA"` — [`run_route_assignment()`](transport_model_interface.py#L1331-L1524) assigns car OD to the corridor subnetwork and iterates with BPR link delays: $t = t_0 \cdot [1 + 0.15 \cdot (V/C)^4]$. MSA update: $f_{\text{new}} = f_{\text{old}} + \tfrac{1}{k}(f_{\text{AON}} - f_{\text{old}})$. Stops once relative L1 flow change < `relative_gap_threshold` (after `min_iterations`).
- `"modal_feedback": True` — when MSA is active, congested car times feed back into mode choice for a coupled equilibrium. In LUT mode this flag has no effect.
- `"regenerate_lut": True` — sweeps demand multipliers 0.8–2.0 through MSA to rebuild the JSON LUT files for each stage.

[`extract_corridor_metrics()`](transport_model_interface.py#L635-L744) aggregates the MNL output into the dict that feeds `pathways.py`:

```python
stage_metrics = {
    0: {"total_trips": ..., "car_trips": ..., "pt_trips": ...,
        "car_share": ..., "pt_share": ...,   # PKM-weighted, distance-filtered
        "car_dist_km": ..., "car_tt_hours": ..., "pt_tt_hours": ..., ...},
    1: {...},
    2: {...},
}
```

### [`IP_course_FSM-main/`](../IP_course_FSM-main/)

The backend FSM. Do not modify. Modules used by this pipeline: [`scenario_generation.py`](../IP_course_FSM-main/scenario_generation.py) (Furness demand balancing), [`travel_times.py`](../IP_course_FSM-main/travel_times.py) (skim adjustments), [`interventions.py`](../IP_course_FSM-main/interventions.py) (skim edits and mobility hubs), [`mode_choice_zurich.py`](../IP_course_FSM-main/mode_choice_zurich.py) (MNL utilities and probabilities).

## Modelling assumptions to state when presenting results

The 40-year loop is a surrogate. FSM mode choice runs once per stage at baseline demand; annual loops scale corridor aggregate metrics rather than re-running the spatial model. PT-affinity trajectories (`u_beta_pt`) adjust reported shares and trips in [pathways.py#L393-L405](pathways.py#L393-L405) after `simulate_year()` has already costed the base stage metrics — behavioral adoption shifts the mode split but does not re-enter the logit model.

Free-flow travel time and road congestion delay are decoupled. Baseline car times scale linearly with demand growth; delay is added separately via LUT or MSA and never double-counted in stage metrics ([transport_model_interface.py#L686-L687](transport_model_interface.py#L686-L687)). Highway assignment uses a corridor subnetwork with gate-compressed external demand and a limited MSA iteration count. Mode choice and road assignment reach bilevel equilibrium only when `modal_feedback: true` is combined with MSA; in LUT mode, congested times do not feed back to alter mode shares.

Primary corridor mode shares (`car_share`, `pt_share`) are PKM-weighted and filtered to trips ≥ `_min_distance_km`. Raw trip-count shares are `car_share_trips` and `pt_share_trips`. Adaptive triggers react with a one-year lag ($t-1$) and then incur the physical lead time.

Unit conventions: skims in minutes/meters; aggregate cost inputs in vehicle-hours and kilometres; annual costs in CHF; NPC in MCHF.
