# How the MehrSpur transport simulation works

This pipeline has **two parts**: 
1. **Transport-model components**
   - the four step transport model engine contained in [`IP_course_FSM-main`](../IP_course_FSM-main/) which runs in the background and doesn't need to be understood in detail or be modified by students
   - [`transport_model_interface.py`](transport_model_interface.py) and [`simulation_engine.py`](simulation_engine.py), which help in handling the modeling process and, in most cases, do not need to be modified
2. **Project/case study files**, which are meant to be edited by you for your case study. They include:  
   - [`parameters.py`](parameters.py) (assumptions, thresholds, costs, uncertainty ranges), 
   - [`stages.py`](stages.py) (infrastructure interventions), 
   - [`pathways.py`](pathways.py) (strategy timing/triggers).

The model has **two layers**: 
- a **spatial layer** (demand, mode choice, travel time/distance, corridor indicators per stage, across 1,223 zones) 
- **40-year pathway layer** (reuses stage results to simulate growth, congestion, stage transitions, investment, and cost year by year). The spatial model runs once per stage, not per year/scenario — cheap exploration, at the cost of the long-term layer being an aggregate approximation of precomputed stage metrics.

## Contents

- [Architecture at a glance](#architecture-at-a-glance)
- [The spatial transport simulation](#the-spatial-transport-simulation)
- [Mode choice in detail](#mode-choice-in-detail)
- [Road congestion: LUT and MSA](#road-congestion-lut-and-msa)
- [The 40-year pathway simulation](#the-40-year-pathway-simulation)
- [Costs, indicators, and appraisal](#costs-indicators-and-appraisal)
- [Uncertainty](#uncertainty)
- [Inputs and outputs](#inputs-and-outputs)
- [Where students should make changes](#where-students-should-make-changes)
- [Important modelling assumptions](#important-modelling-assumptions)

## Architecture at a glance

| File | Responsibility | Student interaction |
|---|---|---|
| [`parameters.py`](parameters.py) | Horizon, corridor, thresholds, capacities, valuation factors, costs, uncertainty registries | Edit assumptions & uncertainty ranges |
| [`stages.py`](stages.py) | Stage 0/1/2 and their effects on skims and mode-choice parameters | Define/revise interventions |
| [`transport_model_interface.py`](transport_model_interface.py) | Loads FSM, applies a stage, runs mode choice/assignment, extracts metrics | Call functions; don't edit internals |
| [`simulation_engine.py`](simulation_engine.py) | Stage metrics → annual indicators/costs; uncertainty helpers; NPC | Edit formulas only if required |
| [`pathways.py`](pathways.py) | Fixed & adaptive strategies; stage changes over 40 years | Edit pathways, signposts, thresholds |

```text
parameters.py ──► stages.py, pathways.py, simulation_engine.py

stages.py ─┐
           ├──► transport_model_interface.py ──► stage metrics
IP_course_FSM-main ─┘

stage metrics ──► pathways.py ⇄ simulation_engine.py (40-yr loop)
                       ▼
        40-year DataFrame ──► discounted NPC & robustness
```

FSM pieces used here: demand in [`scenario_generation.py`](../IP_course_FSM-main/scenario_generation.py#L66-L120), skims in [`travel_times.py`](../IP_course_FSM-main/travel_times.py#L192-L215), interventions in [`interventions.py`](../IP_course_FSM-main/interventions.py#L402-L429), MNL choice in [`mode_choice_zurich.py`](../IP_course_FSM-main/mode_choice_zurich.py#L38-L172).

## The spatial transport simulation

Entry point [`run_simulation()`](transport_model_interface.py#L703-L777) returns a [`ModeChoiceResult`](transport_model_interface.py#L128-L171) (OD matrices by mode, modified skims, scenario, optional assignment) plus a metrics dict for the pathway/cost engines.

```text
FSM inputs (zones, OD demand, multimodal skims, road graph)
   ▼
select infrastructure stage
   ▼
generate and Furness-balance scenario demand
   ▼
apply perceived-time assumptions
   ▼
apply stage interventions and e-bike adjustment
   ▼
calculate preliminary 5-mode utilities
   ▼
preliminary MNL mode choice and OD matrix per mode
   ▼
select congestion method
   ├── LUT
   │    └── interpolate stage-specific congestion delay
   │
   └── MSA
        └── assign preliminary car OD to the local road network
             ▼
             calculate congested link travel times
             ▼
             update the car travel-time skim
             ▼
             recalculate car utility and final MNL mode choice (= going back to modal split over a set of x iterations)
   ▼
extract final corridor metrics
(trips, modal split, PKM shares, travel time, distance, CO₂, and delay)
   ▼
stage metrics used by the 40-year pathway model
```

### Congestion feedback within the modeling pipeline

When MSA assignment is enabled, mode choice and road congestion are connected
through a feedback step.

The model first calculates a preliminary modal split using the initial
multimodal travel-time skims. The resulting car OD matrix is then assigned to
the local road network using the Method of Successive Averages (MSA). Link
travel times are updated with the BPR volume–delay function, based on the
assigned traffic volume and available road capacity.

The resulting congested car travel-time skim is subsequently passed back to
the mode-choice model. Car utilities and multinomial-logit probabilities are
then recalculated, producing the final modal split and final OD matrices by
mode.

This means that congestion can influence travellers' mode choices: when road
travel times increase, the utility of driving decreases, potentially shifting
demand toward public transport, cycling, or walking.


- **Load & cache** — [`load_transport_context()`](transport_model_interface.py#L255-L326) loads zones, baseline OD, background road demand, skims, road network into [`TransportContext`](transport_model_interface.py#L88-L125); cached at `cache/tmi_context.pkl` ([status checks](transport_model_interface.py#L178-L197)). *OD matrix* = trips by origin/destination; *skim* = a level-of-service value over the same dims.
- **Corridor** — municipalities are set in [`parameters.py`](parameters.py#L35-L58); [`get_zone_ids_for_municipalities()`](transport_model_interface.py#L333-L349) maps them to zone IDs. In-corridor = **origin OR destination** is a corridor zone ([`_corridor_mode_summary()`](transport_model_interface.py#L375-L407), [`extract_corridor_metrics()`](transport_model_interface.py#L624-L639)).
- **Stage → scenario** — [`get_stages()`](stages.py#L108-L331): Stage 0 baseline ([def](stages.py#L127-L139)), Stage 1 local stations ([def](stages.py#L161-L232)), Stage 2 core tunnel + hub ([def](stages.py#L253-L325)). Stages act **physically** (skim edits) or **behaviourally** (ASC overrides / `*_affinity` multipliers); converted to an FSM scenario at [L457-480](transport_model_interface.py#L457-L480).
- **Demand** — [`generate_scenario_demand()`](../IP_course_FSM-main/scenario_generation.py#L66-L120) scales productions/attractions from population/employment and Furness-balances ([`furness_balance()`](../IP_course_FSM-main/scenario_generation.py#L30-L63)) the baseline OD to match.
- **Time & interventions** — [`run_transport_mode_choice()`](transport_model_interface.py#L410-L584) applies, in order: [`apply_perceived_time_policies()`](../IP_course_FSM-main/travel_times.py#L192-L215), [`apply_interventions()`](../IP_course_FSM-main/interventions.py#L402-L429) (via [`_od_mask()`](../IP_course_FSM-main/interventions.py#L45-L85), [`_apply_railway_expansion()`](../IP_course_FSM-main/interventions.py#L228-L284), [`_apply_mobility_hub()`](../IP_course_FSM-main/interventions.py#L287-L353), floored by [`_multiply_selected()`](../IP_course_FSM-main/interventions.py#L127-L144)), and [`apply_ebike_share()`](../IP_course_FSM-main/travel_times.py#L218-L290). For reduction `r`% and speed gain `s`%: `reduction factor = 1-r/100`, `speed factor = 1/(1+s/100)`.

## Mode choice in detail

Every OD cell splits across five alternatives — `drive`, `bike`, `walk`, `pt_walk`, `pt_bike` — via a multinomial logit in [`mode_split_aggregated()`](../IP_course_FSM-main/mode_choice_zurich.py#L38-L172). Coefficients: [`config/mode_choice.json`](../IP_course_FSM-main/config/mode_choice.json), unless a stage overrides them.

```text
U_car     = ASC_car + β_car,time·time + β_cost·cost
U_pt_walk = ASC_pt_walk + β_ivt·IVT + β_ovt·OVT + β_fare·fare + β_transfer·transfers
(walk, bike, pt_bike analogous — mode_choice_zurich.py#L72-L110)
```

Car cost = distance × CHF/km; PT fare = minimum + distance component, capped. An **affinity** is an odds multiplier: `adjusted ASC = ASC + ln(affinity)` ([impl](../IP_course_FSM-main/mode_choice_zurich.py#L84-L91)) — `1.0` no change, `>1.0` more attractive, `<1.0` less.

Unavailable alternatives get very negative utility; walking is capped to reachable trips by [`standalone_walk_mask()`](../IP_course_FSM-main/travel_times.py#L293-L299). `P(m) = exp(U_m) / Σ exp(U_j)`, max-shifted for stability ([impl](../IP_course_FSM-main/mode_choice_zurich.py#L122-L154)), scaled by total OD demand per mode ([impl](../IP_course_FSM-main/mode_choice_zurich.py#L156-L172)).

[`extract_corridor_metrics()`](transport_model_interface.py#L591-L700) reduces this to: trips/PKM shares by mode, car VKT, aggregate travel time, car CO₂, mean motorised time. Primary shares (`car_share`, `pt_share`, …) are **PKM-weighted, filtered to trips ≥ `_min_distance_km`** (default 5 km, [mask](transport_model_interface.py#L654-L693)); `*_share_trips` are ordinary trip shares. Car free-flow time comes from the original context ([L641-649](transport_model_interface.py#L641-L649)) — congestion delay is added separately, never double-counted.

## Road congestion: LUT and MSA

Switch: [`ASSIGNMENT_SETTINGS`](transport_model_interface.py#L48-L57).

**LUT (default, fast)** — reads a stage delay table from `data/processed/delay_lut_stage_<stage>.json`; [`get_lut_delay()`](simulation_engine.py#L215-L239) interpolates between stored demand multipliers (missing/unreadable → zero delay). Used for uncertainty runs: thousands of combinations without repeating assignment.

**MSA (slower, demand-responsive)** — [`run_route_assignment()`](transport_model_interface.py#L1331-L1524) clips the network to corridor+buffer, maps external OD to nearest cordon gates, compresses the car OD, and sends it to [`coarse_msa_assignment()`](transport_model_interface.py#L1531-L1683). BPR link time: `t = t₀·[1 + α·(flow/capacity)^β]` ([impl](transport_model_interface.py#L1603-L1608), default `α=0.15`, `β=4.0`, [construction](transport_model_interface.py#L1884-L1897)). MSA loop: recompute link times → all-or-nothing shortest-path assignment → `flow_new = flow_old + (1/iteration)·(AON_flow-flow_old)` ([loop](transport_model_interface.py#L1599-L1646)) → stop once relative L1 flow change < threshold (after a minimum iteration count) or at the max. Outputs: flow, v/c, congested time, speed, delay, VKT, VHT, convergence ([L1648-1683](transport_model_interface.py#L1648-L1683)).

LUTs are (re)generated by sweeping demand multipliers 0.8–2.0 through MSA per stage ([logic](transport_model_interface.py#L1453-L1500)):

```text
stage car OD ──► scale by demand multiplier (0.8 … 2.0) ──► MSA assignment
   ▼
BPR link congestion ──► stage delay LUT ──► interpolate ──► annual costs
```

## The 40-year pathway simulation

[`get_pathways()`](pathways.py#L98-L202) defines nine strategies: 1 baseline, 2 static, 3 fixed-staged, 3 adaptive/trigger-based. Each has an initial stage plus optional `to1`/`to2` transitions: `fixed` (activate in a given year) or `trigger` (signpost + threshold + persistence + lead time — defaults at [pathways.py#L30-L69](pathways.py#L30-L69); signposts: PT trips, annual PT trips, PT share, average travel time, congestion delay).

[`run_pathway()`](pathways.py#L289-L300) builds growth/PT-affinity trajectories, then runs the annual loop in [`run_pathway_from_trajectories()`](pathways.py#L303-L461):

```text
each year:
   fixed transition scheduled? ──yes──► activate stage
        ▼
   read previous year's signpost
        ▼
   signpost > threshold?
        yes ──► persistence += 1 ──► reached? ──yes──► schedule decision
        no  ──► persistence = 0            (opening = decision year + lead time)
        ▼
   opening year reached? ──yes──► activate new stage
        ▼
   simulate year ──► add investment/operating costs ──► store row + signposts
        ▼
   more years? ──yes──► repeat     no ──► return 40 rows + decision metadata
```

Rules: fixed transitions fire at year start ([332-338](pathways.py#L332-L338)); adaptive decisions use the **previous year's** signpost, avoiding look-ahead ([340-385](pathways.py#L340-L385)); threshold test is strict `>`; one below-threshold year resets persistence; activation only after the lead time; Stage 2 triggers evaluate only once Stage 1 is active. *Example:* 2-yr persistence + 5-yr lead time → exceedances in Years 1-2 → decision Year 3 → activation Year 8.

Per year, [`simulate_year()`](simulation_engine.py#L242-L307) turns cumulative growth into demand scale, adds LUT/MSA delay, computes costs & physical indicators; the pathway runner adds shares/trips/CapEx/OpEx/total cost ([387-447](pathways.py#L387-L447)) and stores outputs as next year's signposts. Investment: flexibility premium + initial-stage infrastructure paid Year 1; later-stage investment paid on activation; operating cost paid every active year (Stage 2 OpEx includes Stage 1's, [419-423](pathways.py#L419-L423)). Metadata records decision/activation years for both transitions ([456-461](pathways.py#L456-L461)).

## Costs, indicators, and appraisal

Peak-period quantities are annualised via `PEAK_TO_ANNUAL` (currently `1200`; [param](parameters.py#L110-L129), [calc](simulation_engine.py#L66-L74)) — trust the calculation over docstrings that say "daily".

[`annual_mode_costs()`](simulation_engine.py#L51-L113): car cost = travel-time + fuel + CO₂ + noise + air-pollution + accident cost (time scales with demand + congestion delay; distance terms scale with car VKT; carbon value can grow via `C_CO2_GROWTH`). PT time = in-vehicle + half the stage headway as wait, with crowding:

```text
crowding multiplier = max[1, (peak PT trips / capacity)²]
PT time cost         = (in-vehicle time + wait) × value of time × crowding multiplier
```

([calc](simulation_engine.py#L85-L101); headways/capacities per stage: [parameters.py#L115-L128](parameters.py#L115-L128).)

[`annual_physical_indicators()`](simulation_engine.py#L116-L143): annual car CO₂ (t), mean motorised travel time (car+PT time / car+PT trips, walk/bike excluded), annual congestion-delay hours.

[`npc_by_component()`](simulation_engine.py#L310-L366): `discount factor(t) = 1/(1+discount rate)^t`; returns car/PT/investment/operating/total NPC (MCHF) with NIBA-style car subcomponents. Against a baseline, car/PT time differences use the **rule of half** ([impl](simulation_engine.py#L314-L343)) — user benefits enter as negative costs.

## Uncertainty

**Structural trajectory** — [`STRUCTURAL_UNCERTAINTIES`](parameters.py#L140-L160): demand growth (`u_demand`), PT preference (`u_beta_pt`). A draw `u∈[0,1]` maps to a standard-normal quantile `z`, held constant along the trajectory while its mean/SD evolve — a coherent future, not annual noise. Trajectory shapes (linear, early, late, logistic, flat, mean-reverting walk): [pathways.py#L214-L282](pathways.py#L214-L282).

**Parameter** — [`PERTURBABLE_PARAMS`](parameters.py#L67-L97) sampled via [`sample_perturbed_params()`](simulation_engine.py#L201-L212) (uniform draw → normal realization around nominal, configurable SD fraction). [`get_ema_uncertainties()`](simulation_engine.py#L183-L198) exposes both as EMA Workbench `RealParameter`s.

```text
uniform EMA sample u ──► inverse-normal transform z
   ▼                                    ▼
demand/PT trajectory              perturbed parameter
(mean_t + z·σ_t)                  (nominal + z·10%)
   └───────────────┬────────────────────┘
                    ▼
            40-year pathway run ──► performance, timing, costs, NPC
```

## Inputs and outputs

| Input | Form | Used for |
|---|---|---|
| Zones | GeoDataFrame | Spatial selection, corridor definition, population/employment |
| Baseline demand | Square OD DataFrame | Total trips before mode allocation |
| Travel-time skims | Dict of square DataFrames | Car/walk/bike/PT in-vehicle, access, egress, wait, transfer time |
| Distance skims | Dict of square DataFrames | Monetary costs, VKT/PKM, fares, mode availability |
| Road network | Nodes, directed edges, zone-to-node map | Optional congestion assignment |
| Stage specification | Nested dictionary | Physical interventions and behavioural overrides |

Bridge between layers — dict from [`extract_corridor_metrics()`](transport_model_interface.py#L676-L700):

```python
stage_metrics = {
    0: {"total_trips": ..., "car_trips": ..., "pt_trips": ..., ...},
    1: {...},
    2: {...},
}
```

[`run_pathway()`](pathways.py#L289-L300) returns `(results, metadata)`: `results` is a 40-row DataFrame (stage, demand, shares, trips, delay, physical indicators, disaggregated costs, CapEx, OpEx, total cost); `metadata` records adaptive-decision and stage-activation years. These feed the EMA Workbench analyses, robustness metrics, tipping-point analysis, and appraisal notebooks.

## Where students should make changes

| If you want to change… | Edit… | Relevant section |
|---|---|---|
| Horizon, corridor, thresholds, capacity, values of time, emissions, costs | [`parameters.py`](parameters.py) | [general/corridor](parameters.py#L18-L58), [economic](parameters.py#L61-L97), [engineering](parameters.py#L106-L129) |
| Infrastructure scope or its effect on time/access/wait/transfers/attractiveness | [`stages.py`](stages.py) | [`get_stages()`](stages.py#L108-L331) |
| Strategy timing | [`pathways.py`](pathways.py) | [`get_pathways()`](pathways.py#L98-L202) |
| Adaptive signpost, threshold, persistence, lead time | [`pathways.py`](pathways.py) | [`DEFAULT_TRIGGERS`](pathways.py#L46-L69) |
| Annual cost or indicator formula | [`simulation_engine.py`](simulation_engine.py) | [`annual_mode_costs()`](simulation_engine.py#L51-L113), [`annual_physical_indicators()`](simulation_engine.py#L116-L143) |
| Congestion method or convergence settings | [`transport_model_interface.py`](transport_model_interface.py) | [`ASSIGNMENT_SETTINGS`](transport_model_interface.py#L48-L57) |

Don't edit `transport_model_interface.py` or `IP_course_FSM-main` internals unless the exercise calls for an advanced extension — they're shared mechanics; project assumptions belong in `parameters.py`, `stages.py`, `pathways.py`.

## Important modelling assumptions

1. **Strategic, not predictive** — compares strategies across plausible futures, not a single forecast.
2. **The long-term layer is a surrogate** — mode choice is precomputed by stage; yearly demand is mainly scaled from stage metrics.
3. **PT-preference trajectories apply after annual costs** — they adjust reported shares/trips in [`pathways.py`](pathways.py#L393-L405) after [`simulate_year()`](simulation_engine.py#L242-L307) has already costed the base stage metrics.
4. **Congestion is separate from free-flow metrics** — LUT/MSA delay is added after free-flow car time from stage metrics.
5. **LUT accuracy depends on its source assignments** — missing/unreadable → zero extra delay; out-of-range demand clamps to the nearest endpoint.
6. **Road assignment is deliberately coarse** — corridor subnetwork, gate-compressed external demand, OD-volume threshold, limited MSA iterations.
7. **Mode choice and assignment aren't iterated to joint equilibrium** — congested times don't feed back into another mode-choice pass.
8. **Primary mode shares are PKM-based and distance-filtered** — not the same as ordinary trip shares.
9. **Triggers react with a one-year observation lag**, then incur their configured lead time.
10. **Units must stay consistent** — minutes in skims, hours in aggregate cost inputs, CHF in annual costs, MCHF in NPC.

State these simplifications when presenting findings, and stress-test them via sensitivity/robustness analysis where they could affect a recommendation.
