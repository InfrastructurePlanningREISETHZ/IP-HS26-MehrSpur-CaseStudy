# Guide: MehrSpur Simulation Codebase

This folder contains the Python code for simulating and evaluating transport infrastructure investments on the Zürich–Winterthur corridor over a 40-year planning horizon.

### What this code does

Evaluating long-term infrastructure involves answering two core questions:
- **How does traffic respond to an infrastructure project?** When new railway tracks or highway lanes are added, how do travel times change, and how many travelers switch between driving and public transit?
- **How does an investment strategy perform over 40 years?** When should projects be built under uncertain future conditions—such as changing population growth, shifting travel habits, and rising carbon costs?

The simulation engine evaluates long-term pathways by scaling demand year by year, checking adaptive triggers, and updating link congestion and lifecycle costs via iterative MSA route assignment.


---

## 📑 Inhaltsverzeichnis (Codebase Directory)

* **[Files you edit](#files-you-edit)**
  * [⚙️ `parameters.py`](#parameterspy) — Central registry for boundaries, costs, economic valuation, and uncertainty ranges.
  * [🏗️ `stages.py`](#stagespy) — Physical stage packages, railway interventions, road capacities, and speed skims.
  * [📈 `adaptive_planning.py`](#adaptive_planningpy) — 40-year pathway simulation, dynamic decision triggers, and deployment plans.
  * [🔬 `simulation_engine.py`](#simulation_enginepy) — Annual cost accounting, physical indicator formulas, and discounted NPC.

* **[Files you don't touch & Utilities](#files-you-dont-touch)**
  * [🧠 `transport_model_interface.py` & `IP_course_FSM-main/`](#transport_model_interfacepy--ip_course_fsm-main) — Core discrete mode choice (MNL) and traffic assignment (MSA) engine.
  * [⚡ `generate_luts.py`](#generate_lutspy) — Standalone precomputor for multi-modal Look-Up Tables across demand scales.

* **[Key Modelling Assumptions](#key-modelling-assumptions)** — Spatial cordon limits, distance filtering, and lead-time rules.

---


### Data flow between files

```text
parameters.py ──► stages.py, adaptive_planning.py, simulation_engine.py
stages.py ─┐
           ├──► transport_model_interface.py ──► stage metrics
IP_course_FSM-main ─┘
stage metrics ──► adaptive_planning.py ⇄ simulation_engine.py (40-yr loop)
                       ▼
        40-year DataFrame ──► discounted NPC & robustness
```

### How an infrastructure stage is evaluated (Spatial Transport Pipeline)

When `run_simulation()` is called for an infrastructure stage, the transport model processes the network through the following sequence:

```text
FSM inputs (1,223 zones, baseline OD demand, multimodal skims, road graph)
   ▼
Select infrastructure stage (Stage 0, 1, or 2 from stages.py)
   ▼
Generate scenario demand & Furness-balance OD matrices
   ▼
Apply stage interventions
   ▼
Calculate preliminary mode utilities
   ▼
Multinomial Logit (MNL) mode choice
   ▼
Route assignment & congestion method
   ├── LUT (Fast mode)
   │    └── Interpolate stage-specific delay from precomputed BPR table
   │
   └── MSA (Demand-responsive feedback)
        └── Assign car OD to local road network via shortest paths
             ▼
             Compute congested link travel times via BPR formula
             ▼
             Feed congested times back into mode choice
             ▼
             Recalculate utilities & converge on final multimodal split
   ▼
Extract final corridor metrics (trips, modal shares, PKM, etc.)
   ▼
Pass stage metrics to the 40-year pathway simulation engine
```


The codebase is split into files you configure for your assignments and backend files that handle the underlying transport model.

## Files you edit

### [`parameters.py`](parameters.py)

**What this file does**
This file acts as the single central registry for all assumptions, geographic definitions, and valuation parameters. Every module in the simulation imports from here so that key assumptions are managed in one place.

While pre-configured with data from the SBB MehrSpur Zürich–Winterthur case study, it provides a general template covering:
- **Corridor boundaries:** Defines which municipalities or zones fall within the study area (`CORRIDOR_REGIONS`, `CORRIDOR_MUNICIPALITIES`).
- **Project economics & service levels:** Specifies capital investment costs (`C_INV_STAGE...`), annual maintenance costs (`C_OP_STAGE...`), flexibility option premiums (`C_FLEX`), as well as train capacities (`CAPACITY_STAGE...`) and service headways (`HEADWAY_STAGE...`) for each stage.
- **Socio-economic valuation:** Sets standard Swiss cost rates for travel time, vehicle fuel, carbon prices, the discount rate (`DISCOUNT_RATE`), and road externalities (noise, air pollution, accidents) according to NIBA norms.
- **Policy benchmarks:** Establishes planning acceptability thresholds, such as maximum acceptable corridor travel time (`MAX_AVG_TT = 20` min) and target transit mode share (`PT_SHARE_TARGET = 0.35`).
- **Uncertainty registries:** Formally distinguishes between scalar baseline parameters and dynamic 40-year trajectories:
  - `FIXED_PARAMS` & `UNCERTAIN_PARAMS`: Register the deterministic baseline and scalar parameters. Combined into `NOMINAL_PARAMS`.
  - `STRUCTURAL_UNCERTAINTIES`: Registers the deep 40-year macro trajectory envelopes (start/end anchors and dispersion $\sigma$) for demand growth (`u_demand`), transit preference shifts (`u_pt_affinity_growth`), travel time valuations (`u_C_TT_PT`, `u_C_TT_CAR`), carbon costs (`u_C_CO2`), and capital cost overruns (`u_costs`).
- **Parameter validation:** Runs `validate_params()` upon import to ensure custom inputs remain within physically and economically sound bounds.


**What you will change**
You will edit this file to tailor the simulation to your own project scope, economic assumptions, and policy scenarios:
- **Redefining your study corridor:** Replace the default Zürich–Winterthur municipalities in `CORRIDOR_REGIONS` and `CORRIDOR_MUNICIPALITIES` with the specific municipalities or zones that define your project's corridor.
- **Updating project budgets & operations:** Enter your project's investment costs, operating expenditures, capacities, and service headways for each stage.
- **Setting policy & performance targets:** Adjust planning thresholds (`MAX_AVG_TT`, `PT_SHARE_TARGET`) to evaluate whether your strategy meets your specific planning goals.
- **Testing sensitivity and exploring long-term scenarios:**
  - Adjust baseline values in `UNCERTAIN_PARAMS` to shift central anchors.
  - Modify trajectory ranges, nominal endpoints (e.g., `DEMAND_GROWTH_Y40`), and standard deviations (`DG_SIGMA_1`, `DG_SIGMA_40`) in `STRUCTURAL_UNCERTAINTIES` to evaluate performance under varying 40-year futures.


---

### [`stages.py`](stages.py)

**What this file does**
This file specifies how each infrastructure stage physically and operationally alters the regional transport network. It acts as the direct bridge between your project concepts and the underlying travel model skims.

- **Behavioral & discrete choice parameters:** Overrides mode choice constants (such as ASCs, time/cost betas) and stage-level baseline mode attractiveness multipliers (such as base `pt_affinity`, onto which annual societal `pt_affinity_growth` is added dynamically).
- **Spatial targeting:** Specifies the geographic scope of interventions using corridor OD links (`area_pairs`) or area-wide station catchments (`zones`).
- **Physical network interventions:** Modifies travel times, speeds, and distances for specific transport networks via mode keys like `railway_expansions`, `road_capacity`, and `bike_highways`.
- **Local improvements:** Creates locally explicit improvements (`mobility_hubs` in the MehrSpur Project).


**What you will change**
You will edit this file to configure the specific interventions and stage definitions for your own project:
- **Defining stage packages:** Adapt the infrastructure stages in the `stages` dictionary (`0`, `1`, `2`, ...) to represent your own project's planned interventions.
- **Applying spatial interventions:** Assign your physical interventions to the network by either applying corridor-wide links (reusing the municipalities/zones from `parameters.py`) or picking specific regions in `zones` (e.g., targeting individual station hubs for upgrades).
- **Applying physical improvements:** Set percentage reductions such as in-vehicle travel times, distance changes, etc. under the relevant transport mode.
- **Adjusting mode preferences:** Modify mode affinity multipliers or logit choice parameters (`ASC_...`, `B_...`) to simulate service quality changes (e.g. improved reliability, comfort, or ticketing).


---

### [`adaptive_planning.py`](adaptive_planning.py)

**What this file does**
This file defines deployment plans, adaptive signpost triggers, and runs the dynamic 40-year simulation loop.

- **Standardized deployment plans:** Defines *when* each stage opens over the 40 years in `get_plans()`. Pre-configures 5 canonical plans:
  - `baseline`: No investment across 40 years (Stage 0 throughout).
  - `static`: Full build upfront in Year 1 (Stage 2 throughout).
  - `staged1`: Phased delivery (Stage 1 in Year 9, Stage 2 in Year 13) (SBB Approach).
  - `staged2`: Alternative phased delivery (Stage 1 in Year 10, Stage 2 in Year 20).
  - `flexible`: Fully adaptive pathway (Stage 1 via Trigger 1; Stage 2 via Trigger 2).
- **Adaptive triggers & signposts:** Specifies operational action thresholds in `get_triggers()` that monitor real-time indicators (like `pt_trips`, `avg_tt_min`, `pt_share`, or `congestion_delay_hours`), accounting for consecutive confirmation years (`persistence`) and construction `lead_time`.
- **40-Year simulation runner:** Executes the annual simulation loop in `run_plan()` and `run_plan_from_trajectories()`, managing demand scaling, applying annual transit preference growth (`pt_affinity_growth`), checking trigger rules, logging physical indicators, and recording lifecycle investment, maintenance, and flexibility option costs.
- **Trajectory generators:** Contains `generate_shaped_trajectory()` to construct 40-year non-linear futures (`linear`, `early`, `late`, `logistic`, `random_walk`).

**What you will change**
You will edit this file to design and evaluate long-term deployment strategies for your own project:
- **Creating investment pathways:** Define new planning strategies or adjust existing transition years in `get_plans()`.
- **Designing adaptive triggers:** Formulate trigger logic for flexible strategies in `get_triggers()` by selecting signpost metrics, setting trigger thresholds, required consecutive years (`persistence`), and construction `lead_time`.

```text
Each simulated year (Year 1 to 40):
   Fixed transition scheduled for this year? ──yes──► Activate new stage
        ▼
   Read previous year's signpost metric (pt_trips, avg_tt_min, delay, etc.)
        ▼
   Signpost > threshold?
        yes ──► persistence += 1 ──► persistence reached? ──yes──► Lock in decision
        no  ──► persistence = 0               (opening year = decision year + lead time)
        ▼
   Opening year reached? ──yes──► Activate new stage
        ▼
   Simulate year ──► add inv. and op. costs ──► record indicators & costs as signposts
        ▼
   40 years complete? ──no──► repeat for next year     yes ──► return 40-year results

```     


---
### [`simulation_engine.py`](simulation_engine.py)

**What this file does**
This module executes the annual simulation and socio-economic evaluation. Called by `adaptive_planning.py` for each year of the 40-year horizon, it applies annual demand growth, determines road congestion delays, costs, and environmental externalities, and aggregates 40-year lifecycle impacts into discounted Net Present Cost (NPC).

- **Annual cost accounting:** Computes yearly monetized costs in `annual_mode_costs()` across travel time, operating expenditure, and NIBA externalities (CO2, noise, air pollution, accidents).
- **Physical performance indicators:** Tracks unmonetized system metrics in `annual_physical_indicators()`, such as average motorized travel times (`avg_tt_min`), annual $\text{CO}_2$ emissions (tonnes), and total bottleneck delay hours.
- **Congestion delay integration:** Fast-interpolates road traffic delays via precomputed Look-Up Tables (`get_lut_delay()`).
- **Economic appraisal (NPC & Rule of a Half):** Discounts 40-year costs back to present value and decomposes lifecycle costs by component in `npc_by_component()`. When evaluated against a baseline run, it applies the transport economic **Rule of a Half** to compute consumer surplus and net user benefits.
- **Uncertainty sampling:** Exposes structural trajectory definitions to the EMA Workbench via `get_ema_uncertainties()`, while `sample_uncertain_params()` safely isolates scalar baseline overrides from 40-year trajectory arrays.


**What you will change**
You will edit this file to adjust or expand how costs, externalities, and appraisal metrics are calculated:
- **Adding or modifying cost formulas:** Update existing equations in `annual_mode_costs()` or introduce entirely new cost categories according to your own project.
  > **Example — Custom non-linear penalties (PT Crowding):**  
  > The codebase demonstrates how to model physical capacity limits using an operational penalty: when peak passenger volume exceeds the line capacity set in `parameters.py`, perceived transit time cost is inflated quadratically:
  > $$\text{crowding multiplier} = \max\left[1.0, \left(\frac{\text{peak trips}}{\text{capacity}}\right)^2\right]$$
  > You can follow this exact pattern in `simulation_engine.py` to add your own custom mechanisms—such as highway bottleneck delays, electric vehicle charging queues, or station platform overcrowding.
- **Customizing physical indicators:** Add or adjust non-monetary metrics in `annual_physical_indicators()` to monitor project-specific performance targets.


---

## Files you don't touch

### [`transport_model_interface.py`](transport_model_interface.py) & [`IP_course_FSM-main/`](../IP_course_FSM-main/)


**What this file does**
Acts as the bridge between your stage specifications and the regional 4-step transport model. It takes care of all low-level computations: modifying multimodal skims (rail, road, bike, walk), calculating discrete choice utilities, performing link-level route assignment (MSA), and extracting corridor demand metrics.
- **Data loading & caching:** `load_transport_context()` runs preflight readiness checks on raw FSM files (`zones.parquet`, `skims.pkl.gz`, `demand.pkl.gz`, `assignment_network.pkl`) and automatically caches the loaded model object to `cache/tmi_context.pkl` for near-instant notebook startup.
- **Cordon Gates & Network Clipping:** For fast traffic assignment, the road network is clipped to the corridor boundary, and external trips are compressed to entry/exit "cordon gates" via `collapse_od_to_gates()`. This preserves realistic traffic loads on corridor highways without having to simulate all 1,223 zones across the entire canton.


**What you will change**
You will not need to touch this file, as all standard interventions can be configured through `stages.py` and `parameters.py`. However, if you want to implement more advanced interventions such as custom network policies or specialized spatial metrics, you can extend `transport_model_interface.py`.
- **Clearing the cache:** If you ever update underlying network skims or zone data, delete `cache/tmi_context.pkl` to force `load_transport_context()` to re-parse and rebuild the cached context.
- **4 vs. 5 Mode Choice:** The underlying discrete choice engine calculates 5 separate alternatives (`drive`, `bike`, `walk`, `pt_walk`, `pt_bike`). By default, outputs aggregate these into the standard 4 modes (`Car`, `PT`, `Bike`, `Walk`), but functions like `corridor_od_summary(..., detailed_pt=True)` allow you to toggle the full 5-mode breakdown to inspect bike-and-ride behavior.


---

### [`generate_luts.py`](generate_luts.py)

**What this file does**
This utility precomputes and exports the multi-modal Look-Up Tables (LUTs) across 13 demand multipliers ($0.8\times$ to $2.0\times$) for all infrastructure stages. (Runtime 5-10 min.)

Instead of running slow, iterative traffic assignment (MSA) during high-volume simulations, the downstream 40-year adaptive planning loops (`Notebook 02` / `Notebook 04`) and uncertainty screening experiments (`Notebook 03`) query these precomputed tables. Each generated JSON file (`data/processed/delay_lut_stage_{id}.json`) stores:
- Congestion bottleneck delay hours (`delay_hours`)
- Balanced multi-modal split (`car_share`, `pt_share`, `bike_share`, `walk_share`)
- Total vehicle kilometers (`vkt`) and vehicle hours of travel (`vht`)

---

**When you must run it**
Because the LUT records stage- and network-specific equilibriums, **you must regenerate the LUTs whenever you modify**:
1. **Corridor or zone definitions:** If you change `CORRIDOR_MUNICIPALITIES` or `CORRIDOR_REGIONS` in `parameters.py` (e.g., adapting the corridor to your group project).
2. **Infrastructure stage interventions:** If you alter travel-time percentage savings, speeds, capacities, or hub parameters in `stages.py`.
3. **Discrete mode-choice parameters:** If you modify baseline alternative-specific constants (ASCs) or cost/time sensitivity betas in `stages.py` or `parameters.py`.

> [!IMPORTANT]
> If you change stages or corridor zones without regenerating the LUTs, your 40-year simulations in Notebooks 03, 04, and 05 will read stale delay and mode shares from the previous network setup!

---

**How to run it**

You can regenerate all LUTs using either the command line or directly inside a notebook:

#### Option A: From the Terminal (Recommended)
Run the script from your project root:
python code/generate_luts.py

#### Option B: From inside a Jupyter Notebook
import transport_model_interface as tmi
tmi.generate_all_luts(ctx)

---

## Key Modelling Assumptions

When presenting your results, keep these system boundaries in mind:


- **Distance filtering:** The headline metrics for car and public transport market shares only count trips longer than a minimum distance (e.g., 5 km), intentionally excluding short local walks from the main comparison. This is MehrSpur Project Specific.
- **Adaptive lag:** Triggers look at the previous year's data to make decisions, avoiding "look-ahead" bias. Once triggered, the physical construction lead time must pass before the new stage opens.
