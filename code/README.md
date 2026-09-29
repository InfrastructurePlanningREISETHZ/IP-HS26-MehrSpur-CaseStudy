# Model code

The code supports transport modeling and 40-year appraisal of the MehrSpur Zürich–Winterthur case. Start with the [notebook guide](../notebooks/README.md); detailed parameter and intervention examples are in the scripts.

## Project settings

> **Student Note:** For your case study, these are the **only** files you should need to edit.

| File | What to configure |
| --- | --- |
| [parameters.py](parameters.py) | General parameters, corridor boundaries, counting section, external flow and uncertainties. |
| [stages.py](stages.py) | Infrastructure packages: `railway_expansions`, `mobility_hubs`, other physical interventions and combined effects. Each package also contains CAPEX, construction emissions and asset lifetime assumptions. |
| [adaptive_planning.py](adaptive_planning.py) | Deployment plans, monitored signposts, trigger thresholds, persistence and construction lead times. |

The two investment packages can open independently. Transport states are **0: baseline**, **1: Stage 1**, **2: Stage 2**, and **3: both stages**. Plan timing determines which state operates each year.

Inside `parameters.py`, you will find `NOMINAL_PARAMS` (for baseline/final values) and `STRUCTURAL_UNCERTAINTIES` (for defining uncertainty ranges). Removing an uncertainty simply means the model will use its nominal, fixed trajectory instead. For a list of all supported keys you can add, check out `parameter_catalogue()` in [additional/uncertainty.py](additional/uncertainty.py).

## Calculation modules

> **Student Note:** You generally do not need to modify these files. They handle the core model logic and processing behind the scenes.

| Module | Role |
| --- | --- |
| [transport_model_interface.py](transport_model_interface.py) | Load transport inputs, apply interventions, run coupled mode choice and road assignment, and extract indicators. |
| [transport_core/](transport_core/README.md) | Input loading, travel-time components, interventions and five-alternative mode choice. |
| [surrogate_model/](surrogate_model/README.md) | Train the OD-delay Gaussian process and prepare the response table used in Notebooks 03–05. |
| [simulation_engine.py](simulation_engine.py) | Annual physical indicators, matched OD/submode travel-time welfare, external costs and discounted appraisal. |
| [additional/](additional/) | Uncertainty paths, section coverage and calibration, plots/widgets and trigger optimization. |
| [validate_case_study.py](validate_case_study.py) | Read-only configuration and saved-artifact checks; no transport simulations. |

Notebook 02 runs the native transport model. Notebook 03 trains or loads the surrogate and response table after defining uncertainty ranges. Notebooks 04 and 05 reuse these outputs for deployment plans and appraisal. See the [surrogate guide](surrogate_model/README.md) for rebuild and refresh rules.

After editing imported Python scripts, restart the notebook kernel and rerun from the top. Display selections made within Notebook 02 can be changed by rerunning their configuration and affected display cells.
