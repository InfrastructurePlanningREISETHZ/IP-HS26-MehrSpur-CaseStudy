# Transport surrogate and response table

Notebook 03, Section 3.1 prepares the transport response used by Notebooks 03–05. One Gaussian process (GP) predicts extra car travel-time minutes for each directional OD pair. Operating states use one-hot encoding. Predicted delays feed native mode choice; a saved response table stores physical indicators and matched OD/submode welfare summaries for fast planning queries.

## Preparation

Notebook 03 exposes these controls:

| `REBUILD_TRANSPORT_SURROGATE` | `REFRESH_RESPONSE_TABLE` | Action |
| --- | --- | --- |
| `False` | `False` | Load the saved GP and response table. |
| `False` | `True` | Recreate the response table using the saved GP. |
| `True` | Either | Train the GP and create its response table. |

Both flags default to `False`. Training defaults are **100 training simulations**, **20 validation simulations** (`VALIDATION_FRACTION = 0.20`) and **4 workers**. The Latin-hypercube design covers all four operating states. Reference assignments check feasible-road relative gap every four iterations, stopping at ≤2% after at least eight iterations, with a 64-iteration maximum. Reduce workers if memory is limited. Training can take hours.

Optional uncertainty-guided fill-in can add a limited number of reference simulations when enabled in Notebook 03. It selects new inputs using GP uncertainty and rechecks the fixed validation set after each batch; the validation set is not used to fit the GP.

Training bounds include nominal transport trajectories and `transport_modelling` uncertainties in [parameters.py](../parameters.py). They use bounded distributions' full support or each unbounded distribution's central 99.8%. This is not joint coverage of an entire future. Notebook 03 displays the bounds; scenario values outside them are clipped with a warning.

## When to rerun

- **Rebuild both:** changed transport data, corridor/network, mode-choice assumptions, physical interventions, section coverage, or surrogate input domain.
- **Refresh the table only:** changes confined to reported physical indicators, matched-welfare aggregation, strategic-distance filter or interpolation grid, with unchanged GP transport equations and inputs.
- **Rerun appraisal only:** changed monetary values, discounting, peak hours, external passenger count, comfort capacity, CAPEX, asset lifetime, construction emissions, or deployment rules.

Missing or incompatible artifacts require preparation. Accuracy or reference-convergence failures produce warnings; review Notebook 03 diagnostics before interpreting results. Editing future training settings does not retrain saved artifacts.

## Saved files

Outputs are in [data/processed/transport_surrogate/](../../data/processed/transport_surrogate/):

- `od_emulator.joblib`: fitted OD-delay GP.
- `response_table.npz`: prepared physical indicators and welfare summaries.
- `manifest.json`: input domain, compatibility information and build diagnostics.
- `validation.json`, `od_validation.npz`, `msa_design_and_outputs.parquet`: validation results, reference/predicted OD arrays and simulation diagnostics.

Keep files from the same build together. Normal planning loads the response table; the larger GP is loaded when direct OD prediction or table preparation needs it. `load_transport_surrogate()` and `predict_transport_state()` provide the shared interface.
