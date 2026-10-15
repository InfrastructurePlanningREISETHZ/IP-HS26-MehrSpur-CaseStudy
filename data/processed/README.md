# Derived model data

These files support model calculations. Exported analysis results are stored separately in `results/`.

| Files | Purpose |
| --- | --- |
| `mehrspur_detailed_network.pkl` | Detailed corridor road network created or reused in Notebook 02. A different project can select its own `DETAILED_NETWORK_FILE` in `parameters.py`. |
| `section_coverage.npz`, `section_coverage.json` | Prepared MehrSpur counting coverage and matching input/definition metadata. Keep this pair together. |
| `transport_surrogate/manifest.json`, `od_emulator.joblib`, `response_table.npz` | Matching surrogate and response-table artifacts loaded by Notebooks 03–05. |
| `transport_surrogate/design.json`, `msa_design_and_outputs.parquet`, `validation.json`, `od_validation.npz` | Training design, reference results and validation diagnostics. Notebook 03 uses the saved diagnostics. |
| `transport_surrogate/msa_cache/` | Cached reference simulations used during training. Not required to evaluate a completed surrogate. |

Prepare or refresh the surrogate artifacts in Notebook 03. Keep the model, manifest and response table from the same build together. See [surrogate instructions](../../code/surrogate_model/README.md).

For counting-section preparation and calibration, use [section_flows.py](../../code/additional/section_flows.py) with `--help`. The calibration report suggests an external-flow count; it does not edit `parameters.py`.

Counting definitions are optional. The MehrSpur example loads the supplied coverage pair through `SECTION["coverage_file"]` in `parameters.py`. Missing coverage or changed routing inputs/definitions trigger automatic regeneration. For another project, choose its own coverage filename or set `coverage_file=None` to use `cache/sections/`. The shared routing inputs are required to validate or rebuild coverage. The section definition is in `parameters.py`; a separate `section_config.json` is not used.
