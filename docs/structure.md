# Project Structure

This repository is designed to support a clear project pipeline from raw transport data to notebook-based analysis and phase documentation.

## Folder layout

- `data/raw/`
  - Store original files provided by the project team.
  - Contains OD matrices, utility coefficients, formulas, and other source data.
- `data/processed/`
  - Store cleaned and exported datasets.
  - Includes CSV, Parquet, and other intermediate files.
- `docs/`
  - General documentation and phase exercise materials.
  - One PDF exercise sheet per lecture/phase plus `additional/` subfolders for extra resources.
- `notebooks/`
  - Notebook-based workflow for setup, demand, policy, simulation, scenario discovery, and evaluation.
- `code/`
  - Reusable Python modules for data loading, validation, preprocessing, and notebook utilities.
- `figures/`
  - Export-ready figure files and visualizations.

## Phase structure

Each phase contains:
- one exercise PDF under the phase folder
- one `additional/` folder with supplementary material

Phases:
1. Introduction to case study
2. System modeling
3. Deep uncertainty, scenarios, robust decision making
4. Adaptive planning, real options
5. Appraisals

## Notebook workflow

1. `notebooks/01_setup_and_config.ipynb`
   - Install and import packages.
   - Configure the environment and document requirements.
2. `notebooks/02_demand_and_modal_split.ipynb`
   - Estimate demand and compute modal split.
   - Include export-ready visualizations.
3. `notebooks/03_policy_design.ipynb`
   - Model four predefined policy pathways.
   - Allow optional student extensions.
4. `notebooks/04_simulation_model.ipynb`
   - Define simulation horizon and interaction logic.
5. `notebooks/05_ema_scenario_discovery.ipynb`
   - Use EMA Workbench scenario discovery on uncertainty.
6. `notebooks/06_evaluation_ahp.ipynb`
   - AHP evaluation with stakeholder preferences and robustness.
7. `notebooks/07_evaluation_cba.ipynb`
   - CBA evaluation with distributions, ranks, and value evolution.

## Code roles

- `code/data_loader.py`
  - Helpers to load raw data, including OD matrices.
- `code/preprocessing.py`
  - Cleaning and normalization routines.
- `code/validation.py`
  - Sanity checks for modal split, probability sums, and error detection.
- `code/notebook_writer.py`
  - Utilities for notebook generation and exports.
