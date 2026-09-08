<p align="center">
  <img src="figures/mehrspur_corridor_map.jpg" alt="SBB MehrSpur Zürich–Winterthur Corridor and Brüttenertunnel" width="100%">
  <br>
  <em>Figure: The SBB MehrSpur project layout between Zürich and Winterthur. Source: <a href="https://news.sbb.ch/de/019d7b77-a0a9-780c-972b-6e31f13102ac/gruenes-licht-fuer-grossprojekt-mehrspur-zuerich-winterthur">SBB News</a>.</em>
</p>

# Infrastructure Planning HS2026 — MehrSpur Exercise

This repository contains the teaching material and coded reference case for ETH's Infrastructure Planning course (HS2026 | [VVZ course description](https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=206119&semkez=2026W&ansicht=LEHRVERANSTALTUNGEN&lang=de)). The **SBB MehrSpur Zürich–Winterthur** project is used to demonstrate how a complex infrastructure project can be modelled, tested under uncertainty, developed into adaptive pathways, and appraised over a long planning horizon.

The material is organised into five phases. For every phase, the corresponding **exercise sheet in `docs/` is the primary source for tasks, expected outputs, and submission requirements**. The notebooks provide examples and a coded workflow that you can use as a reference for your own group project.

> **Release schedule:** Course documents and exercise sheets will be released phase by phase during the semester. Pull the latest version of the repository when a new phase begins. Material for a later phase may be missing, incomplete, or subject to change until that phase is officially released.

**Contents**

1. [Getting started with the course's repository + GitHub, IDEs, and Python](#getting-started)
2. [Repository structure](#repository-structure)
3. [Reporting bugs and getting help](#reporting-bugs-and-getting-help)

<a id="getting-started"></a>
## Getting started with the course's repository + GitHub, IDEs, and Python

Already comfortable with Python, Git, and Jupyter? Skip the installation guide and go directly to the [course workflow](#course-workflow).

<a id="first-time-setup"></a>
<details>
<summary><strong>First-time setup: installation, cloning, and Python environment</strong></summary>

### 1. Install the required software

Install the following before the first exercise:

- [Python 3.11](https://www.python.org/downloads/) (recommended version)
- [Visual Studio Code](https://code.visualstudio.com/) or another Python IDE of your choice
- If you use VS Code, install the **Python** and **Jupyter** extensions
- [Git](https://git-scm.com/downloads/)

### 2. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/InfrastructurePlanningREISETHZ/IP_exercise_HS2026.git  ### @alemaisano, we need to change this before the actual release
cd IP_exercise_HS2026
```

At the beginning of each phase, update your local copy:

```bash
git pull
```

If you have local changes, commit or safely store them before pulling. Keep your own work in a separate copy of this repository so that you can freely work on adapting the project to your case study.

### 3. Create and activate a Python environment

Using a virtual environment keeps the course packages separate from the rest of your system.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, open the repository, select the `.venv` Python interpreter, and use the same environment as the Jupyter kernel. Some later notebooks require additional packages such as EMA Workbench or geospatial libraries; follow the setup instructions in the relevant exercise sheet and notebook when that phase is released.

</details>

### Course workflow

For each phase:

1. Pull the latest repository changes.
2. Open the relevant exercise sheet under `docs/phase-*/exercise.pdf` #
3. Read the complete exercise sheet before starting; it defines the required and optional tasks.
4. Work through the matching notebook in `notebooks/`.
5. Adapt the demonstrated methods to your own case study and prepare the deliverables specified in the exercise sheet.

Run the notebooks in this order:

| Phase | Exercise material | Reference notebook | Main topic |
|---|---|---|---|
| 1 | `docs/phase-1-introduction/` | No notebooks for this phase | Case study, problem framing, stakeholders, and criteria |
| 2 | `docs/phase-2-system-modeling/` | `notebooks/02_system_modeling.ipynb` | System model, infrastructure stages, and plans/pathways |
| 3 | `docs/phase-3-uncertainty-scenarios-rdm/` | `notebooks/03_uncertainty_modeling.ipynb` | (Deep) uncertainty, scenarios, and robust decision making |
| 4 | `docs/phase-4-adaptive-planning-real-options/` | `notebooks/04_adaptive_planning.ipynb` | Adaptive planning, triggers, pathways, and real options |
| 5 | `docs/phase-5-appraisals/` | `notebooks/05_appraisal.ipynb` | Appraisal, robustness, and final strategy comparison |

The notebooks are designed to be read and executed sequentially. Earlier phases establish concepts, model components, and outputs used in later phases. Unless an exercise sheet says otherwise, do not treat notebook examples as ready-made answers for your group project.

## Repository structure

```text
IP_exercise_HS2026/
├── code/           Underlying simulation enging and reusable Python modules for the MehrSpur reference model
├── data/
│   ├── raw/        Original input data
│   └── processed/  Cleaned and derived data
├── docs/           Phase-specific exercise sheets and (potentially) supporting material
├── notebooks/      Numbered reference notebooks for the five phases
├── figures/        Figures generated by the analyses
├── results/        Generated and exported analysis results
├── helpers/        Supporting and legacy utility functions #### @alemaisano check if these are still needed
├── requirements.txt
```

Each phase directory in `docs/` is intended to contain:

- `exercise.pdf`: the exercise sheet for that phase
- `additional/`: supplementary material, where applicable

The main model implementation is in `code/`:

| File | Purpose |
|---|---|
| [`README.md`](code/README.md) | An in-depth explanation of the underlying simulation pipeline, with direct links to the relevant files |
| `parameters.py` | Model assumptions, fixed values, and uncertain parameters |
| `stages.py` | Infrastructure stages and their effects |
| `pathways.py` | Static, staged, and adaptive decision pathways |
| `simulation_engine.py` | Simulation logic across the planning horizon |
| `transport_model_interface.py` | Interface to the underlying transport model |

### Working with notebooks

- Start Jupyter from the repository root with `jupyter lab`, or open an `.ipynb` file directly in VS Code.
- Confirm that the selected kernel is the `.venv` environment before running cells.
- Run cells from top to bottom; later cells often depend on earlier results.
- Paths are set up for running notebooks from either the repository root or the `notebooks/` directory where indicated.
- Generated files belong in `figures/`, `results/`, or `data/processed/`, as appropriate. Do not modify files in `data/raw/`.
- If a notebook contains an additional package-installation cell, run it in the active notebook kernel and restart the kernel if requested.

## Reporting bugs and getting help

### Reporting bugs and proposing improvements

If you identify a bug, error, or potential improvement in the course material, please feel free to submit a **Pull Request** to the course repository. Students have read-only access to the official repository, so changes must be proposed from a fork:

1. Fork the official repository on GitHub.
2. Clone your fork, or add it as a remote to your existing local clone.
3. Create a new branch for the correction.
4. Make and clearly describe the change.
5. Push the branch to your fork and open a Pull Request against the official course repository.

Please keep each Pull Request focused on one issue and avoid including unrelated generated files or changes to your group solution.

### Getting help

First, check the relevant exercise sheet, notebook instructions, and error message. If you encounter problems with the repository, code, setup, or course material and are unable to resolve them, please contact the teaching assistants. Include enough information to reproduce the problem, such as:

- the exercise phase and file you are working with;
- your operating system and Python version;
- the command or notebook cell that failed;
- the complete error message; and
- what you already tried.

In any case, deadlines, deliverables, and phase-specific instructions are defined in the released exercise sheets and communicated through the official course channels.
