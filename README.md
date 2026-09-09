<p align="center">
  <img src="figures/mehrspur_corridor_map.jpg" alt="SBB MehrSpur Zürich–Winterthur Corridor and Brüttenertunnel" width="100%">
  <br>
  <em>Figure: SBB MehrSpur project layout between Zürich and Winterthur, showing the planned 9 km Brüttenertunnel. Source: <a href="https://news.sbb.ch/de/019d7b77-a0a9-780c-972b-6e31f13102ac/gruenes-licht-fuer-grossprojekt-mehrspur-zuerich-winterthur">SBB News</a>.</em>
</p>

# Infrastructure Planning HS2026 — MehrSpur Exercise

This repository contains the teaching material and coded reference case for ETH's Infrastructure Planning course (HS2026 | [VVZ course description](https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=206119&semkez=2026W&ansicht=LEHRVERANSTALTUNGEN&lang=de)). The **SBB MehrSpur Zürich–Winterthur** project is used to demonstrate how a complex infrastructure project can be modelled, tested under uncertainty, developed into adaptive pathways, and appraised over a long planning horizon.

### The Case Study at a Glance
The railway corridor between **Zürich HB** and **Winterthur** is one of the most heavily congested passenger transport axes in Switzerland. Currently, rail traffic converges onto a single double-track line via Effretikon, creating a major capacity bottleneck.

Throughout this course, you will model, test, and appraise three progressive infrastructure stages:
- **Stage 0 (Baseline / Status Quo):** The existing network layout relying on the shared Effretikon bottleneck.
- **Stage 1 (Interim Interventions / Station Upgrades):** Local capacity and station enhancements (e.g. Dietlikon, Bassersdorf, Wallisellen) and mobility hub integration to optimize existing tracks.
- **Stage 2 (Full Expansion / Brüttenertunnel):** Construction of the new 9 km underground double-track Brüttenertunnel, providing a direct bypass and unlocking quarter-hourly (15-minute) express rail service across the corridor.

The material is organised into five phases. For every phase, the corresponding **exercise sheet in `docs/` is the primary source for tasks, expected outputs, and submission requirements**. The notebooks provide examples and a coded workflow that you can use as a reference for your own group project.

> **Release schedule:** Course documents and exercise sheets will be released phase by phase during the semester. Pull the latest version of the repository when a new phase begins. Material for a later phase may be missing, incomplete, or subject to change until that phase is officially released.

---

**Contents**

1. [Getting started with the repository, IDEs, and Python](#getting-started)
2. [Course workflow](#course-workflow)
3. [Repository structure](#repository-structure)
4. [Reporting bugs and getting help](#reporting-bugs-and-getting-help)

---

<a id="getting-started"></a>
## Getting started with the repository, IDEs, and Python

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
git clone https://github.com/InfrastructurePlanningREISETHZ/IP-HS26-MehrSpur-CaseStudy.git
cd IP-HS26-MehrSpur-CaseStudy
```

At the beginning of each phase, update your local copy:

```bash
git pull
```

If you have local changes, commit or stash them before pulling. Keep your group's case study work organized so that you can freely adapt the project code.

### 3. Create and activate a Python environment

Using a virtual environment keeps the course packages isolated from your system Python.

**Windows PowerShell:**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select `.venv` as your Python interpreter and as the Jupyter kernel.

### 4. Verify your setup

Confirm that your environment, dependencies, and data files are ready by running:

```bash
python verify_setup.py
```

</details>

---

<a id="course-workflow"></a>
## Course workflow

For each phase:
1. Pull the latest repository changes (`git pull`).
2. Open the relevant exercise sheet under `docs/phase-*/`.
3. Read the complete exercise sheet before starting; it defines mandatory deliverables, deadlines, and criteria.
4. Work through the corresponding reference notebook in `notebooks/`.
5. Adapt the demonstrated methods to your group's assigned case study.

### Reference Notebooks

| Phase | Exercise Material | Reference Notebook | Main Topic |
|---|---|---|---|
| **1** | `docs/phase-1-introduction/` | *(No notebook)* | Case study, problem framing, stakeholders, and criteria |
| **2** | `docs/phase-2-system-modeling/` | `notebooks/phase2_system_modeling.ipynb` | System model, infrastructure stages, and decision pathways |
| **3** | `docs/phase-3-uncertainty-scenarios-rdm/` | `notebooks/phase3_uncertainty_modeling.ipynb` | Deep uncertainty, scenario discovery, and RDM |
| **4** | `docs/phase-4-adaptive-planning-real-options/` | `notebooks/phase4_adaptive_planning.ipynb` | Adaptive planning, triggers, pathways, and real options |
| **5** | `docs/phase-5-appraisals/` | `notebooks/phase5_appraisal.ipynb` | Appraisal, robustness, CBA, and final strategy comparison |

The notebooks are designed to be executed sequentially. Earlier phases establish model components and outputs required by later phases.

---

<a id="repository-structure"></a>
## Repository structure

```text
IP-HS26-MehrSpur-CaseStudy/
├── code/               Core simulation engine and MehrSpur reference modules
├── data/
│   ├── raw/            Original input data
│   └── processed/      Cleaned and derived data
├── docs/               Phase-specific exercise sheets and resources
├── figures/            Figures generated by the simulation and notebooks
├── notebooks/          Reference Jupyter notebooks for Phases 2–5
├── results/            Exported analysis results
├── IP_course_FSM-main/ Canton Zürich Four-Step Model backend (read-only)
├── requirements.txt    Python dependencies
└── verify_setup.py     Environment and setup verification script
```

The course simulation logic is located in `code/`:

| File | Purpose |
|---|---|
| [`code/README.md`](code/README.md) | Technical architecture guide with links to the simulation pipeline |
| [`parameters.py`](code/parameters.py) | Model assumptions, nominal values, and uncertain parameters |
| [`stages.py`](code/stages.py) | Infrastructure stages (Stage 0, 1, 2) and physical interventions |
| [`pathways.py`](code/pathways.py) | Static, staged, and adaptive deployment pathways |
| [`simulation_engine.py`](code/simulation_engine.py) | Annual simulation loop, congestion delay, and 40-year socio-economic appraisal |
| [`transport_model_interface.py`](code/transport_model_interface.py) | Bridge connecting stage parameters to the regional transport model |

### Working with notebooks

- Start Jupyter from the repository root with `jupyter lab`, or open an `.ipynb` file directly in VS Code.
- Confirm that the active kernel is `.venv`.
- Run cells from top to bottom; downstream cells depend on variables from earlier cells.
- Generated figures and exported data are saved in `figures/` and `results/`. Do not overwrite files in `data/raw/`.

---

<a id="reporting-bugs-and-getting-help"></a>
## Reporting bugs and getting help

### Reporting bugs and proposing improvements

If you identify a bug, typo, or potential improvement, please submit a **Pull Request**:
1. Fork the repository on GitHub.
2. Clone your fork and create a new feature/fix branch.
3. Commit your changes with a clear explanation.
4. Open a Pull Request against the main course repository.

### Getting help

If you encounter issues with code or setup that you cannot resolve:
1. Check the relevant exercise sheet, notebook comments, and [`code/README.md`](code/README.md).
2. Run `python verify_setup.py` to identify missing packages or paths.
3. Contact the teaching assistants, including your OS, Python version, the failing command or cell, and the complete error traceback.
