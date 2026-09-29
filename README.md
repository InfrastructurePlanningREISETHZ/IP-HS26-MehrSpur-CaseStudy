<p align="center">
  <img src="figures/mehrspur_corridor_map.jpg" alt="SBB MehrSpur Zürich–Winterthur Corridor and Brüttenertunnel" width="100%">
  <br>
  <em>Figure: SBB MehrSpur project layout between Zürich and Winterthur, showing the planned 9 km Brüttenertunnel. Source: <a href="https://news.sbb.ch/de/019d7b77-a0a9-780c-972b-6e31f13102ac/gruenes-licht-fuer-grossprojekt-mehrspur-zuerich-winterthur">SBB News</a>.</em>
</p>



# Infrastructure Planning HS2026 — MehrSpur Exercise

This repository contains the teaching material and coded reference case for ETH's Infrastructure Planning course (HS2026 | [VVZ course description](https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=206119&semkez=2026W&ansicht=LEHRVERANSTALTUNGEN&lang=de)). The **SBB MehrSpur Zürich–Winterthur** project is used to demonstrate transport modeling, analysis under uncertainty, adaptive planning and appraisal over a 40-year horizon.

### The case study at a glance

The reference model compares two investment packages in the Zürich–Winterthur corridor. It represents four infrastructure configurations:

- **Baseline (state 0):** The existing transport system without either investment package.
- **Stage 1 – Local Stations & Access Package (state 1):** Railway and mobility-hub improvements at Dietlikon, Bassersdorf and Wallisellen.
- **Stage 2 – Tunnel & Winterthur Hub only (state 2):** The Brüttenertunnel package and Winterthur hub improvements.
- **Both stages – Tunnel and Hubs (state 3):** Both packages, including their combined effects.

The packages can be built and opened independently. Deployment plans specify when each opens, either at a fixed date or after an adaptive trigger.

The material is organized into five phases. The **released exercise sheet is the primary source for tasks, expected outputs and submission requirements**. The notebooks provide examples and a coded workflow for your group's project.

> **Release schedule:** Course documents and exercise sheets are released phase by phase through the course channels and the [phase folders](docs/README.md). Pull the latest repository version when a new phase begins. Material for later phases may be incomplete or subject to change until released.

---

**Contents**

1. [Getting started with the repository, IDEs, and Python](#getting-started)
2. [Course workflow](#course-workflow)
3. [Repository structure](#repository-structure)
4. [Reporting bugs and getting help](#reporting-bugs-and-getting-help)

---

<a id="getting-started"></a>
## Getting started with the repository, IDEs, and Python

Already comfortable with Python, Git and Jupyter? Go directly to the [course workflow](#course-workflow).

<a id="first-time-setup"></a>
<details>
<summary><strong>First-time setup: installation, forking, cloning, and Python environment</strong></summary>

### 1. Install the required software

Install the following before the first exercise:

- **[Python 3.11](https://www.python.org/downloads/release/python-3119/) (Required version: Python 3.11 or 3.12)**  
  *(Windows quick install: `winget install Python.Python.3.11`)*
  > [!WARNING]
  > **Do not use Python 3.13+!**  Please stick to **Python 3.11** (or 3.12).
- [Visual Studio Code](https://code.visualstudio.com/) or another Python IDE of your choice
- If using VS Code, its **Python** and **Jupyter** extensions
- [Git](https://git-scm.com/downloads/)
- [Git LFS](https://git-lfs.com/) for the large transport inputs and surrogate files

<a id="group-repository"></a>
### 2. Fork the course repository and clone it

We use the standard "Fork & Upstream" workflow. You will create a personal copy (a fork) of the public course repository on your own GitHub account. Each group member can then work in a local clone of this fork. When the teaching team releases new course material, you will pull those updates from the main course repository (upstream) into your local copy, and then push them to your fork (origin).

*(Note: A GitHub fork is always public. If your group requires a private workspace, please follow the [Alternative Private Group Repository Setup](docs/alternative_group_repo_setup.md) instead of these steps.)*

1. **Fork the repository**: Go to the main [course repository on GitHub](https://github.com/InfrastructurePlanningREISETHZ/IP-HS26-MehrSpur-CaseStudy) and click the **Fork** button in the top right. This creates a public copy under your own account.
   - For group work, one member creates the fork and invites the other members as collaborators via **Settings → Collaborators → Add people**.
2. **Clone the fork**: Open your terminal or VS Code and run the following commands to clone your fork and set up the connection to the main course repository. Replace `<Your-Username>` with the GitHub username of the fork's owner:

```bash
git lfs install
git clone https://github.com/<Your-Username>/IP-HS26-MehrSpur-CaseStudy.git
cd IP-HS26-MehrSpur-CaseStudy
git remote add upstream https://github.com/InfrastructurePlanningREISETHZ/IP-HS26-MehrSpur-CaseStudy.git
git lfs pull
git remote -v
```

Check that `origin` points to your **fork** and `upstream` points to the **official public course repository**. Push your group's work to `origin`. Fetching or pulling from `upstream` will not overwrite your local work, but rather bring in new updates.

Git LFS downloads the large data files. If you already have a clone, run `git lfs install` and `git lfs pull` inside it to retrieve those files.

Before updating, save and commit your work or stash unfinished changes. If a command reports a merge conflict, stop and resolve it before continuing; ask the teaching assistants for help if needed.

At the beginning of each phase, **one group member** incorporates the latest course changes into your fork:

```bash
git switch main
git pull --no-rebase upstream main
git lfs pull
git push origin main
```

After that member has pushed the update, the other members update their local copies:

```bash
git switch main
git pull --no-rebase origin main
git lfs pull
```

If you stashed unfinished work, restore it with `git stash pop` after updating and resolve any conflicts. Coordinate edits to the same notebook to reduce merge conflicts.

### 3. Create and activate a Python environment

A virtual environment keeps the course packages separate from your system Python. Run the following commands from the repository root.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> [!TIP]
> **Windows Troubleshooting:**
> - If `py -3.11 -m venv .venv` outputs `No suitable Python runtime found`, check installed versions with `py --list`. Install Python 3.11 with `winget install Python.Python.3.11`.
> - If `Activate.ps1` gives an execution policy error (`running scripts is disabled on this system`), run:  
>   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try activating again.

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open the repository folder in VS Code. Select `.venv` as the Python interpreter and, using **Select Kernel** in each notebook, as the notebook environment. An existing environment with the required packages can also be used.

### 4. Check the case-study configuration

Run this from the repository root in your selected environment:

```bash
python code/validate_case_study.py
```

This checks the model settings, configured corridor and counting section, and compatibility of saved surrogate files. It does not run a transport assignment. Notebook 02 also checks the transport inputs when loading the model.

</details>

---

<a id="course-workflow"></a>
## Course workflow

For each phase:

1. Save your work and pull course updates into your fork using the [repository instructions](#group-repository).
2. Open the released exercise sheet from the course channels or `docs/phase-*/`.
3. Read the complete sheet; it defines the required tasks, deliverables and submission requirements.
4. Work through the corresponding reference notebook.
5. Adapt the methods to your group's case study.

### Submitting your group's repository

For each submission, follow the released exercise sheet and include your group's repository URL, all members' names and the **full commit SHA** identifying the version to review. A repository URL alone can point to work that changes after submission.

Commit all required files and push the submitted work to your fork's `main` branch. Once the group has agreed on the final version, run:

```bash
git switch main
git push origin main
git rev-parse HEAD
```

Only submit after the push succeeds. Submit the printed SHA and repository URL through the submission channel specified in the exercise sheet. Check on GitHub that the commit exists and contains all required deliverables; local, untracked or ignored files are not included. Keep the repository and submitted commit available until grading is complete, even if you continue working on later phases.

*(If you chose the alternative private repository setup, remember that you must invite the lecturer and teaching assistants as collaborators so they can access your submission. A URL alone does not grant access to a private repository.)*

Pull requests to the official course repository are for corrections to course material; keep submissions and grading feedback in your fork or the course submission channel.

### Reference notebooks

| Phase | Exercise material | Reference notebook | Main topic |
| --- | --- | --- | --- |
| 1 | [Introduction](docs/phase-1-introduction/) | No notebook | Problem framing, stakeholders and criteria |
| 2 | [System modeling](docs/phase-2-system-modeling/) | [02_system_modeling.ipynb](notebooks/02_system_modeling.ipynb) | Transport model, infrastructure stages and deterministic comparisons |
| 3 | [Uncertainty and scenarios](docs/phase-3-uncertainty-scenarios-rdm/) | [03_uncertainty_modeling.ipynb](notebooks/03_uncertainty_modeling.ipynb) | Surrogate preparation, uncertainty, sensitivity and scenario discovery |
| 4 | [Adaptive planning](docs/phase-4-adaptive-planning-real-options/) | [04_adaptive_planning.ipynb](notebooks/04_adaptive_planning.ipynb) | Trajectories, triggers, deployment plans and pathways |
| 5 | [Appraisal](docs/phase-5-appraisals/) | [05_appraisal.ipynb](notebooks/05_appraisal.ipynb) | Costs and benefits, robustness and plan comparison |

Follow the notebooks in this order. Notebook 02 runs the native transport model and prepares the corridor network. Notebook 03 defines the uncertainty ranges and prepares or loads the shared surrogate and response table. Notebooks 04 and 05 reuse that transport response and calculate their own futures and plan results; Notebook 05 does not require Notebook 04's exported results.

In Notebook 03, keep `REBUILD_TRANSPORT_SURROGATE = False` and `REFRESH_RESPONSE_TABLE = False` to load compatible saved files. Training is required after changes to the underlying transport model or its input domain. See the [surrogate guide](code/surrogate_model/README.md) for when to rebuild, refresh only the response table, or rerun appraisal. The [notebook guide](notebooks/README.md) summarizes dependencies and rerun instructions.

---

<a id="configuration-and-files"></a>
<a id="repository-structure"></a>
## Repository structure

```text
IP-HS26-MehrSpur-CaseStudy/
├── code/               Model configuration and implementation
│   ├── transport_core/ Transport inputs, mode choice and interventions
│   ├── surrogate_model/ Surrogate training and response-table preparation
│   └── additional/     Supporting calculations, plots and widgets
├── data/
│   ├── transport/      Prepared inputs, raw sources, configuration and routing graphs
│   └── processed/      Derived corridor, section and surrogate files
├── docs/               Phase exercise sheets and supplementary resources
├── notebooks/          Reference notebooks for Phases 2–5
├── figures/            Generated figures
├── results/            Exported analysis results
└── requirements.txt    Python dependencies
```

The main case-study settings are in **parameters.py**, **stages.py** and **adaptive_planning.py**. Their comments describe the supported settings and give examples.

| File or folder | Purpose |
| --- | --- |
| [code/README.md](code/README.md) | Guide to configuration and calculation modules |
| [parameters.py](code/parameters.py) | General parameters, corridor and counting section, external flow and uncertainties |
| [stages.py](code/stages.py) | Package interventions, combined effects, investment costs, construction emissions and asset lifetimes |
| [adaptive_planning.py](code/adaptive_planning.py) | Plan schedules, construction lead times and adaptive triggers |
| [simulation_engine.py](code/simulation_engine.py) | Annual physical indicators, costs, benefits and discounted appraisal |
| [transport_model_interface.py](code/transport_model_interface.py) | Transport data loading, coupled mode choice and road assignment, and corridor indicators |
| [transport_core/](code/transport_core/README.md) | Shared transport-model functions |
| [surrogate_model/](code/surrogate_model/README.md) | Surrogate and response-table preparation and use |

The [transport-data guide](data/transport/README.md) describes supplied inputs and raw sources. The [processed-data guide](data/processed/README.md) identifies derived files required for analysis and files used for diagnostics. Phase folders in `docs/` contain exercise sheets as they are released and an `additional/` folder for supporting material.

### Working with notebooks

- Open an `.ipynb` file in VS Code, or start Jupyter from the repository root with `jupyter lab`.
- Confirm that the selected kernel uses your project environment.
- Run cells from top to bottom; later cells depend on variables created earlier.
- After editing imported Python scripts, restart the kernel and rerun from the top. Notebook-only display selections need only their configuration and affected display cells rerun.
- Figures and analysis exports are saved in `figures/` and `results/`. Keep supplied transport inputs unchanged unless you are deliberately adapting the transport data for your case study.
- After changing case-study settings, run `python code/validate_case_study.py` before starting lengthy calculations.

---

<a id="reporting-bugs-and-getting-help"></a>
## Reporting bugs and getting help

### Reporting bugs and proposing improvements

To propose a correction to the course material, you can use your fork to submit a Pull Request to the public course repository.

1. In your local clone, create a correction branch from the course's current default branch (`upstream/main`). Apply only the relevant correction, keeping your group's solution out of this branch.
2. Make and commit the change with a clear explanation.
3. Push the correction branch to your fork and open a Pull Request against the course repository.

Keep each Pull Request focused on one issue. Include only relevant files, without unrelated generated outputs or changes to your group's solution.

### Getting help

If you encounter a problem with setup or an analysis:

1. Check the exercise sheet, notebook instructions and complete error message.
2. For configuration or saved-artifact problems, run `python code/validate_case_study.py` and read its report.
3. Contact the teaching assistants with your operating system, Python version, notebook section or failing command, complete error traceback and what you already tried.

Deadlines and deliverables are defined in the released exercise sheets and communicated through the course channels.
