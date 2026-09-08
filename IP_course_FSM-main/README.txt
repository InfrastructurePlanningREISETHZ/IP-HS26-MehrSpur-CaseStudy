IP COURSE FSM
=============

This repository contains a standalone afternoon-peak transport model for
Canton Zurich. It includes the Python model, prepared runtime packages, and the
public raw datasets needed to rebuild those packages.

Running the model and rebuilding its inputs are separate workflows. Main.py
loads the supplied prepared inputs; it does not start preprocessing or write
to input_data/. prepare_inputs.py is the explicit raw-to-prepared workflow.


QUICK START
-----------

1. Create the environment:

       conda env create -f environment.yml
       conda activate ip-course-fsm

2. Check the supplied runtime inputs:

       python prepare_inputs.py --check-runtime

3. Edit the RUN SETTINGS block near the top of Main.py.

4. Run the model:

       python Main.py

Assignment is enabled by default. For a quick free-flow run:

       python Main.py --no-assignment

Every normal run creates a timestamped folder under outputs/.

To use an exact output directory instead of a timestamped one:

       python Main.py --output-dir outputs/my_scenario

Main.py requires the five files in input_data/prepared/. If one is missing or
invalid, the run stops with an input error. Rebuilds are never launched
automatically because network and GTFS processing can take a long time.


RUN SETTINGS
------------

Main.py contains the settings students normally edit.

RUN_NAME
    Short label used in the output folder name.

RUN_ASSIGNMENT
    True performs road assignment. False uses prepared free-flow road times.
    The --no-assignment command-line option overrides True for a quick run.

APPLY_SCENARIO
    False runs baseline demand. True applies SCENARIO_SETTINGS.

SCENARIO_SETTINGS
    population_multiplier and jobs_multiplier set endpoint totals relative to
    the baseline; the model does not assign a calendar year. The City growth
    shares allocate the net population and job change between the City and the
    rest of the Canton. trip_rate_multiplier changes trip productions. The
    four positive affinity values multiply mode-choice odds; 1.0 means no
    change. ebike_share sets the share of cycling represented by e-bikes.

INTERVENTION_SELECTION
    Select 0 for no intervention, or option 1 or 2 for each intervention type.
    The three choices are independent and can be combined.

Intervention locations and effects are edited near the top of interventions.py.
Each type represents one place with two intensity levels:

* bike highway: Altstetten-Schlieren-Dietikon OD pairs with bicycle paths of
  2-15 km. Level 1 increases cycling speed by 20 percent and level 2 by
  35 percent. Network distance is unchanged;
* railway expansion: City of Zurich-Winterthur OD pairs. Level 1 reduces PT
  in-vehicle time by 8 percent, initial waiting by 20 percent, and transfer
  waiting by 10 percent. Level 2 applies reductions of 15, 40, and 25 percent;
* mobility hub: paths whose baseline selected origin or destination stop is
  Forch station. Level 1 reduces access and egress by 10 percent, physical
  transfer time by 20 percent, and transfer waiting by 10 percent. Level 2
  applies reductions of 25, 25, 40, and 25 percent respectively and also
  reduces initial waiting by 15 percent.

Bike and railway locations use readable attributes from zones.parquet, such
as city_quartier, municipality_name, and Level. Direct NPVM OD pairs remain a
supported alternative. Mobility hubs use explicit GTFS stop IDs so the Forch
examples do not depend on which nearby stop happens to be closest to a zone
centroid. They retain the prepared baseline stop choice; an intervention does
not reroute passengers to a different stop.

Available effects include travel-time reduction, distance reduction, speed
increase, initial-wait reduction, transfer-wait reduction, physical-transfer
reduction, access-time reduction, and egress-time reduction. Effects that do
not apply to a particular intervention are omitted or set to zero.

There are no scenario or intervention JSON files. Scenario values and the
0/1/2 intervention selections belong to the editable block in Main.py;
scenario_generation.py contains the demand mechanics. The three intervention
types and their two levels belong to interventions.py. config/ contains
only the mode-choice coefficients in mode_choice.json.


MODEL SEQUENCE
--------------

1. Load direct NPVM zones, passenger demand, and fixed road-background demand.
2. Apply optional population, jobs, trip-rate, and mode-affinity modifiers.
3. Load direct-mode and public-transport skim components.
4. Apply perceived connector, terminal, intrazonal, and PT floor assumptions.
5. Apply the selected interventions and configured e-bike share (25 percent by
   default).
6. Calculate preliminary mode choice.
7. If enabled, assign car demand and fixed background traffic with ta_lab and
   Frank-Wolfe iterations.
8. If assignment is enabled, recalculate final mode choice with the assigned
   road skim; otherwise use the preliminary result as the final result.
9. For intervention runs, calculate a matched case with all selected
   interventions removed while retaining the same demand and final road skim.
10. Save modal, spatial, distance-bin, accessibility, intervention, and
    assignment outputs.

The assignment is one outer pass: preliminary mode choice, road assignment,
then final mode choice. It is not repeated until mode choice converges.


ZONING AND DEMAND
-----------------

The model uses 1,223 NPVM zones directly. A source NPVM polygon is retained
when more than 40 percent of its area lies within Canton Zurich. Zones whose
centroids lie inside the City of Zurich are marked Level == "Quartier"; all
other retained zones are marked Level == "Canton". The runtime zone table is
input_data/prepared/zones.parquet.

City zones also contain city_quartier, assigned from the public Zurich
quartier boundaries using each NPVM zone centroid. Municipality names are
retained for all zones. These fields make intervention definitions readable
without hard-coding long lists of zone IDs.

Baseline population is allocated from STATPOP 2024 grid cells. Baseline jobs
are allocated from STATENT 2023 grid cells. Passenger demand combines NPVM car,
PT, walking, bicycle, and e-bike matrices for the 17:00-18:00 ASP. Freight and
commercial traffic remain fixed road-background demand. The 80 peripheral NPVM
zones are represented through the documented gateway mapping rather than as
model zones.


NETWORKS AND SKIMS
------------------

Road, walk, and bicycle skims use frozen OSM-derived network snapshots. Bicycle
speed is 13 km/h everywhere. Walking is available for reachable trips up to
5 km across the full Canton. The GTFS feed is processed for a representative
weekday in the 17:00-18:00 period.

PT skims retain in-vehicle time, access, egress, initial waiting, physical
transfer, transfer waiting, transfer count, selected stops, and mode-specific
distance. PT fares therefore use the distance of the selected pt_walk or
pt_bike path. E-bikes alter standalone bicycle time and the bicycle access and
egress portions of PT, but do not alter waiting, transfer, or in-vehicle time.


INPUT DATA
----------

input_data/raw/
    Public source archives and frozen OSM network snapshots.

input_data/prepared/
    Five runtime inputs:

    zones.parquet
        Zone geometry, municipality and City-quartier names, City/Canton
        level, population, and jobs.

    demand.pkl.gz
        Passenger OD and fixed road-background OD tables.

    skims.pkl.gz
        Direct-mode, PT-component, selected-stop, and distance matrices.

    lookups.pkl.gz
        Zone-to-stop tables used by mobility-hub interventions.

    assignment_network.pkl
        Road network, capacities, and zone-node mapping used by ta_lab.

input_data/work/
    Disposable intermediate files created only by prepare_inputs.py. During a
    rebuild these can include extracted NPVM OMX matrices under work/npvm,
    direct-mode and PT matrices under work/skims, and processed GTFS tables and
    stop lookups under work/gtfs. They are consolidated into prepared packages
    and the directory is cleared after a successful build unless --keep-work
    is supplied. It is therefore normally empty except for README.txt.

The compressed pickle packages are dictionaries of pandas DataFrames. They
consolidate related matrices without changing their values or labels.

The supplied prepared files are sufficient for every Main.py run, including
assignment and both levels of all three interventions. The supplied raw files
are sufficient to regenerate the prepared files with prepare_inputs.py.


REBUILDING INPUTS
-----------------

Check the prepared runtime packages:

       python prepare_inputs.py --check-runtime

Check all raw sources:

       python prepare_inputs.py --check-raw

Rebuild everything from raw data:

       python prepare_inputs.py --all

Individual stages are also available:

       python prepare_inputs.py --zones-demand
       python prepare_inputs.py --networks
       python prepare_inputs.py --gtfs

Package compatible intermediate skim and lookup files already retained in
input_data/work/:

       python prepare_inputs.py --package

The complete rebuild is computationally intensive, particularly GTFS and
full-Canton walking-network processing. Add --keep-work when intermediate
tables are needed for teaching or diagnostics. Without --keep-work, a
successful build removes those intermediates after updating and validating the
five files in input_data/prepared/. Raw source files are never deleted.


OUTPUTS
-------

Each timestamped run contains:

mode_summary.csv
    Four-mode trips, trip shares, passenger-kilometres, distance shares, and
    mean trip distance.

mode_summary_detailed.csv
    The same totals with pt_walk and pt_bike shown separately.

mode_summary_by_area.csv
    Four-mode results for the total model, City origins, and rest-of-Canton
    origins.

mode_summary_by_area_detailed.csv
    Area results with the two PT access alternatives shown separately.

mode_share_by_distance_bin.csv
mode_share_by_distance_bin_detailed.csv
    Mode shares by centroid-distance bin and origin area.

modal_share_pies.png
    Side-by-side mode-share charts for trips and passenger-kilometres.

od_<mode>.parquet
    Final OD trips allocated to each mode-choice alternative.

zone_accessibility.parquet
    Final logsum accessibility by zone.

road_link_flows.parquet
    Assigned road-link flows when assignment is enabled.

intervention_summary.csv
    Four-mode changes within each selected intervention's OD coverage, plus a
    combined row group when several interventions are selected. It reports
    trips, shares, mean time, mean distance, and PT component times against a
    matched case with all selected interventions removed. Scenario demand,
    coefficients, e-bike share, and the intervention run's final road skim are
    held fixed. Mean-time and mean-distance comparisons use the same
    counterfactual mode-trip weights on both sides, so they describe skim
    changes rather than changes in traveller composition. A baseline run
    writes an empty table with the same headers.

run_metadata.json
diagnostics.json
    Effective settings and model diagnostics for reproducibility.
    Assignment diagnostics distinguish low-volume OD pairs omitted by the
    loading threshold from retained OD pairs that the road network cannot
    connect.

outputs/example_output contains a compact real-run example with summaries and
the modal-share figure. Large OD and link-flow files are not duplicated there.
Using --output-dir writes to the exact requested directory and bypasses the
automatic timestamped name.


CODE MAP
--------

Main.py                     Run settings and model sequence
config.py                   Repository paths and core modelling assumptions
config/mode_choice.json     Mode-choice coefficients and monetary assumptions
zoning.py                   Direct NPVM zoning and runtime input loading
demand_preparation.py       Raw NPVM demand aggregation
background_traffic.py       Peripheral and commercial road-background demand
network.py                  Direct networks, skims, and assignment network
routing.py                  Shortest-path helper used in preprocessing
gtfs_analysis.py            GTFS processing and PT component skims
scenario_generation.py      Population, jobs, and trip-demand scenario
mode_choice_zurich.py       Mode utilities and probabilistic mode choice
interventions.py            Three editable interventions with two levels each
travel_times.py             Skim loading and perceived-time policies
trip_assignment_zurich.py   ta_lab road assignment
model_outputs.py            Tables, figures, and output writing
input_packages.py           Prepared-package reading, writing, and alignment
prepare_inputs.py           Raw-to-prepared preprocessing workflow
ta_lab/                     Frank-Wolfe traffic-assignment library
tests/                      Focused automated tests
