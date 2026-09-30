# Transport data

| Folder | Contents |
| --- | --- |
| `prepared/` | Zones, passenger and background OD demand, time/distance skims, intervention lookups and the assignment network used by the notebooks. |
| `config/` | Mode-choice coefficients. |
| `routing/` | Saved road, cycling and walking graphs, a PT service graph with station groups and transfer times, and the city boundary. |
| `raw/` | Original NPVM, GTFS, boundary and population archives, plus source network snapshots. |

The notebooks load the prepared inputs and configuration. Derived corridor networks
and surrogate artifacts are in [data/processed](../processed/README.md). The MehrSpur counting coverage is supplied in `data/processed/section_coverage.npz` and its matching JSON. It is regenerated from the selected definition and routing inputs when needed; projects with `coverage_file=None` use `cache/sections/`.
Raw archives are retained as source data; the notebooks do not rebuild the prepared
inputs from these archives.

The transport archives and binary datasets use Git LFS. After cloning, run
`git lfs pull` before loading the transport model.

`routing/pt_routing_input.pkl` supplies fixed PT service connections and transfer components. It contains no physical railway alignments. Students can regenerate counting coverage from it without processing GTFS. The prepared skims and this routing snapshot must be kept together.

A dated MehrSpur OSM extract is retained in `raw/osm/mehrspur_2026-09-29_overpass.json.gz`, with its provenance. Regenerating a different corridor requests its own OSM data.
