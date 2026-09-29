# Transport data

| Folder | Contents |
| --- | --- |
| `prepared/` | Zones, passenger and background OD demand, time/distance skims, intervention lookups and the assignment network used by the notebooks. |
| `config/` | Mode-choice coefficients. |
| `routing/` | Saved road, cycling and walking graphs and the city boundary used for route and network preparation. |
| `raw/` | Original NPVM, GTFS, boundary and population archives, plus source network snapshots. |

The notebooks load the prepared inputs and configuration. Derived corridor networks,
counting-section coverage and surrogate artifacts are in [data/processed](../processed/README.md).
Raw archives are retained as source data; the notebooks do not rebuild the prepared
inputs from these archives.

The transport archives and binary datasets use Git LFS. After cloning, run
`git lfs pull` before loading the transport model.
