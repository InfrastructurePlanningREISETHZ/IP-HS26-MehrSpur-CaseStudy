# Transport core

These modules support the native transport model. Use [transport_model_interface.py](../transport_model_interface.py) to load inputs and run it. Project settings belong in [parameters.py](../parameters.py), [stages.py](../stages.py) and [adaptive_planning.py](../adaptive_planning.py).

| Module | Role |
| --- | --- |
| `config.py` | Input paths and physical defaults. |
| `input_packages.py`, `zoning.py` | Load prepared inputs and align OD labels. |
| `travel_times.py` | Travel-time components, floors and e-bike adjustments. |
| `interventions.py` | Apply physical interventions. |
| `mode_choice_zurich.py` | Allocate demand across car, cycling, walking, PT with walking access and PT with cycling access. |
| `network.py`, `routing.py` | Routing support for preparing counting sections. |

[Transport data](../../data/transport/) contains prepared model inputs, mode-choice configuration, routing graphs and raw source archives. Normal notebook calculations use the prepared inputs. See the [code guide](../README.md) for the calculation workflow.
