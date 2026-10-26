# Emran's Python style

Match this so code reads as Emran's own and drops into their files without reformatting. When editing an existing file, that file's own conventions win over anything here; these notes are for new code and for resolving what the surrounding code does not settle.

## Naming

- **Long, descriptive, unabbreviated.** `randomness_generator`, `known_water_transitions`, `viable_path_threshold`, `predicted_position`, `state_visit_count`. Not `rng`, `env`, `obs`, `col`, `tmp`. A name should say what the thing is in full.
- **Booleans read as questions**, with an `is_` or `has_` prefix: `is_episode_done`, `has_unattempted_actions`, `is_against_conveyor`, `is_goal_state`.
- **Two privacy levels, used deliberately.** A single leading underscore marks a method meant to be overridden by subclasses (a hook): `_default_agent_parameters`, `_build_tables`, `_select_action`. A double leading underscore marks a genuinely internal method: `__calculate_frontier_score`, `__learn_optimal_trajectory`, `__seed_table_from_opening_frame`.
- **Constants in `UPPER_CASE`**: `LEAP`, `DARK_COLORS`, `SERIES_COLORS`.
- **Numeric literals grouped with underscores**: `69_420`, `42_069`.
- **Configuration keys are strings**, accessed with `["key"]` rather than attributes: `agent_parameters["novelty"]`, `seeds["environment_seed"]`.

## Whitespace and layout (the strongest tells)

- **Spaces around `=` for keyword arguments and default values.** This is the clearest signature of Emran's code and is applied everywhere:
  ```python
  agent = GoExploreAgent(environment = environment, available_actions = available_actions, seed = None)

  def __calculate_manhattan_distance(self, new_position: int, initial_position: Optional[int] = None) -> int:
  ```
- **One parameter per line** once a signature has more than about two parameters, indented two levels (eight spaces) past the `def`, with the closing `):` on its own line aligned under the `def`:
  ```python
  def __init__(
          self,
          environment: GridworldEnv,
          available_actions: Dict[int, Tuple[int, int]],
          agent_parameters: Optional[dict] = None,
          seed: Optional[int] = None
  ):
  ```
- **Generous vertical whitespace.** Two (sometimes three) blank lines between methods, a blank line immediately after the signature and after the docstring before the body begins, and blank lines separating logical blocks inside a function.
- **One step per line.** Prefer named intermediates over dense nested expressions: compute `predicted_position`, then `is_novel_state`, then `score`, rather than folding them into one line.
- **Guard clauses and early returns** to handle the exceptional case first (`if not has_unattempted_actions: return None`).

## Imports

- **No aliases.** Import the module or submodule and use its full name:
  ```python
  import numpy

  from matplotlib import pyplot
  from matplotlib import colors
  from matplotlib import patches
  ```
  Then `numpy.mean(...)`, `pyplot.subplots(...)`. Never `import numpy as np` or `import matplotlib.pyplot as plt`.
- **Grouped**: standard library, then third-party, then local project modules, with a blank line between groups.
- `from typing import Dict, List, Optional, Tuple` for the hints.
- `from config import *` appears only because a provided module uses that pattern; do not introduce new wildcard imports of your own.

## Type hints

- Annotate signatures, including return types: `-> Tuple[int, int]`, `-> bool`, `-> List`, `-> float`. Draw the generics from `typing`.
- Local variables are generally left unannotated; the signature carries the contract.

## Structure and configuration

- **Configuration flows through dictionaries** — `agent_parameters`, `simulation_parameters`, `environment_parameters`, `seeds` — passed in and merged over a defaults dictionary:
  ```python
  def __initialize_variables(self):

      self.agent_parameters = {
          "learning_rate": 0.1,
          "discount": 0.99
      }

  def update_agent_parameters(self, agent_parameters: dict):

      if agent_parameters:
          self.agent_parameters.update(agent_parameters)
  ```
  Anything a caller might reasonably want to change is exposed this way rather than hard-coded. Emran has asked for this explicitly: configurable, not rigid.
- **A clear class lifecycle.** `__init__` stores the arguments, calls a private `__initialize_variables()` to set up defaults and empty state, then applies the overrides through `update_*_parameters()`.
- **Base class plus hook subclasses.** A base (for example a `TabularAgent`) defines the shared machinery and small overridable hooks (`_default_agent_parameters`, `_build_tables`); each concrete class (`QLearningAgent`, `DynaQAgent`) overrides only what differs.
- **Reproducibility is built in.** Seed everything explicitly: a local `Random(self.seed)` rather than the global module, seed dictionaries, and deterministic increments (`seed + seed_increment_count * index`). Never depend on implicit global RNG state.

## A canonical shape

```python
from random import Random
from typing import Dict, Optional, Tuple


class ExampleAgent:

    def __init__(
            self,
            environment,
            available_actions: Dict[int, Tuple[int, int]],
            agent_parameters: Optional[dict] = None,
            seed: Optional[int] = None
    ):
        """
        Initialise the agent for a given environment.

        Parameters
        ----------
        environment : object
            The environment to act in.
        available_actions : dict of int to tuple of int
            Mapping from each action to its ``(row, column)`` offset.
        agent_parameters : dict, optional
            Overrides for the default weights.
        seed : int, optional
            Seed for the agent's own randomness.
        """

        self.environment = environment
        self.available_actions = available_actions
        self.seed = seed

        self.__initialize_variables()
        self.update_agent_parameters(agent_parameters)


    def __initialize_variables(self):

        self.agent_parameters = {
            "learning_rate": 0.1,
            "discount": 0.99
        }
        self.randomness_generator = Random(self.seed)


    def update_agent_parameters(self, agent_parameters: dict):

        if agent_parameters:
            self.agent_parameters.update(agent_parameters)
```

## Notebooks

- Configuration dictionaries at the top, then function definitions, then the execution loops.
- `try/except ImportError` around optional dependencies (for example Optuna), falling back gracefully.
- f-strings for logging: `print(f"Environment: {run_index + 1} | Final reward: {rewards[-1]}")`.

## Spelling

British spelling runs through docstrings and comments: behaviour, colour, initialise, optimise, neighbour. Code identifiers follow their library, so matplotlib's `color` argument stays `color` even though the surrounding prose says "colour".
