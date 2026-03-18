# Game-Theoretic Prompting

Experiment framework for studying how LLM agents behave in iterated game-theoretic scenarios. Tests how prompt framing (cooperative, competitive, etc.) and information asymmetry (prompt transparency) affect agent strategies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires a local [LM Studio](https://lmstudio.ai/) server on `localhost:1234` (default, zero cost) or an OpenAI API key.

## Quick Start

### CLI (experiment runner)

```bash
# Preview what will run
python scripts/run_experiment.py experiments/pilot.yaml --dry-run

# Run the experiment
python scripts/run_experiment.py experiments/pilot.yaml

# Full sweep: 16 conditions x 3 reps = 48 runs
python scripts/run_experiment.py experiments/transparency_sweep.yaml
```

### Notebook

Open `prisoners_dilemma.ipynb` for interactive single-game exploration with plots and reasoning traces.

## Experiment Configs

Experiments are defined in YAML. Two modes:

**Explicit runs:**
```yaml
name: quick_test
repetitions: 1
defaults:
  game: prisoners_dilemma
  num_rounds: 5
runs:
  - id: coop_vs_comp
    agent_a: {prompt: cooperative, can_see_opponent: false}
    agent_b: {prompt: competitive, can_see_opponent: true}
```

**Grid mode** (Cartesian product):
```yaml
name: sweep
repetitions: 3
grid:
  prompt_a: [cooperative, competitive]
  prompt_b: [cooperative, competitive]
  transparency:
    - {a_sees_b: false, b_sees_a: false}
    - {a_sees_b: true,  b_sees_a: true}
```

Available prompts: `cooperative`, `competitive`, `neutral`, `tit_for_tat`, `deceptive` (see `gtp/config.py`).

## Adding a New Game

1. Create `gtp/games/your_game.py`:

```python
from gtp.games.base import BaseGame

class YourGame(BaseGame):
    @property
    def name(self): return "your_game"
    def get_moves(self): return ["MOVE_A", "MOVE_B"]
    def resolve(self, a, b): return {("MOVE_A","MOVE_A"): (3,3), ...}[(a,b)]
    def get_payoff_description(self): return "  MOVE_A vs MOVE_A: each gets 3\n  ..."
```

2. Register in `gtp/games/__init__.py`:
```python
GAME_REGISTRY["your_game"] = YourGame
```

3. Use `game: your_game` in experiment YAML.

Built-in games: `prisoners_dilemma`, `chicken`, `stag_hunt`.

## Results

Results are saved to `results/` (gitignored):
- `results.jsonl` — one summary line per run (scores, cooperation rates)
- `runs/<run_id>.json` — full history + reasoning traces

Load in Python:
```python
from gtp.analysis import load_experiment_results, cooperation_rate_table
df = load_experiment_results("results/pilot_20260317_133440")
cooperation_rate_table(df, group_by=["transparency_a", "transparency_b"])
```

## Project Structure

```
gtp/                    Core package
  config.py             Dataclasses, prompt registry, YAML loader
  graph.py              LangGraph state machine
  runner.py             Experiment runner (grid expansion, resumption)
  results.py            Result storage (JSONL + JSON)
  analysis.py           Aggregation and plotting
  games/
    base.py             BaseGame ABC
    prisoners_dilemma.py
    chicken.py
    stag_hunt.py
  agents/
    tools.py            Agent tool factory (history, scores, opponent prompt)
    builder.py          ReAct agent construction
experiments/            YAML experiment definitions
scripts/
  run_experiment.py     CLI entry point
```
