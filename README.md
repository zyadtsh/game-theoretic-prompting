# Game-Theoretic Prompting

Experiment framework for studying how LLM agents behave in iterated game-theoretic scenarios. Tests how prompt framing (cooperative, competitive, etc.) and information asymmetry (prompt transparency) affect agent strategies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires a local [LM Studio](https://lmstudio.ai/) server on `localhost:1234` (default, zero cost) or a cloud provider like [OpenRouter](https://openrouter.ai/).

**For cloud inference (OpenRouter):**
```bash
cp .env.example .env
# Edit .env and add your API key
```

Then reference it in your experiment YAML:
```yaml
defaults:
  model:
    model: meta-llama/llama-3.1-8b-instruct
    base_url: https://openrouter.ai/api/v1
    api_key: $OPENROUTER_API_KEY
```

API keys prefixed with `$` are resolved from environment variables at runtime. The `.env` file is gitignored.

**LangSmith tracing (optional).** If `.env` contains `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...`, and `LANGSMITH_PROJECT=...`, every LLM call and ReAct step is automatically traced to the named project on [smith.langchain.com](https://smith.langchain.com). No code changes required — `scripts/run_experiment.py` calls `load_dotenv()` before importing LangChain, so the env vars are picked up by the global tracer.

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

**LLM prompts:** `cooperative`, `competitive`, `neutral`, `tit_for_tat`, `deceptive`, `minimal`, plus endgame-aware variants and the 14 Axelrod 1980 strategy twins (`tft_llm`, `grudger_llm`, `joss_llm`, etc. — see `gtp/config.py` and the "Axelrod 1980" section below).

## Fixed-Strategy Agents

Deterministic agents that bypass the LLM entirely — useful as baselines and for fast experiments with no API costs.

| YAML key | Behavior |
|---|---|
| `always_cooperate_fixed` | Always plays the first move (COOPERATE / SWERVE / STAG) |
| `always_defect_fixed` | Always plays the second move (DEFECT / STRAIGHT / HARE) |
| `tit_for_tat_fixed` | First move cooperates, then mirrors opponent's last move |
| `suspicious_tit_for_tat_fixed` | First move defects, then mirrors opponent's last move |

The **14 Axelrod 1980 first-tournament strategies** are also available as fixed agents (wrappers around the `axelrod` library): `tft_fixed`, `tideman_chieruzzi_fixed`, `nydegger_fixed`, `shubik_fixed`, `grudger_fixed`, `davis_fixed`, `downing_fixed`, `random_fixed`, `grofman_fixed`, `joss_fixed`, `feld_fixed`, `tullock_fixed`, `graaskamp_fixed`, `stein_rapoport_fixed`. Each has a matching `*_llm` prompt. See the "Axelrod 1980 strategies vs minimal-LLM" section below.

Use them in YAML exactly like prompt keys:
```yaml
runs:
  - id: tft_vs_competitive_llm
    agent_a:
      prompt: tit_for_tat_fixed
    agent_b:
      prompt: competitive
```

Fixed agents work in grid mode too:
```yaml
grid:
  prompt_a: [always_cooperate_fixed, tit_for_tat_fixed]
  prompt_b: [cooperative, competitive]
```

## Axelrod 1980 strategies vs minimal-LLM

A controlled experiment testing **strategy-execution fidelity**: when an LLM is told the natural-language description of a named Axelrod 1980 strategy, does it actually execute that strategy? Each of 14 first-tournament strategies has a `*_fixed` ground-truth (wrapping the `axelrod` library) and a `*_llm` twin (a system prompt describing the strategy). All 28 agents are played against the same uncoached `minimal` LLM opponent.

**Single-model scope.** This experiment is run on exactly one model — the model named in each experiment YAML's `defaults.model.model` — at `temperature: 0`. The `minimal`-LLM's strategic personality is part of the experimental apparatus, not a nuisance variable, so results are reported as "strategy X executed by model M against minimal-M". Rerunning on a new model is a single-line change (the `model` field) in each of the three YAMLs; comparing models across the fidelity matrix is on the roadmap.

### The 14 families

Deterministic (7, each run once — no variance at T=0):
| YAML key (fixed) | YAML key (LLM twin) | Backing class |
|---|---|---|
| `tft_fixed` | `tft_llm` | `axelrod.TitForTat` |
| `tideman_chieruzzi_fixed` | `tideman_chieruzzi_llm` | `axelrod.FirstByTidemanAndChieruzzi` |
| `nydegger_fixed` | `nydegger_llm` | `axelrod.FirstByNydegger` |
| `shubik_fixed` | `shubik_llm` | `axelrod.FirstByShubik` |
| `grudger_fixed` | `grudger_llm` | `axelrod.Grudger` |
| `davis_fixed` | `davis_llm` | `axelrod.FirstByDavis` |
| `downing_fixed` | `downing_llm` | `axelrod.FirstByDowning` |

Stochastic (7, each run 5 times — average over RNG):
| YAML key (fixed) | YAML key (LLM twin) | Backing class |
|---|---|---|
| `random_fixed` | `random_llm` | `axelrod.Random` |
| `grofman_fixed` | `grofman_llm` | `axelrod.FirstByGrofman` |
| `joss_fixed` | `joss_llm` | `axelrod.FirstByJoss` |
| `feld_fixed` | `feld_llm` | `axelrod.FirstByFeld` |
| `tullock_fixed` | `tullock_llm` | `axelrod.FirstByTullock` |
| `graaskamp_fixed` | `graaskamp_llm` | `axelrod.FirstByGraaskamp` |
| `stein_rapoport_fixed` | `stein_rapoport_llm` | `axelrod.FirstBySteinAndRapoport` |

Stochastic fixed wrappers are seeded per-run from `hash(run_id, agent_name)` so reps produce reproducible but distinct trajectories.

### Running the experiment

The experiment is run in two stages, gated on a stage 0 strategic-competence probe:

```bash
# Stage 0: 4 runs, ~$0.10, must pass before stage 1
python scripts/run_experiment.py experiments/axelrod_diagnostic.yaml

# Stage 1a: 14 runs, deterministic half
python scripts/run_experiment.py experiments/axelrod_deterministic.yaml

# Stage 1b: 70 runs (14 conditions x 5 reps), stochastic half
python scripts/run_experiment.py experiments/axelrod_stochastic.yaml
```

Total ~88 runs on `minimax/minimax-m2.5:free`, roughly $1.

**Resuming after failures.** OpenRouter occasionally returns malformed responses (rate limits, upstream provider blips). The runner catches per-run errors, logs them to `errors.jsonl`, and continues. To finish only the missing/errored runs without re-executing the completed ones, point `--resume` at the existing results directory:

```bash
python scripts/run_experiment.py experiments/axelrod_deterministic.yaml \
  --resume results/axelrod_deterministic_20260423_053833
```

Runs whose `runs/<id>.json` already exists are skipped; everything else re-executes, picking up the new retry behavior in `gtp/agents/builder.py` (transient JSON-decode and 5xx errors retry with exponential backoff up to 8 times).

**Stage 0 gating criteria** (see `gtp.analysis.diagnostic_report`):
- `minimal` cooperates >70% against `always_cooperate_fixed`.
- `minimal` defects on >50% of rounds in the first 20 against `always_defect_fixed`.
- Reactive behavior against `tft_fixed` and `suspicious_tit_for_tat_fixed` (not stuck in a loop).

If a model fails, try `temperature: 0.1`; if it still fails, the model is unsuited for the experiment — document and stop.

### Headline output: the fidelity analysis

`gtp.analysis.fidelity_analysis` is the main deliverable. For each Axelrod family, it reports:

1. **`replay_agreement`** (headline) — round-by-round move agreement between fixed-X and LLM-X. Computed by replaying the LLM twin offline against the recorded opponent (`minimal`) trajectory from the fixed-twin match. This isolates strategy execution from opponent drift.
2. **`first_divergence_online`** — the earliest round at which LLM-X and fixed-X diverged in the actual online matches. After that round the two agents face different opponents, so this is the frontier of "LLM ran the strategy faithfully".
3. **`aggregate_coop_gap`** — the absolute difference in overall cooperation rate between fixed-X-vs-minimal and LLM-X-vs-minimal.

```python
from gtp.analysis import (
    load_experiment_results, standings_table,
    fidelity_analysis, diagnostic_report,
    cooperation_vs_minimal_plot,
)

diag = diagnostic_report("results/axelrod_diagnostic_<timestamp>", model_label="minimax-m2.5")
print(diag.render())

df = load_experiment_results("results/axelrod_deterministic_<timestamp>")
print(standings_table(df))

model_cfg = {
    "model": "minimax/minimax-m2.5:free",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "<your key>",
    "temperature": 0.0,
}
print(fidelity_analysis("results/axelrod_deterministic_<timestamp>", model_cfg))
fig = cooperation_vs_minimal_plot(df)
fig.savefig("cooperation_vs_minimal.png")
```

### Compact history format (research variable)

LLM agents (including `minimal`) see history as a compact summary rather than a per-round dump: move sequences, cooperation rates, and the last 10 rounds. This is on by default via `BaseGame.verbose_history = False`; set it to `True` to reproduce the pre-Axelrod experiments' prompts. Documented as a known research variable — at 100+ rounds the verbose format blows up prompt length nonlinearly and starts to dominate any strategy-framing signal.

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

### CLI report

`scripts/analyze.py` writes a full report into the results directory: cooperation-by-round plot, move heatmap, score trajectories, reasoning/action mismatch CSV, endgame-reference CSV, and a markdown summary.

```bash
# Full report (default)
python scripts/analyze.py results/axelrod_deterministic_20260423_053833

# Or pick subsets
python scripts/analyze.py <results_dir> --plot cooperation --plot heatmap
python scripts/analyze.py <results_dir> --reasoning-mismatches
python scripts/analyze.py <results_dir> --summary
```

### Python API

```python
from gtp.analysis import load_experiment_results, cooperation_vs_minimal_plot
df = load_experiment_results("results/axelrod_deterministic_20260423_053833")
fig = cooperation_vs_minimal_plot(df)
fig.savefig("cooperation_vs_minimal.png")
```

For the older transparency/endgame experiments, `first_move_rate_table(df, group_by=[...])` produces a pivot of cooperation rates across whatever conditions you grouped by.

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
    fixed.py            Fixed-strategy (non-LLM) agents
experiments/            YAML experiment definitions
scripts/
  run_experiment.py     CLI entry point for running experiments
  analyze.py            CLI report generator (plots, CSVs, markdown summary)
tests/                  Test suite (axelrod wrapper, compact history, replay harness)
```
