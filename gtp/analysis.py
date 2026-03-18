"""Analysis utilities for loading experiment results, aggregating, and plotting."""
from __future__ import annotations

import json
import os

import pandas as pd
import matplotlib.pyplot as plt


def load_experiment_results(results_dir: str) -> pd.DataFrame:
    """Load results.jsonl from a results directory as a DataFrame."""
    return pd.read_json(os.path.join(results_dir, "results.jsonl"), lines=True)


def load_run(results_dir: str, run_id: str) -> dict:
    """Load full result for a specific run."""
    path = os.path.join(results_dir, "runs", f"{run_id}.json")
    with open(path) as f:
        return json.load(f)


def cooperation_rate_table(
    df: pd.DataFrame, group_by: list[str]
) -> pd.DataFrame:
    """Pivot table of mean cooperation rates grouped by conditions."""
    return (
        df.groupby(group_by)
        .agg(
            coop_a_mean=("coop_rate_a", "mean"),
            coop_a_std=("coop_rate_a", "std"),
            coop_b_mean=("coop_rate_b", "mean"),
            coop_b_std=("coop_rate_b", "std"),
            score_a_mean=("score_a", "mean"),
            score_b_mean=("score_b", "mean"),
            n=("run_id", "count"),
        )
        .round(3)
    )


def plot_score_trajectories(results_dir: str, run_ids: list[str]):
    """Plot cumulative score trajectories for selected runs."""
    fig, axes = plt.subplots(1, len(run_ids), figsize=(6 * len(run_ids), 5), squeeze=False)
    for ax, run_id in zip(axes[0], run_ids):
        data = load_run(results_dir, run_id)
        df = pd.DataFrame(data["history"])
        df["cumulative_a"] = df["payoff_a"].cumsum()
        df["cumulative_b"] = df["payoff_b"].cumsum()
        ax.plot(df["round"], df["cumulative_a"], "b-o", label="Agent A", markersize=4)
        ax.plot(df["round"], df["cumulative_b"], "r-o", label="Agent B", markersize=4)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative Score")
        ax.set_title(run_id, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_transparency_effect(df: pd.DataFrame):
    """Bar chart comparing cooperation rates across transparency conditions."""
    # Build transparency label
    df = df.copy()
    df["transparency"] = df.apply(
        lambda r: (
            "both_see"
            if r["transparency_a"] and r["transparency_b"]
            else (
                "both_blind"
                if not r["transparency_a"] and not r["transparency_b"]
                else "a_sees_b" if r["transparency_a"] else "b_sees_a"
            )
        ),
        axis=1,
    )
    grouped = df.groupby("transparency").agg(
        coop_a=("coop_rate_a", "mean"),
        coop_b=("coop_rate_b", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean Cooperation Rate")
    ax.set_title("Cooperation Rate by Transparency Condition")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig
