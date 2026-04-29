#!/usr/bin/env python3
"""
Analyze an experiment results directory.

Default: generate a full report (markdown + plots + CSVs) in the results dir.
Flags override to produce subset outputs.

Usage:
    python scripts/analyze.py results/endgame_effect_20260413_203824
    python scripts/analyze.py <results_dir> --plot cooperation
    python scripts/analyze.py <results_dir> --plot heatmap
    python scripts/analyze.py <results_dir> --plot trajectories
    python scripts/analyze.py <results_dir> --reasoning-mismatches
    python scripts/analyze.py <results_dir> --endgame-refs
    python scripts/analyze.py <results_dir> --summary
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtp.analysis import (
    load_per_round_df,
    load_per_reasoning_df,
    plot_cooperation_by_round,
    plot_move_heatmap,
    plot_score_trajectories_from_df,
    find_reasoning_action_mismatches,
    count_endgame_references,
    summarize_experiment,
)


def _plots_dir(results_dir: str) -> str:
    path = os.path.join(results_dir, "plots")
    os.makedirs(path, exist_ok=True)
    return path


def do_cooperation(results_dir, df_rounds, group_by="condition_id"):
    fig = plot_cooperation_by_round(df_rounds, group_by=group_by)
    out = os.path.join(_plots_dir(results_dir), "cooperation_by_round.png")
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def do_heatmap(results_dir, df_rounds):
    fig = plot_move_heatmap(df_rounds)
    out = os.path.join(_plots_dir(results_dir), "move_heatmap.png")
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def do_trajectories(results_dir, df_rounds):
    fig = plot_score_trajectories_from_df(df_rounds)
    out = os.path.join(_plots_dir(results_dir), "score_trajectories.png")
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def do_mismatches(results_dir, df_reasoning, df_rounds):
    df = find_reasoning_action_mismatches(df_reasoning, df_rounds)
    out = os.path.join(results_dir, "reasoning_mismatches.csv")
    df.to_csv(out, index=False)
    print(f"  wrote {out} ({len(df)} candidates)")


def do_endgame_refs(results_dir, df_reasoning):
    df = count_endgame_references(df_reasoning)
    out = os.path.join(results_dir, "endgame_refs.csv")
    df.to_csv(out, index=False)
    print(f"  wrote {out} ({len(df)} entries)")


def do_summary(results_dir):
    text = summarize_experiment(results_dir)
    out = os.path.join(results_dir, "report.md")
    with open(out, "w") as f:
        f.write(text)
    print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("results_dir", help="Path to experiment results directory")
    parser.add_argument(
        "--plot",
        choices=["cooperation", "heatmap", "trajectories"],
        action="append",
        help="Generate a specific plot (can pass multiple times)",
    )
    parser.add_argument(
        "--group-by",
        default="condition_id",
        help="Column to group cooperation plot by (default: condition_id)",
    )
    parser.add_argument(
        "--reasoning-mismatches",
        action="store_true",
        help="Find reasoning/action mismatches",
    )
    parser.add_argument(
        "--endgame-refs",
        action="store_true",
        help="Count endgame references in reasoning",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Write markdown report",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"error: not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Detect whether any flag was passed. If none, run the full default report.
    flags_passed = any([
        args.plot,
        args.reasoning_mismatches,
        args.endgame_refs,
        args.summary,
    ])

    print(f"Loading data from {results_dir}...")
    df_rounds = load_per_round_df(results_dir)
    print(f"  {len(df_rounds)} per-round rows across {df_rounds['run_id'].nunique() if not df_rounds.empty else 0} runs")

    if flags_passed:
        # Flag-based mode: run only what was requested
        if args.plot:
            for p in args.plot:
                if p == "cooperation":
                    do_cooperation(results_dir, df_rounds, group_by=args.group_by)
                elif p == "heatmap":
                    do_heatmap(results_dir, df_rounds)
                elif p == "trajectories":
                    do_trajectories(results_dir, df_rounds)
        if args.reasoning_mismatches:
            df_reasoning = load_per_reasoning_df(results_dir)
            do_mismatches(results_dir, df_reasoning, df_rounds)
        if args.endgame_refs:
            df_reasoning = load_per_reasoning_df(results_dir)
            do_endgame_refs(results_dir, df_reasoning)
        if args.summary:
            do_summary(results_dir)
    else:
        # Default: full report. Reasoning/action mismatch is opt-in via
        # --reasoning-mismatches because the heuristic is noisy.
        print("Generating full report...")
        do_cooperation(results_dir, df_rounds, group_by=args.group_by)
        do_heatmap(results_dir, df_rounds)
        do_trajectories(results_dir, df_rounds)
        df_reasoning = load_per_reasoning_df(results_dir)
        print(f"  {len(df_reasoning)} reasoning entries loaded")
        do_endgame_refs(results_dir, df_reasoning)
        do_summary(results_dir)

    print("Done.")


if __name__ == "__main__":
    main()
