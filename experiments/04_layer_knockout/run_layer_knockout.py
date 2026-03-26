"""
Experiment 04 — Layer Knockout

Skip each ResBlock one at a time (replace with identity) and measure:
  1. Win rate against a random opponent (with Wilson CI)
  2. Policy KL divergence vs the full model (much more sensitive)

Usage:
    python run_layer_knockout.py
    python run_layer_knockout.py --num-games 500 --max-turns 400
    python run_layer_knockout.py --no-stratify
"""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.common import (
    advance_stuck_turn,
    load_checkpoint_model,
    collect_states_stratified,
    PHASE_LABELS,
)
import td_ludo_cpp as ludo_cpp


def parse_args():
    parser = argparse.ArgumentParser(description="Layer knockout experiment.")
    parser.add_argument(
        "--weights",
        default="../../weights/model_latest_323k_shaped.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument("--num-games", type=int, default=500, help="Games per evaluation.")
    parser.add_argument("--max-turns", type=int, default=500, help="Max turns per game.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified state collection for KL divergence (use uniform random instead).",
    )
    parser.add_argument(
        "--kl-states", type=int, default=200,
        help="Number of states per phase for KL divergence measurement.",
    )
    return parser.parse_args()


def resolve_weights_path(raw_path):
    path = Path(raw_path)
    if path.exists():
        return str(path)
    return str((Path(__file__).resolve().parent / raw_path).resolve())


# ── Statistical helpers ──────────────────────────────────────────────────────


def wilson_ci(wins, total, z=1.96):
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = wins / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return center, center - margin, center + margin


# ── Identity block ───────────────────────────────────────────────────────────


class IdentityBlock(nn.Module):
    """Replaces a ResBlock — just passes input through."""

    def forward(self, x):
        return x


# ── Win-rate evaluation ──────────────────────────────────────────────────────


def evaluate_win_rate(model, num_games, max_turns, seed=0):
    """
    Play `num_games` 2-player games: player 0 uses the model, player 1 plays
    randomly.

    Returns dict with:
      - win_rate (percentage, over completed games only)
      - wins, completed, total
      - completion_rate
      - wilson_center, wilson_lo, wilson_hi  (95% CI)
    """
    rng = np.random.default_rng(seed)
    env = ludo_cpp.VectorGameState(batch_size=num_games, two_player_mode=True)

    wins = 0
    completed = 0

    for turn in range(max_turns):
        # Check which games are still active
        active_games = []
        for i in range(num_games):
            game = env.get_game(i)
            if game.is_terminal:
                continue
            active_games.append(i)

        if not active_games:
            break

        # Roll dice for games that need it
        for i in active_games:
            game = env.get_game(i)
            if game.current_dice_roll == 0:
                game.current_dice_roll = int(rng.integers(1, 7))

        legal_moves_batch = env.get_legal_moves()
        states_tensor = env.get_state_tensor()

        # Batch model inference for all player-0 games
        model_games = []
        for i in active_games:
            game = env.get_game(i)
            if game.current_player == 0 and legal_moves_batch[i]:
                model_games.append(i)

        model_actions = {}
        if model_games:
            batch_tensors = torch.tensor(
                np.array([states_tensor[i] for i in model_games]),
                dtype=torch.float32,
            )
            batch_masks = torch.zeros(len(model_games), 4, dtype=torch.float32)
            for j, i in enumerate(model_games):
                for m in legal_moves_batch[i]:
                    batch_masks[j, m] = 1.0

            with torch.no_grad():
                policy, _ = model(batch_tensors, batch_masks)

            for j, i in enumerate(model_games):
                model_actions[i] = int(policy[j].argmax().item())

        # Build action list
        actions = []
        for i in range(num_games):
            game = env.get_game(i)
            moves = legal_moves_batch[i]

            if game.is_terminal:
                actions.append(-1)
                continue

            if not moves:
                advance_stuck_turn(game)
                actions.append(-1)
                continue

            if i in model_actions:
                actions.append(model_actions[i])
            elif game.current_player == 0:
                actions.append(int(rng.choice(moves)))
            else:
                # Random opponent
                actions.append(int(rng.choice(moves)))

        _, _, _, infos = env.step(actions)

        for i, info in enumerate(infos):
            if info["is_terminal"]:
                completed += 1
                game = env.get_game(i)
                scores = np.array(game.scores, dtype=np.int8)
                if scores[0] > scores[1]:
                    wins += 1
                env.reset_game(i)

    # Count unfinished games — do NOT count them toward win rate
    unfinished = 0
    for i in range(num_games):
        game = env.get_game(i)
        if not game.is_terminal:
            unfinished += 1

    total = completed + unfinished
    completion_rate = completed / max(total, 1) * 100.0
    wr = wins / max(completed, 1) * 100.0

    w_center, w_lo, w_hi = wilson_ci(wins, completed)

    return {
        "win_rate": wr,
        "wins": wins,
        "completed": completed,
        "total": total,
        "unfinished": unfinished,
        "completion_rate": completion_rate,
        "wilson_center": w_center * 100.0,
        "wilson_lo": w_lo * 100.0,
        "wilson_hi": w_hi * 100.0,
    }


# ── Policy KL divergence ────────────────────────────────────────────────────


def _get_policy_logits(model, samples, batch_size=256):
    """Run model on a list of samples and return raw log-softmax policy outputs."""
    all_log_probs = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        tensors = torch.tensor(
            np.array([s["tensor"] for s in batch]), dtype=torch.float32
        )
        masks = torch.tensor(
            np.array([s["legal_mask"] for s in batch]), dtype=torch.float32
        )
        with torch.no_grad():
            policy, _ = model(tensors, masks)
        # policy is already softmax output from the model; clamp for log stability
        log_probs = torch.log(policy.clamp(min=1e-8))
        all_log_probs.append(log_probs)
    return torch.cat(all_log_probs, dim=0)


def measure_policy_divergence(model, phase_bins, flat_samples):
    """
    Compute policy KL divergence between the full model and the knocked-out
    model for each ResBlock.

    The *full* model's outputs are computed once; then for each knockout we
    compute KL(full || knocked_out) over the held-out states.

    Returns:
        per_block: list of dicts with kl_mean, kl_std, per_phase KL
    """
    num_blocks = len(model.res_blocks)

    # 1. Collect baseline (full model) policy
    print("Computing baseline policy for KL divergence...")
    baseline_log_probs = _get_policy_logits(model, flat_samples)
    baseline_probs = torch.exp(baseline_log_probs)

    # Also compute per-phase baseline
    phase_baseline = {}
    phase_indices = {}
    idx = 0
    for phase in PHASE_LABELS:
        n = len(phase_bins.get(phase, []))
        phase_indices[phase] = (idx, idx + n)
        idx += n
    # flat_samples may be shuffled; instead compute per-phase from phase_bins directly
    phase_log_probs_baseline = {}
    for phase in PHASE_LABELS:
        if phase_bins.get(phase):
            phase_log_probs_baseline[phase] = _get_policy_logits(model, phase_bins[phase])

    # 2. For each block knockout, measure KL
    per_block = []
    for i in range(num_blocks):
        original_block = model.res_blocks[i]
        model.res_blocks[i] = IdentityBlock()

        # Global KL
        ko_log_probs = _get_policy_logits(model, flat_samples)
        # KL(baseline || knockout) = sum baseline * (log baseline - log knockout)
        kl_per_sample = F.kl_div(ko_log_probs, baseline_probs, reduction="none", log_target=False).sum(dim=-1)
        kl_mean = kl_per_sample.mean().item()
        kl_std = kl_per_sample.std().item()

        # Per-phase KL
        phase_kl = {}
        for phase in PHASE_LABELS:
            if not phase_bins.get(phase):
                phase_kl[phase] = {"mean": 0.0, "std": 0.0}
                continue
            ko_phase_log = _get_policy_logits(model, phase_bins[phase])
            bl_phase_probs = torch.exp(phase_log_probs_baseline[phase])
            kl_phase = F.kl_div(ko_phase_log, bl_phase_probs, reduction="none", log_target=False).sum(dim=-1)
            phase_kl[phase] = {
                "mean": kl_phase.mean().item(),
                "std": kl_phase.std().item(),
            }

        per_block.append({
            "block": i,
            "kl_mean": kl_mean,
            "kl_std": kl_std,
            "phase_kl": phase_kl,
        })

        print(f"  Block {i} KL: {kl_mean:.4f} +/- {kl_std:.4f}  "
              f"[early={phase_kl['early']['mean']:.4f}, "
              f"mid={phase_kl['mid']['mean']:.4f}, "
              f"late={phase_kl['late']['mean']:.4f}]")

        # Restore
        model.res_blocks[i] = original_block

    return per_block


# ── Main knockout driver ─────────────────────────────────────────────────────


def run_layer_knockout(model, num_games, max_turns, seed):
    """Run knockout for each ResBlock and return win-rate results."""
    num_blocks = len(model.res_blocks)

    # Baseline: full model
    print("Evaluating baseline (all blocks active)...")
    baseline = evaluate_win_rate(model, num_games, max_turns, seed)
    print(f"  Baseline WR: {baseline['win_rate']:.1f}% "
          f"[{baseline['wilson_lo']:.1f}%, {baseline['wilson_hi']:.1f}%] "
          f"({baseline['completed']}/{baseline['total']} completed)")

    results = {
        "baseline": baseline,
        "knockouts": [],
    }

    for i in range(num_blocks):
        original_block = model.res_blocks[i]
        model.res_blocks[i] = IdentityBlock()

        print(f"Evaluating with block {i} knocked out...")
        wr_result = evaluate_win_rate(model, num_games, max_turns, seed)
        delta = wr_result["win_rate"] - baseline["win_rate"]
        print(f"  Block {i} skipped: WR={wr_result['win_rate']:.1f}% "
              f"[{wr_result['wilson_lo']:.1f}%, {wr_result['wilson_hi']:.1f}%] "
              f"(delta={delta:+.1f}%), "
              f"{wr_result['completed']}/{wr_result['total']} completed")

        entry = {
            "block": i,
            "delta": delta,
            **wr_result,
        }
        results["knockouts"].append(entry)

        # Restore
        model.res_blocks[i] = original_block

    return results


# ── Visualization ────────────────────────────────────────────────────────────


def visualize(results, save_path):
    """Generate bar charts of WR deltas and KL divergence per block."""
    knockouts = results["knockouts"]
    blocks = [k["block"] for k in knockouts]
    deltas = [k["delta"] for k in knockouts]
    win_rates = [k["win_rate"] for k in knockouts]
    baseline_wr = results["baseline"]["win_rate"]

    has_kl = "kl_divergence" in results

    nrows = 3 if has_kl else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 5 * nrows))

    # ── Plot 1: Win rate per knockout with CIs ──
    ax1 = axes[0]
    colors = ["#e74c3c" if d < -5 else "#f39c12" if d < -2 else "#2ecc71" for d in deltas]

    ci_lo = [k["wilson_lo"] for k in knockouts]
    ci_hi = [k["wilson_hi"] for k in knockouts]
    yerr_lo = [wr - lo for wr, lo in zip(win_rates, ci_lo)]
    yerr_hi = [hi - wr for wr, hi in zip(win_rates, ci_hi)]

    ax1.bar(blocks, win_rates, color=colors, edgecolor="black", linewidth=0.5,
            yerr=[yerr_lo, yerr_hi], capsize=4, error_kw={"linewidth": 1.2})
    ax1.axhline(y=baseline_wr, color="blue", linestyle="--", linewidth=2,
                label=f"Baseline: {baseline_wr:.1f}%")
    ax1.set_xlabel("ResBlock Index")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Layer Knockout — Win Rate with Each Block Removed (95% Wilson CI)")
    ax1.set_xticks(blocks)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # ── Plot 2: Delta from baseline ──
    ax2 = axes[1]
    colors2 = ["#e74c3c" if d < -5 else "#f39c12" if d < -2 else "#2ecc71" for d in deltas]
    ax2.bar(blocks, deltas, color=colors2, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("ResBlock Index")
    ax2.set_ylabel("WR Delta (%)")
    ax2.set_title("Layer Knockout — Win Rate Change from Baseline")
    ax2.set_xticks(blocks)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    for block, delta in zip(blocks, deltas):
        label = "CRITICAL" if delta < -5 else "important" if delta < -2 else "redundant"
        ax2.annotate(
            label,
            (block, delta),
            textcoords="offset points",
            xytext=(0, -15 if delta >= 0 else 10),
            ha="center",
            fontsize=8,
            color="gray",
        )

    # ── Plot 3: KL divergence (if available) ──
    if has_kl:
        ax3 = axes[2]
        kl_data = results["kl_divergence"]
        kl_means = [k["kl_mean"] for k in kl_data]
        kl_stds = [k["kl_std"] for k in kl_data]

        bar_width = 0.2
        x = np.arange(len(blocks))

        # Global KL bar
        ax3.bar(x, kl_means, width=bar_width * 3, color="#3498db", edgecolor="black",
                linewidth=0.5, yerr=kl_stds, capsize=3, alpha=0.4, label="Global KL")

        # Per-phase KL bars
        phase_colors = {"early": "#e74c3c", "mid": "#f39c12", "late": "#2ecc71"}
        for j, phase in enumerate(PHASE_LABELS):
            phase_means = [k["phase_kl"][phase]["mean"] for k in kl_data]
            offset = (j - 1) * bar_width
            ax3.bar(x + offset, phase_means, width=bar_width, color=phase_colors[phase],
                    edgecolor="black", linewidth=0.3, alpha=0.85, label=f"{phase}")

        ax3.set_xlabel("ResBlock Index")
        ax3.set_ylabel("KL Divergence (nats)")
        ax3.set_title("Layer Knockout — Policy KL Divergence vs Full Model (by Phase)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(blocks)
        ax3.legend()
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    np.random.seed(args.seed)

    weights_path = resolve_weights_path(args.weights)
    model = load_checkpoint_model(weights_path)

    # 1. Win-rate knockout
    results = run_layer_knockout(model, args.num_games, args.max_turns, args.seed)

    # 2. Policy KL divergence
    print("\n--- Policy KL Divergence ---")
    if args.no_stratify:
        print("Collecting states WITHOUT stratification (--no-stratify)...")
        # Collect flat random states via stratified with high targets so it
        # fills as much as it can, then use the flat list.
        phase_bins, flat_samples = collect_states_stratified(
            num_games=100, per_phase_target=args.kl_states,
            max_loops=5000, seed=args.seed + 1,
        )
        # Merge all into one bin and clear phase bins for reporting
        phase_bins = {p: [] for p in PHASE_LABELS}
        phase_bins["early"] = flat_samples  # dump all into one bucket
    else:
        print("Collecting stratified states for KL divergence...")
        phase_bins, flat_samples = collect_states_stratified(
            num_games=100, per_phase_target=args.kl_states,
            max_loops=5000, seed=args.seed + 1,
        )

    kl_results = measure_policy_divergence(model, phase_bins, flat_samples)
    results["kl_divergence"] = kl_results

    # Save metrics
    output_dir = Path(__file__).resolve().parent
    save_path = output_dir / "layer_knockout_metrics.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {save_path}")

    # Visualize
    visualize(results, output_dir / "layer_knockout_results.png")

    # Print summary
    print("\n=== LAYER KNOCKOUT SUMMARY ===")
    baseline = results["baseline"]
    print(f"Baseline WR: {baseline['win_rate']:.1f}% "
          f"[{baseline['wilson_lo']:.1f}%, {baseline['wilson_hi']:.1f}%] "
          f"(completion: {baseline['completion_rate']:.0f}%)")
    print()

    sorted_knockouts = sorted(results["knockouts"], key=lambda k: k["delta"])
    for k in sorted_knockouts:
        marker = "***" if k["delta"] < -5 else "**" if k["delta"] < -2 else ""
        print(f"  Block {k['block']:2d}: WR={k['win_rate']:.1f}% "
              f"[{k['wilson_lo']:.1f}%, {k['wilson_hi']:.1f}%] "
              f"(delta={k['delta']:+.1f}%, completion={k['completion_rate']:.0f}%) {marker}")

    print("\n--- Policy KL Divergence (sorted by impact) ---")
    sorted_kl = sorted(kl_results, key=lambda k: k["kl_mean"], reverse=True)
    for k in sorted_kl:
        phase_str = ", ".join(
            f"{p}={k['phase_kl'][p]['mean']:.4f}" for p in PHASE_LABELS
        )
        print(f"  Block {k['block']:2d}: KL={k['kl_mean']:.4f} +/- {k['kl_std']:.4f}  [{phase_str}]")

    # Identify redundant / critical blocks using KL as primary metric
    kl_threshold_critical = np.percentile([k["kl_mean"] for k in kl_results], 75)
    kl_threshold_redundant = np.percentile([k["kl_mean"] for k in kl_results], 25)

    redundant_wr = [k["block"] for k in results["knockouts"] if k["delta"] > -2]
    critical_wr = [k["block"] for k in results["knockouts"] if k["delta"] < -5]
    redundant_kl = [k["block"] for k in kl_results if k["kl_mean"] <= kl_threshold_redundant]
    critical_kl = [k["block"] for k in kl_results if k["kl_mean"] >= kl_threshold_critical]

    print(f"\nBy win rate — Redundant (delta > -2%): {redundant_wr}")
    print(f"By win rate — Critical (delta < -5%): {critical_wr}")
    print(f"By KL div   — Redundant (bottom 25%):  {redundant_kl}")
    print(f"By KL div   — Critical  (top 25%):     {critical_kl}")
    print(f"Potential compression: blocks with low KL may be removable")


if __name__ == "__main__":
    main()
