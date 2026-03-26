# AlphaLudo MechInterp — Experiment Improvement Journal

**Date**: 2026-03-26
**Model**: AlphaLudoV5 (10 ResBlocks, 128 channels, 17 input channels)
**Checkpoint**: model_latest_532k.pt (532K training games)

---

## The Token 0 Bias Discovery

### What we observed
Experiment 1 (Channel Ablation) showed Token 0's input channel (ch 0) had dramatically higher policy KL divergence when ablated compared to Tokens 1-3. This appeared to indicate the model was biased toward Token 0.

### Root cause: Sampling bias, not model bias
The experiment collected 500 random decision states from random game rollouts. In Ludo, games start with all 4 tokens at base. Tokens deploy one at a time (only on a roll of 6). This means:
- **Early-game states dominate random samples** — most of the game is spent with 0-1 tokens on the board
- When only Token 0 is on the board, it is literally the **only legal move**
- Zeroing out Token 0's channel in those states removes the only relevant information
- The resulting high KL divergence measures "what happens when I remove the only option", not "does the model prefer this token"

### The confirmation: Gradient attribution (Experiment 4)
The gradient attribution showed Channel 4 (opponent density) dominates at 23.7%, and all 4 token channels are roughly equal (6.8-7.3%). This is the real picture — the model is NOT biased toward Token 0.

---

## Fixes Applied Across All Experiments

### 1. Phase-Stratified Sampling (ALL experiments)
**Problem**: Random game rollouts oversample early-game states.

**Fix**: Added `collect_states_stratified()` to `experiments/common.py` that:
- Classifies states into phases: **early** (0-1 tokens out), **mid** (2 tokens out), **late** (3-4 tokens out)
- Collects equal numbers per phase
- Reports phase distributions
- Available to all experiments via shared import

**Impact**: Every experiment now has balanced representation across game phases.

### 2. Multi-Legal Token Filter (Experiment 1)
**Problem**: Ablating a token channel when that token is the only legal move measures "removing the only option", not "model preference".

**Fix**: Added `require_multi_legal=True` option to stratified collection. Only includes states where >= 2 tokens are legal moves.

**Rule for future experiments**: Only ablate in states where the ablated token AND at least one other token are both legal moves.

### 3. Backward Compatibility
All experiments accept `--no-stratify` to revert to the original random sampling. This allows comparing old vs new methodology.

---

## Per-Experiment Improvements

### Experiment 1: Channel Ablation
| Change | Why |
|--------|-----|
| Stratified sampling with `require_multi_legal=True` | Eliminates early-game Token 0 dominance |
| **Swap test** (swap channels i<->j) | Distinguishes spatial-position vs channel-index sensitivity. If model prefers "Token 0", swapping ch0<->ch2 should move preference to follow the channel, not the spatial position |
| Per-phase ablation results | Token 0 KL=0.55 overall means nothing if it's 0.95 in early-game and 0.15 in late-game |
| Per-phase visualizations | Separate PNGs for early/mid/late |

**Key insight for interpretation**: The swap test is more informative than zeroing because it distinguishes *spatial-position sensitivity* from *channel-index sensitivity*.

### Experiment 2: Dice Sensitivity
| Change | Why |
|--------|-----|
| Stratified sampling | Dice sensitivity may differ across game phases |
| Per-phase dice sweep | Early game (few options) vs late game (many options) may show very different sensitivity patterns |
| Bucket collection from stratified pool first | More efficient; falls back to extra random collection only if needed |

### Experiment 3: Linear Probes
| Change | Why |
|--------|-----|
| Phase-stratified labeled dataset | Probes trained on biased samples may learn "is this an early-game state?" rather than the intended concept |
| New probe: `game_phase` | Sanity check — should be highly decodable. If it's not, the model may not internally represent game stage |
| New probe: `num_tokens_out` | Tests whether the model knows how many pieces are deployed |
| Post-stratification balance reporting | Verifies the fix actually worked |

**Key insight**: The `eventual_win` probe is noisy because it depends on random opponent play. Consider replacing with MCTS win estimate in future experiments.

### Experiment 4: Layer Knockout
| Change | Why |
|--------|-----|
| **Policy KL divergence** metric | Win rate vs random is saturated (100% even with blocks removed). KL divergence between full-model and knocked-out-model policy is far more sensitive |
| Wilson score confidence intervals | 200 games isn't enough for precise win rate estimates |
| Default games: 200 -> 500 | Better statistical power |
| Fix unfinished games | Were incorrectly counted as wins; now excluded with completion rate reported separately |
| Per-phase KL divergence | Some blocks may be critical only in specific game phases |

**Key insight**: Win rate vs random opponent has saturated for this model. Future experiments should use self-play or MCTS as the opponent baseline, or rely primarily on policy KL divergence.

### Experiment 5: Channel Activation
| Change | Why |
|--------|-----|
| Stratified sampling | Activation patterns may differ between early game (sparse board) and late game (crowded board) |
| Per-phase dead channel analysis | A channel may be "dead" in early game but alive in late game |
| Phase differential reporting | Identifies channels that are phase-specific vs universally dead |

### Experiment 6: CKA Similarity
| Change | Why |
|--------|-----|
| Stratified sampling | Representational similarity may differ across game phases |
| Per-phase CKA matrices | Blocks 7-8-9 might be redundant in early game but distinct in late game |
| Cross-phase comparison | Flags pairs whose CKA varies > 0.05 across phases |
| Grouped bar chart | Visual comparison of consecutive CKA across phases |

---

## General Principles for Future Experiments

### 1. Always stratify by game phase
Equal numbers of early (0-1 tokens out), mid (2 tokens out), and late (3-4 tokens out) states. Never use raw random samples.

### 2. Only ablate when alternatives exist
When ablating a token's channel, ensure that token AND at least one other token are both legal moves. Otherwise you're measuring "removing the only option".

### 3. Use the swap test as ground truth
Swapping channels i<->j is more informative than zeroing:
- **Zeroing** measures: "how much does this channel matter?"
- **Swapping** measures: "does the model track this channel's identity?"

### 4. Report per-phase results separately
Aggregate numbers can be misleading. Token 0 KL=0.55 overall means nothing if it's 0.95 in early-game and 0.15 in late-game.

### 5. Win rate vs random is a floor, not a ceiling
Once the model beats random consistently, win rate differences disappear. Use:
- Policy KL divergence (most sensitive)
- Self-play Elo
- MCTS vs MCTS with modified network

### 6. Be skeptical of dominant features
If one feature appears disproportionately important, ask: "Is this because the model is biased, or because my sample is biased?" Check by:
- Stratifying the sample
- Filtering to states where alternatives exist
- Running the swap test

### 7. Probe labels should be deterministic
`eventual_win` depends on the random opponent's play, making it noisy. Prefer deterministic labels derived from the game state itself (capture available, danger, home stretch count, etc.).

---

## File Changes Summary

| File | Status |
|------|--------|
| `experiments/common.py` | Added: `count_tokens_out`, `game_phase`, `legal_token_set`, `has_multiple_legal_tokens`, `snapshot_state`, `collect_states_stratified` |
| `experiments/01_channel_ablation/run_ablation.py` | Added: stratified sampling, swap test, per-phase reporting, `--no-stratify` flag |
| `experiments/02_dice_sensitivity/run_dice_sensitivity.py` | Added: stratified sampling, per-phase analysis, `--no-stratify` flag |
| `experiments/03_linear_probes/run_linear_probes.py` | Added: stratified collection, `game_phase` + `num_tokens_out` probes, `--no-stratify` flag |
| `experiments/04_layer_knockout/run_layer_knockout.py` | Added: policy KL metric, Wilson CI, fixed unfinished games, increased defaults, per-phase KL, `--no-stratify` flag |
| `experiments/05_channel_activation/run_channel_activation.py` | Added: stratified sampling, per-phase analysis, phase differential, `--no-stratify` flag |
| `experiments/06_cka_similarity/run_cka_similarity.py` | Added: stratified sampling, per-phase CKA, cross-phase comparison, `--no-stratify` flag |
