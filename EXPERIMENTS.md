# AlphaLudo — Mechanistic Interpretability Experiments

A series of experiments to understand **what the model has learned** and **how it makes decisions**, ordered from easiest to most advanced.

> **Model**: AlphaLudoV5 — 5 ResBlocks, 64ch, ~250K params
> **Checkpoint**: `weights/model_latest_323k_shaped.pt` (PPO + shaped rewards, 323K games)
> **Input**: `(B, 17, 15, 15)` · **Policy**: 4 token logits · **Value**: 1 scalar

---

## Experiment Tracker

| # | Experiment | Difficulty | Status |
|---|-----------|-----------|--------|
| 1 | Channel Ablation Study | ⭐ Easy | ✅ Done |
| 2 | Dice Sensitivity Analysis | ⭐ Easy | ✅ Done |
| 3 | Linear Probes on GAP Features | ⭐⭐ Medium | 🔲 Not started |
| 4 | Logit Lens Across Residual Blocks | ⭐⭐ Medium | 🔲 Not started |
| 5 | Saliency / Input Gradient Attribution | ⭐⭐ Medium | 🔲 Not started |
| 6 | Policy vs Value Head Divergence | ⭐⭐ Medium | 🔲 Not started |
| 7 | Residual Block Importance (Skip-Block) | ⭐⭐⭐ Hard | 🔲 Not started |
| 8 | Concept-Specific Neurons | ⭐⭐⭐ Hard | 🔲 Not started |
| 9 | Sparse Autoencoders (SAEs) | ⭐⭐⭐⭐ Advanced | 🔲 Not started |

---

## Experiment 1 — Channel Ablation Study

**Core Question**: Which of the 17 input channels does the model actually rely on?

**Method**:
- Generate a diverse set of game states (varying stages, dice rolls, token positions)
- For each state, zero out one input channel at a time (channels 0–16)
- Record the change in policy distribution (KL divergence from baseline) and value shift
- Rank channels by importance

**What to Look For**:
- Do dice channels (11–16) dominate, or does the model rely equally on spatial info?
- Is the safe-zone mask (ch 5) important, or has the model learned safety from token positions alone?
- How much do the broadcast stats (ch 8–10: score diff, locked fractions) influence the value head vs the policy head?
- Do individual token channels (0–3) matter differently (e.g., does the model pay more attention to the leading token)?

**Expected Output**: Bar chart of per-channel importance for both policy and value heads.

---

## Experiment 2 — Dice Sensitivity Analysis

**Core Question**: How does the model's strategy change with different dice rolls?

**Method**:
- Pick ~10 representative board states (early game, mid game, capture opportunity, home stretch, etc.)
- For each state, create 6 copies varying only the dice channel (one-hot for rolls 1–6)
- Record the full policy distribution and value for each roll
- Visualize how the token preference shifts across rolls

**What to Look For**:
- Does a roll of 6 cause a dramatic shift (since 6 lets tokens leave base)?
- Are there states where the model's preferred token completely flips based on dice?
- Does the value head respond to dice? (It shouldn't much for "fair" positions, since dice are random)
- Evidence of roll-conditional tactics (e.g., "with a 1, advance the token closest to home")

**Expected Output**: Heatmap grid — rows = board states, columns = dice values, cells = policy distribution stacked bars.

---

## Experiment 3 — Linear Probes on GAP Features

**Core Question**: What game concepts does the model represent in its 64-dim backbone features?

**Method**:
- Collect ~5K–10K game states with labels for various concepts:
  - "Can I capture an opponent this turn?" (binary)
  - "Is my leading token in danger?" (binary)
  - "How many tokens are in the home stretch?" (0–4)
  - "Am I winning?" (binary, from ground-truth game outcome)
  - "Which token is closest to home?" (categorical, 0–3)
- Run each state through the backbone, extract the 64-dim post-GAP feature vector
- Train a logistic/linear regression on top of these features for each concept
- Report accuracy / AUC

**What to Look For**:
- Which concepts are **linearly decodable** (high accuracy = the model represents this explicitly)?
- Which concepts are NOT linearly represented (low accuracy = either the model doesn't track this, or it's stored in a nonlinear way)?
- Compare probing accuracy at different depths (after each residual block) to see where concepts emerge

**Expected Output**: Table of (concept × probe accuracy) and optionally a depth-vs-accuracy curve.

---

## Experiment 4 — Logit Lens Across Residual Blocks

**Core Question**: How does the model's decision build up through its layers?

**Method**:
- For a given game state, extract the feature map after each residual block (5 checkpoints)
- At each checkpoint apply GAP + the policy FC layers to get "early predictions"
- Compare the intermediate policy/value predictions to the final output

**What to Look For**:
- Does the value head converge quickly (e.g., block 2) while the policy head needs all 5 blocks?
- Are there states where the policy *flips* mid-network (e.g., block 3 prefers token 0, but block 5 prefers token 2)?
- Which blocks cause the biggest "jumps" in prediction confidence?

**Expected Output**: Line chart per game state — x-axis = block index, y-axis = policy logits or value, showing the build-up.

---

## Experiment 5 — Saliency / Input Gradient Attribution

**Core Question**: Which spatial cells on the 15×15 board matter most for a specific decision?

**Method**:
- For a chosen game state + decision (e.g., "model picks token 2"), compute the gradient of that policy logit w.r.t. the full `(17, 15, 15)` input tensor
- Aggregate gradients across channels to produce a saliency heatmap over the board
- Optionally: use Integrated Gradients for a cleaner attribution

**What to Look For**:
- Does the model focus on the chosen token's position and the path ahead of it?
- Does it look at opponent tokens that could be captured or might capture it?
- Does it attend to safe zones near its tokens?
- Are there surprising spatial regions the model looks at (e.g., the opponent's home run)?

**Expected Output**: 15×15 saliency heatmaps overlaid on the board layout, one per decision.

---

## Experiment 6 — Policy vs Value Head Divergence

**Core Question**: Do the policy and value heads "see" different things?

**Method**:
- Collect a large set of game states and record both policy entropy and value
- Find "disagreement" states: low value (thinks it's losing) but high policy confidence, or vice versa
- Analyze what makes these states special
- Use gradient-based attribution separately for each head to compare what board features each relies on

**What to Look For**:
- States where the model "knows it's losing but still has a clear best move" (desperation moves?)
- States where value is high but policy is uniform (winning no matter what?)
- Whether the value head relies more on broadcast stats while the policy head relies more on spatial channels

**Expected Output**: Scatter plot of (value, policy entropy) with highlighted outlier states analyzed qualitatively.

---

## Experiment 7 — Residual Block Importance (Skip-Block)

**Core Question**: Which residual blocks are load-bearing?

**Method**:
- For each of the 5 residual blocks: replace it with an identity (skip it)
- Measure degradation in policy accuracy (KL-div from full model) and value accuracy (MSE from full model) across many game states
- Optionally: also try removing pairs of blocks

**What to Look For**:
- Are later blocks more important (deeper = more abstract reasoning)?
- Are any blocks redundant (can be removed with minimal impact)?
- Do different blocks matter for policy vs value?

**Expected Output**: Table/chart of per-block degradation metrics.

---

## Experiment 8 — Concept-Specific Neurons

**Core Question**: Are there individual neurons (among the 64 post-GAP features) that encode specific Ludo concepts?

**Method**:
- Collect game states with known properties (capture available, token at risk, base-exit possible, etc.)
- For each of the 64 features, compute correlation with each game property
- Identify monosemantic neurons (high correlation with one concept) vs polysemantic ones (correlated with multiple concepts)
- Manually inspect the top game states that maximize/minimize each neuron

**What to Look For**:
- "Capture neurons" that fire specifically when a capture opportunity exists
- "Danger neurons" that activate when a token is about to be sent back to base
- "Endgame neurons" that light up when tokens are in the home stretch
- Polysemantic neurons that blend multiple concepts

**Expected Output**: Correlation matrix (64 neurons × N concepts) and per-neuron "max-activating states" gallery.

---

## Experiment 9 — Sparse Autoencoders (SAEs)

**Core Question**: Can we decompose the model's representations into more interpretable features than the raw 64 neurons?

**Method**:
- Collect 50K+ post-GAP feature vectors from diverse game states
- Train a sparse autoencoder (overcomplete: 64 → ~256–512 features → 64) with an L1 sparsity penalty
- Analyze the learned dictionary features: what game states activate each feature?
- Compare SAE features to the concepts found in Experiment 8

**What to Look For**:
- Do SAE features split polysemantic neurons into cleaner, monosemantic concepts?
- Can we find features that correspond to specific Ludo tactics (blocking, racing, safe-zone usage)?
- How many "dead" features are there (never activate)?
- Can we do targeted interventions (clamp a feature to zero → observe behavior change)?

**Expected Output**: Dashboard of the top SAE features with their activating game states, plus intervention results.

---

## Notes

- All experiments will be implemented as Jupyter notebooks or standalone scripts in a `experiments/` directory
- Each experiment should save results (figures, metrics) alongside its code
- We'll share the same model-loading and state-generation boilerplate across experiments
