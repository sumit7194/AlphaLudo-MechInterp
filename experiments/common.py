import os

import numpy as np
import torch

from src.model import AlphaLudoV5, AlphaLudoV63, AlphaLudoV10

try:
    import td_ludo_cpp as ludo_cpp
except ImportError:
    ludo_cpp = None

# Which AlphaLudo variant the experiments target. Override via env var.
# Default preserves original V6 behavior.
VARIANT = os.environ.get("MECH_INTERP_VARIANT", "v6").lower()

VARIANT_KWARGS = {
    "v6":   {"num_res_blocks": 10, "num_channels": 128, "in_channels": 17},
    "v6_1": {"num_res_blocks": 10, "num_channels": 128, "in_channels": 24},
    "v6_3": {"num_res_blocks": 10, "num_channels": 128, "in_channels": 27},
    "v10":  {"num_res_blocks": 6,  "num_channels": 96,  "in_channels": 28},
}
CHECKPOINT_MODEL_KWARGS = VARIANT_KWARGS[VARIANT]

ENCODER_NAME = {
    "v6":   "encode_state",
    "v6_1": "encode_state_v6",
    "v6_3": "encode_state_v6_3",
    "v10":  "encode_state_v10",
}[VARIANT]

IN_CHANNELS = CHECKPOINT_MODEL_KWARGS["in_channels"]

BASE_POS = -1
HOME_POS = 99


def load_checkpoint_model(weights_path, device="cpu"):
    """Load a checkpoint matching the active VARIANT (default 'v6')."""
    print(f"Loading {VARIANT} model from {weights_path} on {device}...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if VARIANT == "v10":
        model = AlphaLudoV10(**CHECKPOINT_MODEL_KWARGS)
    elif VARIANT == "v6_3":
        model = AlphaLudoV63(**CHECKPOINT_MODEL_KWARGS)
    else:
        model = AlphaLudoV5(**CHECKPOINT_MODEL_KWARGS)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def encode_state(game, consecutive_sixes=0):
    """Encode a GameState using the variant-correct C++ encoder."""
    if ludo_cpp is None:
        raise RuntimeError("td_ludo_cpp not available")
    if VARIANT == "v10":
        return ludo_cpp.encode_state_v10(game)
    if VARIANT == "v6_3":
        return ludo_cpp.encode_state_v6_3(game, consecutive_sixes)
    if VARIANT == "v6_1":
        return ludo_cpp.encode_state_v6(game)
    return ludo_cpp.encode_state(game)


def advance_stuck_turn(game):
    """
    Advance to the next active player when the current player has no legal move.

    The C++ env treats action=-1 as a no-op, so callers need to explicitly
    advance the turn to avoid deadlocked games during random rollouts.
    """
    active_players = np.array(game.active_players, dtype=bool)
    next_player = (int(game.current_player) + 1) % len(active_players)
    while not active_players[next_player]:
        next_player = (next_player + 1) % len(active_players)

    game.current_player = next_player
    game.current_dice_roll = 0


# ── Game-phase helpers ──────────────────────────────────────────────────────


def count_tokens_out(sample):
    """Count how many of the current player's tokens are on the board (not at base, not home)."""
    positions = sample["player_positions"][sample["current_player"]].astype(int)
    return int(np.sum((positions != BASE_POS) & (positions != HOME_POS)))


def game_phase(sample):
    """
    Classify a state into a game phase based on how many tokens the current
    player has on the board.

    Returns:
        "early"  — 0-1 tokens out
        "mid"    — 2 tokens out
        "late"   — 3-4 tokens out
    """
    n = count_tokens_out(sample)
    if n <= 1:
        return "early"
    elif n == 2:
        return "mid"
    else:
        return "late"


PHASE_LABELS = ("early", "mid", "late")


def legal_token_set(sample):
    """Return the set of token indices that are legal moves in this state.

    Requires the td_ludo_cpp module.
    """
    state = ludo_cpp.GameState()
    state.player_positions = sample["player_positions"]
    state.scores = sample["scores"]
    state.active_players = sample["active_players"]
    state.current_player = sample["current_player"]
    state.current_dice_roll = sample["current_dice_roll"]
    state.is_terminal = False
    return set(int(m) for m in ludo_cpp.get_legal_moves(state))


def has_multiple_legal_tokens(sample):
    """True when at least 2 different token indices are legal moves."""
    return len(legal_token_set(sample)) >= 2


# ── Stratified state collection ─────────────────────────────────────────────


def snapshot_state(game, state_tensor):
    """Snapshot a game into a dict suitable for offline analysis."""
    return {
        "tensor": state_tensor.copy(),
        "player_positions": np.array(game.player_positions, dtype=np.int8).copy(),
        "scores": np.array(game.scores, dtype=np.int8).copy(),
        "active_players": np.array(game.active_players, dtype=bool).copy(),
        "current_player": int(game.current_player),
        "current_dice_roll": int(game.current_dice_roll),
    }


def collect_states_stratified(
    num_games=100,
    per_phase_target=200,
    max_loops=5000,
    seed=0,
    require_multi_legal=False,
):
    """
    Collect game states stratified by game phase (early / mid / late).

    This avoids the sampling bias where early-game states (Token 0 is the only
    piece) dominate a random sample.

    Args:
        per_phase_target: states to collect per phase.  Total ≈ 3 × per_phase_target.
        require_multi_legal: if True, only keep states where ≥2 tokens are legal
            moves.  Useful for token-ablation experiments so we never measure
            "what happens when I remove the only option".

    Returns:
        phase_samples: dict  phase_label -> list[sample_dict]
        flat_samples:  list[sample_dict]  (all phases concatenated, shuffled)
    """
    rng = np.random.default_rng(seed)
    env = ludo_cpp.VectorGameState(batch_size=num_games, two_player_mode=True)

    phase_bins = {p: [] for p in PHASE_LABELS}

    loop = 0
    while loop < max_loops:
        loop += 1
        if loop % 200 == 0:
            counts = {p: len(phase_bins[p]) for p in PHASE_LABELS}
            print(f"  ...loop {loop}, phases {counts}")

        for i in range(num_games):
            game = env.get_game(i)
            if not game.is_terminal and game.current_dice_roll == 0:
                game.current_dice_roll = int(rng.integers(1, 7))

        legal_moves_batch = env.get_legal_moves()
        states_tensor = env.get_state_tensor()

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

            action = int(rng.choice(moves))
            actions.append(action)

            sample = snapshot_state(game, states_tensor[i])
            phase = game_phase(sample)

            if len(phase_bins[phase]) >= per_phase_target:
                continue

            if require_multi_legal and not has_multiple_legal_tokens(sample):
                continue

            # Store legal mask alongside the sample
            mask = np.zeros(4, dtype=np.float32)
            for m in moves:
                mask[m] = 1.0
            sample["legal_mask"] = mask

            phase_bins[phase].append(sample)

        _, _, _, infos = env.step(actions)
        for i, info in enumerate(infos):
            if info["is_terminal"]:
                env.reset_game(i)

        if all(len(phase_bins[p]) >= per_phase_target for p in PHASE_LABELS):
            break

    if loop >= max_loops:
        for p in PHASE_LABELS:
            if len(phase_bins[p]) < per_phase_target:
                print(f"  Phase '{p}' under-filled: {len(phase_bins[p])}/{per_phase_target}")

    # Build flat list — equal count per phase, shuffled
    min_count = min(len(phase_bins[p]) for p in PHASE_LABELS)
    flat = []
    for p in PHASE_LABELS:
        flat.extend(phase_bins[p][:min_count])
    rng.shuffle(flat)

    counts = {p: len(phase_bins[p]) for p in PHASE_LABELS}
    print(f"Collected {len(flat)} stratified states (phases: {counts})")

    return phase_bins, flat
