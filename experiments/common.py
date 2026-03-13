import numpy as np
import torch

from src.model import AlphaLudoV5


CHECKPOINT_MODEL_KWARGS = {
    "num_res_blocks": 10,
    "num_channels": 128,
    "in_channels": 17,
}


def load_checkpoint_model(weights_path, device="cpu"):
    """Load the exported AlphaLudo checkpoint with the correct architecture."""
    print(f"Loading model from {weights_path} on {device}...")
    checkpoint = torch.load(weights_path, map_location=device)
    model = AlphaLudoV5(**CHECKPOINT_MODEL_KWARGS)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


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
