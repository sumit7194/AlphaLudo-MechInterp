import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add the project root to the path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.common import advance_stuck_turn, load_checkpoint_model
import td_ludo_cpp as ludo_cpp

def load_model(weights_path):
    return load_checkpoint_model(weights_path)

def collect_interesting_states(num_games=100, max_steps=200, target_states=15):
    """Collects states where multiple tokens are off base, to see interesting decisions."""
    print(f"Collecting {target_states} interesting states...")
    env = ludo_cpp.VectorGameState(batch_size=num_games, two_player_mode=True)
    
    collected_states = []
    
    for step in range(max_steps):
        for i in range(num_games):
            game = env.get_game(i)
            if not game.is_terminal and game.current_dice_roll == 0:
                game.current_dice_roll = int(np.random.randint(1, 7))
                
        legal_moves_batch = env.get_legal_moves()
        states_tensor = env.get_state_tensor()
        
        actions = []
        for i in range(num_games):
            moves = legal_moves_batch[i]
            game = env.get_game(i)
            
            if game.is_terminal:
                actions.append(-1)
            elif not moves:
                advance_stuck_turn(game)
                actions.append(-1)
            else:
                action = int(np.random.choice(moves))
                actions.append(action)
                
                # State selection criteria: we want states where tokens are spread out.
                # Channel 9 is "My Locked %". A value < 0.75 means at least 2 tokens are out of base.
                my_locked = states_tensor[i][9][0][0]
                if my_locked < 0.75 and len(collected_states) < target_states:
                    if np.random.rand() < 0.2: # Sample to ensure diversity
                        collected_states.append(states_tensor[i].copy())
                        
        _, _, _, infos = env.step(actions)
        
        if len(collected_states) >= target_states:
            break
            
        for i, info in enumerate(infos):
            if info["is_terminal"]:
                env.reset_game(i)
                
    print(f"Collected {len(collected_states)} states.")
    X = torch.tensor(np.array(collected_states), dtype=torch.float32)
    return X


def run_dice_sweep(model, X):
    """
    For each state, sweep dice roll from 1 to 6.
    Returns:
      policy_distributions: np array of shape (N_states, 6_rolls, 4_tokens)
      value_predictions: np array of shape (N_states, 6_rolls)
    """
    num_states = X.shape[0]
    policy_distributions = np.zeros((num_states, 6, 4))
    value_predictions = np.zeros((num_states, 6))
    
    # We do NOT apply legal mask here. We want to see the model's RAW preference.
    # The pure policy distribution will tell us if it attempts to move a locked token when it rolls a 6.
    # We use all 1s mask just to not zero stuff out.
    dummy_mask = torch.ones((num_states, 4), dtype=torch.float32)

    for roll in range(1, 7):
        X_sweep = X.clone()
        # Zero out all dice channels (11 to 16)
        X_sweep[:, 11:17, :, :] = 0.0
        # Set the specific active dice channel to 1.0 (channel 11 is roll 1)
        channel_idx = 11 + (roll - 1)
        X_sweep[:, channel_idx, :, :] = 1.0
        
        with torch.no_grad():
            policy, value = model(X_sweep, dummy_mask)
            
        policy_distributions[:, roll-1, :] = policy.numpy()
        value_predictions[:, roll-1] = value.squeeze(-1).numpy()
        
    return policy_distributions, value_predictions


def visualize_sweep(policy_distributions, value_predictions, save_path):
    num_states = policy_distributions.shape[0]
    
    fig, axes = plt.subplots(nrows=num_states, ncols=1, figsize=(10, 2 * num_states))
    fig.suptitle('Policy Distribution across Dice Rolls (1-6) per State', fontsize=16, y=0.99)
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    labels = ['Token 0', 'Token 1', 'Token 2', 'Token 3']
    
    for s in range(num_states):
        ax = axes[s] if num_states > 1 else axes
        
        # Policy for this state (6 rolls x 4 tokens)
        pol = policy_distributions[s]
        
        # We want a stacked bar chart. X axis is rolls (1-6)
        x = np.arange(1, 7)
        bottoms = np.zeros(6)
        
        for t in range(4):
            ax.bar(x, pol[:, t], bottom=bottoms, color=colors[t], edgecolor='white', label=labels[t] if s == 0 else "")
            bottoms += pol[:, t]
            
        # Add critic output text above each bar
        for r in range(6):
            v = value_predictions[s, r]
            ax.text(x[r], 1.05, f"C:{v:+.2f}", ha='center', va='bottom', fontsize=8, rotation=45)
            
        ax.set_ylim(0, 1.3) # Space for value text
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel(f'State {s+1}')
        ax.set_xticks(x)
        if s == num_states - 1:
            ax.set_xlabel('Simulated Dice Roll')
        else:
            ax.set_xticklabels([])
            
    if num_states > 1:
        fig.legend(labels, loc='upper right', bbox_to_anchor=(0.95, 0.98), ncol=4)
        
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    weights_path = "../../weights/model_latest_323k_shaped.pt"
    if not os.path.exists(weights_path):
        weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../weights/model_latest_323k_shaped.pt"))
        
    model = load_model(weights_path)
    X = collect_interesting_states(num_games=100, target_states=15)
    
    if len(X) > 0:
        pol, val = run_dice_sweep(model, X)
        save_path = os.path.join(os.path.dirname(__file__), "dice_sensitivity_results.png")
        visualize_sweep(pol, val, save_path)
    else:
        print("Failed to collect any states.")
