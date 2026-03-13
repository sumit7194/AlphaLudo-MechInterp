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

def collect_states(num_games=200, max_steps_per_game=200):
    print(f"Collecting states from {num_games} random games...")
    env = ludo_cpp.VectorGameState(batch_size=num_games, two_player_mode=True)
    
    collected_states = []
    collected_masks = []
    
    for step in range(max_steps_per_game):
        legal_moves_batch = env.get_legal_moves()
        states_tensor = env.get_state_tensor() # (B, 17, 15, 15) numpy array
        
        for i in range(num_games):
            game = env.get_game(i)
            # If game needs a dice roll, we must roll it
            if not game.is_terminal and game.current_dice_roll == 0:
                game.current_dice_roll = int(np.random.randint(1, 7))
                
        # Re-fetch legal moves after rolling dice
        legal_moves_batch = env.get_legal_moves()
        states_tensor = env.get_state_tensor() # (B, 17, 15, 15) numpy array
        
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
                # Random legal move
                action = int(np.random.choice(moves))
                actions.append(action)
                
                # Save state if it's a valid decision point
                if len(collected_states) < 500: # Collect up to 500
                    # Create legal mask
                    mask = np.zeros(4, dtype=np.float32)
                    for m in moves:
                        mask[m] = 1.0
                    
                    collected_states.append(states_tensor[i].copy())
                    collected_masks.append(mask)
                    
        _, _, _, infos = env.step(actions)
        
        # Stop early if we have enough states
        if len(collected_states) >= 500:
            break
            
        # If all games are terminal, let's reset them so we can keep collecting
        for i, info in enumerate(infos):
            if info["is_terminal"]:
                env.reset_game(i)
            
    print(f"Collected {len(collected_states)} decision states.")
    
    # Take up to 500 states to keep the experiment fast
    max_states = min(500, len(collected_states))
    indices = np.random.choice(len(collected_states), max_states, replace=False)
    
    X = torch.tensor(np.array([collected_states[i] for i in indices]), dtype=torch.float32)
    masks = torch.tensor(np.array([collected_masks[i] for i in indices]), dtype=torch.float32)
    
    return X, masks

def kl_divergence(p, q):
    # p and q are batches of probability distributions (B, 4)
    # KL(p || q) = sum(p * log(p / q)) = sum(p * log(p) - p * log(q))
    # Add epsilon to prevent log(0)
    eps = 1e-8
    p_safe = p + eps
    q_safe = q + eps
    return torch.sum(p * (torch.log(p_safe) - torch.log(q_safe)), dim=1)

def run_ablation(model, X, masks):
    print("Running baseline predictions...")
    with torch.no_grad():
        baseline_policy, baseline_value = model(X, masks)
        
    num_channels = X.shape[1]
    policy_kl_impacts = []
    value_mae_impacts = []
    
    print(f"Ablating {num_channels} channels individually...")
    for c in range(num_channels):
        # Create an ablated copy where channel c is zeroed out
        X_ablated = X.clone()
        X_ablated[:, c, :, :] = 0.0
        
        with torch.no_grad():
            ablated_policy, ablated_value = model(X_ablated, masks)
            
        # Compute KL divergence for policy
        kl = kl_divergence(baseline_policy, ablated_policy).mean().item()
        
        # Compute Mean Absolute Error for value
        v_mae = torch.abs(baseline_value - ablated_value).mean().item()
        
        policy_kl_impacts.append(kl)
        value_mae_impacts.append(v_mae)
        print(f"  Channel {c:2d} -> Policy KL: {kl:.4f}, Value MAE: {v_mae:.4f}")
        
    return policy_kl_impacts, value_mae_impacts

def visualize(policy_impacts, value_impacts, save_path):
    print(f"Generating visualization at {save_path}...")
    
    channel_names = [
        "0: My Token 0",
        "1: My Token 1",
        "2: My Token 2",
        "3: My Token 3",
        "4: Opp Density",
        "5: Safe Zones",
        "6: My Home Path",
        "7: Opp Home Path",
        "8: Score Diff",
        "9: My Locked %",
        "10: Opp Locked %",
        "11: Dice = 1",
        "12: Dice = 2",
        "13: Dice = 3",
        "14: Dice = 4",
        "15: Dice = 5",
        "16: Dice = 6"
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    y_pos = np.arange(len(channel_names))
    
    # Policy Impact plot
    ax1.barh(y_pos, policy_impacts, align='center', color='skyblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(channel_names)
    ax1.invert_yaxis()  # labels read top-to-bottom
    ax1.set_xlabel('Average KL Divergence')
    ax1.set_title('Impact on Policy Decision (Higher = More Important)')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Value Impact plot
    ax2.barh(y_pos, value_impacts, align='center', color='salmon')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(channel_names)
    ax2.invert_yaxis()
    ax2.set_xlabel('Average Absolute Critic Shift')
    ax2.set_title('Impact on Critic Output')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print("Done!")

if __name__ == "__main__":
    weights_path = "../../weights/model_latest_323k_shaped.pt"
    if not os.path.exists(weights_path):
        weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../weights/model_latest_323k_shaped.pt"))
        
    model = load_model(weights_path)
    X, masks = collect_states(num_games=100) # using 100 for speed
    p_impact, v_impact = run_ablation(model, X, masks)
    
    save_path = os.path.join(os.path.dirname(__file__), "channel_ablation_results.png")
    visualize(p_impact, v_impact, save_path)
