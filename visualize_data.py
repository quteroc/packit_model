# language: python
# filepath: /c:/Users/butel/General_PackIt!/visualize_data.py

import json
import torch
import matplotlib.pyplot as plt
from packit_env import PackItEnv  # see [packit_env.py](packit_env.py)
from MCTS_alphazero_extra import MCTS  # see [MCTS_alphazero_extra.py](MCTS_alphazero_extra.py)
from nn_model_extra import AlphaZeroNet  # see [nn_model_extra.py](nn_model_extra.py)

def load_training_stats(filename="training_stats.json"):
    """Load training stats from a JSON file."""
    try:
        with open(filename, "r") as f:
            stats = json.load(f)
        # Expecting stats stored under key 'loss_stats'
        games = [entry["game"] for entry in stats.get("loss_stats", [])]
        losses = [entry["avg_loss"] for entry in stats.get("loss_stats", [])]
        return games, losses
    except Exception as e:
        print(f"Error loading training stats: {e}")
        return [], []

def plot_training_stats(games, losses):
    plt.figure(figsize=(8, 5))
    plt.plot(games, losses, marker="o")
    plt.title("Training Loss over Self-play Games")
    plt.xlabel("Game Number")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.show()

def simulate_model_game_6x6(device='cpu'):
    """
    Simulate a single game between two model players on a 5x5 board.
    Returns the winner as 1 (first mover) or 2 (second mover).
    """
    board_size = 6
    env = PackItEnv(board_size=board_size)
    num_actions = board_size ** 4
    # Note: extra_features_dim must match training (here assumed to be 7).
    net = AlphaZeroNet(board_size, in_channels=1, num_actions=num_actions, extra_features_dim=7)
    try:
        net.load_state_dict(torch.load("alphazero6x6.pth", map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    net.to(device)
    net.eval()

    done = False
    while not done:
        mcts = MCTS(env, net, device=device)
        mcts_policy = mcts.get_move(temperature=1)
        best_move = max(mcts_policy.items(), key=lambda x: x[1])[0]
        _, reward, done, _ = env.step(best_move)
    
    # Determine the winner.
    # Here we assume the winner is the opponent of the last mover.
    winner = 1 if env.current_player == 2 else 2
    return winner

def simulate_model_games_6x6(num_games=100, device='cpu'):
    wins = {1: 0, 2: 0}
    for i in range(num_games):
        winner = simulate_model_game_6x6(device=device)
        wins[winner] += 1
        print(f"Game {i+1}/{num_games} winner: Player {winner}")
    return wins

def plot_model_vs_model(wins):
    players = ['Player 1 (First mover)', 'Player 2 (Second mover)']
    win_counts = [wins.get(1, 0), wins.get(2, 0)]
    plt.figure(figsize=(6, 4))
    plt.bar(players, win_counts, color=['blue', 'red'])
    plt.title("Model vs Model Wins (100 games on 5x5)")
    plt.ylabel("Number of Wins")
    plt.savefig("model_vs_model_5x5.png")
    plt.show()

if __name__ == "__main__":
    # Visualize training loss.
    games, losses = load_training_stats()
    if games and losses:
        plot_training_stats(games, losses)
    else:
        print("No training stats available to plot.")

    # Simulate 100 games of model vs model on a proper board.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wins = simulate_model_games_6x6(num_games=100, device=device)
    print("Simulation results:", wins)
    plot_model_vs_model(wins)