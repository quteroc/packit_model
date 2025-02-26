# language: python
import torch
from torchsummary import summary
from torchviz import make_dot
import shap
import matplotlib.pyplot as plt
from graphviz import Digraph
import numpy as np
import random
from multiprocessing import Pool

from nn_model_extra import AlphaZeroNet  # [nn_model_extra.py]
from MCTS_alphazero_extra import MCTS    # [MCTS_alphazero_extra.py]
from packit_env import PackItEnv         # [packit_env.py]

# ----- Part 1. Visualize the Neural Network Architecture -----
# language: python
# language: python

# def visualize_nn_architecture(board_size=6, in_channels=1, extra_features_dim=7):
#     num_actions = board_size ** 4
#     device = 'cpu'
#     model = AlphaZeroNet(board_size, in_channels, num_actions, extra_features_dim).to(device)
#     model.eval()

#     # Generate dummy inputs on CPU.
#     dummy_board = torch.randn(1, in_channels, board_size, board_size, device=device)
#     dummy_extra = torch.randn(1, extra_features_dim, device=device)

#     # Print summary using input_size instead of input_data.
#     print("=== Model Summary ===")
#     summary(model, input_size=[(in_channels, board_size, board_size), (extra_features_dim,)], device=device)

#     # Generate a dummy forward pass and create a graph.
#     log_probs, value = model(dummy_board, dummy_extra)
#     dot = make_dot(log_probs, params=dict(list(model.named_parameters())))
#     dot.format = 'png'
#     filename = f"nn_architecture_{board_size}x{board_size}"
#     dot.render(filename)
#     print(f"Neural Network architecture graph saved as '{filename}.png'.")

# ----- Part 2. Visualize a High-Level MCTS Flow Diagram -----
def visualize_mcts_structure():
    dot = Digraph(comment='MCTS Flow')
    dot.node('S', 'Start at Root Node\n(deepcopy(env))')
    dot.node('E', 'Evaluate leaf node\n(with NN and extra_features)')
    dot.node('X', 'Expand: add children nodes\n(assign priors)')
    dot.node('U', 'Select Best Child\n(using UCB)')
    dot.node('B', 'Backpropagate Value')
    
    dot.edges(['SE', 'EX'])
    dot.edge('S', 'U', label='if expanded')
    dot.edge('U', 'B')
    dot.edge('B', 'S', label='update\n(visits, Q)')
    
    dot.render('mcts_structure', format='png', cleanup=True)
    print("MCTS flow structure diagram saved as 'mcts_structure.png'.")

# Helper function for SHAP analysis
def simulate_game_features(board_size):
    """
    Simulate one game by choosing random legal moves.
    Collect extra_features from every move (ignoring the very first state if needed).
    Returns a list of extra_features samples (each as a numpy array).
    """
    env = PackItEnv(board_size=board_size)
    env.reset()
    samples = []
    done = False
    while not done:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            break
        # Collect extra_features at this state.
        features = env.get_extra_features().cpu().numpy().squeeze()
        samples.append(features)
        move = random.choice(legal_moves)
        _, reward, done, _ = env.step(move)
    return samples

# Helper function for SHAP analysis
def collect_random_game_features(board_size, num_games=40, processes=4):
    """
    Run num_games in parallel to gather extra_features.
    Returns an array of collected feature vectors.
    """
    with Pool(processes=processes) as pool:
        results = pool.map(simulate_game_features, [board_size] * num_games)
    # Flatten the list of lists.
    all_samples = [feat for game in results for feat in game]
    return np.array(all_samples)

# Helper function for SHAP analysis
def build_background_from_distribution(feature_samples, background_size=20):
    """
    Given an array of feature samples (shape: [N, extra_features_dim]),
    compute mean and std for each column and sample background_size points
    from the estimated normal distributions.
    """
    # Estimate per-feature mean and std.
    mu = np.mean(feature_samples, axis=0)
    sigma = np.std(feature_samples, axis=0)
    # Sample background points.
    background = np.random.normal(loc=mu, scale=sigma, size=(background_size, feature_samples.shape[1]))
    return background

# ----- Part 3. Use SHAP to analyze extra_features effect -----
def shap_extra_features_analysis(board_size=6, in_channels=1, extra_features_dim=7, num_samples=100):
    num_actions = board_size ** 4
    device = 'cpu'
    model = AlphaZeroNet(board_size, in_channels, num_actions, extra_features_dim).to(device)
    model.eval()

    # Instead of using just 20 resets, simulate 40 games in parallel to collect extra_features
    all_feature_samples = collect_random_game_features(board_size, num_games=40, processes=4)
    # Build a background based on the distribution of extra_features you collected:
    background = build_background_from_distribution(all_feature_samples, background_size=20)
    
    # For test samples, either collect fresh samples or randomly select from all_feature_samples.
    test_samples = all_feature_samples[
        np.random.choice(all_feature_samples.shape[0], size=num_samples, replace=True)
    ]

    # Prepare fixed board input – this will be paired with varying extra_features
    fixed_board = torch.randn(1, in_channels, board_size, board_size).to(device)
    
    def model_wrapper(extra):
        # extra is expected to have shape [batch, extra_features_dim]
        extra = extra.float()
        batch_size = extra.shape[0]
        board_rep = fixed_board.repeat(batch_size, 1, 1, 1)
        # Only use the value head for the SHAP analysis.
        _, value = model(board_rep, extra)
        return value

    explainer = shap.DeepExplainer(model_wrapper, background)
    shap_values = explainer.shap_values(test_samples)
    shap_values = shap_values[0]  # use the output for the value head

    plt.title("SHAP Summary for Extra Features")
    shap.summary_plot(shap_values, test_samples, feature_names=[f"feat{i+1}" for i in range(extra_features_dim)])
    shap_filename = f"shap_extra_features_summary_{board_size}x{board_size}.png"
    plt.savefig(shap_filename)
    plt.show()
    print(f"SHAP analysis for extra_features saved as '{shap_filename}'.")

if __name__ == "__main__":
    # Visualize a high-level MCTS structure.
    visualize_mcts_structure()

    board_size = 6
    in_channels = 1
    extra_features_dim = 7

    # Visualize the neural network architecture.
    # visualize_nn_architecture(board_size, in_channels, extra_features_dim)
    
    # Perform SHAP analysis for extra features.
    shap_extra_features_analysis(board_size, in_channels, extra_features_dim)

