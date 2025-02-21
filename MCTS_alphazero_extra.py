# language: python
# filepath: /c:/Users/butel/General_PackIt!/MCTS_alphazero_extra.py

import math
import numpy as np
import torch
import random
from copy import deepcopy
from packit_env import PackItEnv  # [packit_env.py](packit_env.py)
from nn_model_extra import AlphaZeroNet  # [nn_model.py](nn_model.py)

def action_to_index(action, board_size):
    #Map an action tuple (row, col, width, height) to a flattened index.
    #FIX: As width and height belong to [1, board_size], we need to subtract 1 to get the correct index
    row, col, width, height = action
    return row * (board_size ** 3) + col * (board_size**2) + (width - 1) * board_size + (height - 1)

def index_to_action(index, board_size):
    #Inverse mapping from index to (row, col, width, height).
    #FIX: As width and height belong to [1, board_size], we need to add 1 to get the correct value
    value = index

    height = value % board_size + 1
    value = value // board_size

    width = value % board_size + 1
    value = value // board_size

    col = value % board_size
    value = value // board_size

    row = value
    return (row, col, width, height)


class MCTSNode:
    def __init__(self, env: PackItEnv, parent=None, action=None, prior=0.0):
        self.env = env                        # Current environment state at this node.
        self.parent = parent                  # Parent node.
        self.action = action                  # Action taken to reach this node.
        self.children = {}                    # Mapping: action -> MCTSNode.
        self.N = 0                          # Visit count.
        self.W = 0.0                        # Total value.
        self.Q = 0.0                        # Mean value.
        self.P = prior                      # Prior probability.

    def value(self):
        # Return mean value; if not visited yet, return 0.
        return self.W / self.N if self.N > 0 else 0.0

class MCTS:
    def __init__(self, env: PackItEnv, net: AlphaZeroNet, device='cpu',
                 c_puct=1.0, num_simulations=100):
        self.env = env                      # The current environment.
        self.net = net                      # The neural network.
        self.device = device
        # Constant board features.
        self.m_const = env.m
        self.n_const = env.n
        self.gap = env.gap
        self.P_val = env.P_val
        self.K = env.K
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def get_move(self, temperature=1.0):
        # Initialize the root node with a deepcopy of the current environment.
        self.root = MCTSNode(env=deepcopy(self.env))
        for _ in range(self.num_simulations):
            self.simulate(self.root)
        actions = list(self.root.children.keys())
        visits = np.array([self.root.children[action].N for action in actions])
        if temperature == 0:
            best_action = actions[np.argmax(visits)]
            return {best_action: 1.0}
        adjusted = visits ** (1 / temperature)
        probs = adjusted / np.sum(adjusted)
        return {action: prob for action, prob in zip(actions, probs)}

    def simulate(self, node: MCTSNode):
        # Use the absence of legal actions as the terminal condition.
        legal_actions = node.env.get_legal_moves()
        if not legal_actions:
            return -node.env.get_reward()  # Terminal node.
        # If node is not expanded yet, expand it.
        if not node.children:
            return self.expand(node)
        # Otherwise, select the best child using UCB.
        best_score = -float("inf")
        best_action = None
        best_child = None
        for action, child in node.children.items():
            ucb = child.Q + self.c_puct * child.P * math.sqrt(node.N) / (1 + child.N)
            # if random.random() < 0.001:
            #     # DEBUG: Print the UCB score for each action.
            #     print(f"DEBUG: Action {action} -> N: {child.N}, Q: {child.Q:.4f}, Prior: {child.P:.4f}, UCB: {ucb:.4f}")

            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
        # Apply the chosen action.
        next_env = deepcopy(node.env)
        next_env.step(best_action)
        value = self.simulate(best_child)
        # Backpropagation.
        best_child.N += 1
        best_child.W += value
        best_child.Q = best_child.W / best_child.N
        node.N += 1
        return -value

    def expand(self, node: MCTSNode):
        # Instead of using the full observation:
        # state = node.env._get_obs()
        state = node.env.get_binary_obs()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # For simulations, dynamic extra features are not available—use a zero placeholder.
        # extra_features = torch.zeros(1, self.net.extra_features_dim).to(self.device)

        extra_features = node.env.get_extra_features()
        extra_features = extra_features.to(self.device)
        log_probs, value = self.net(state_tensor, extra_features)
        # Convert log probabilities to probabilities.
        policy = torch.exp(log_probs).cpu().detach().numpy()[0]
        legal_actions = node.env.get_legal_moves()
        total_prob = 0.0
        for action in legal_actions:
            idx = action_to_index(action, node.env.board_size)
            prior = policy[idx]
            total_prob += prior
            node.children[action] = MCTSNode(env=deepcopy(node.env), parent=node, action=action, prior=prior)
        # Normalize priors for children.
        if total_prob > 0:
            for child in node.children.values():
                child.P /= total_prob
        return value.item()

if __name__ == "__main__":
    # Example usage for debugging.
    from packit_env import PackItEnv  # [packit_env.py](packit_env.py)
    
    board_size = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_actions = (board_size)**4
    
    # Initialize environment, network, and MCTS.
    env = PackItEnv(board_size=board_size)
    net = AlphaZeroNet(board_size, in_channels=1, num_actions=num_actions, extra_features_dim=7)
    net.to(device)
    net.eval()
    
    #!Constant features are in the environment.

    # m_const = board_size
    # n_const = board_size
    # gap = 10  # Example value.
    # P_val = 3  # Example value.
    # K = 13     # Example value.
    
    mcts = MCTS(env, net, device=device,
                num_simulations=50)
    move_policy = mcts.get_move(temperature=1.0)
    print("Computed move probabilities:", move_policy)