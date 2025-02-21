import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sympy import primerange
import math
import torch


class PackItEnv(gym.Env):
    """
    Custom Environment for the PackIt! game.
    The board is an N x N grid.
    On turn i (starting from 1), a player can place a rectangular tile
    of area i or i+1. A move is represented as (row, col, width, height).
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, board_size=8):
        super(PackItEnv, self).__init__()
        self.board_size = board_size
        self.m = board_size
        self.n = board_size

        # Define the observation space:
        # We'll use a board_size x board_size grid with values 0 (empty), 1, or 2.
        self.observation_space = spaces.Box(low=0, high=2, 
                                            shape=(self.board_size, self.board_size), dtype=np.int8)
        
        # Define the action space.
        # Since the legal actions vary based on the board state and current turn,
        # we define a generic action as a tuple: (row, col, width, height)
        # For simplicity, we set maximum values for row, col, width, and height.
        # In practice, you'll generate legal moves dynamically.
        self.action_space = spaces.MultiDiscrete([self.board_size, self.board_size,
                                            self.board_size, self.board_size])
        
        self.K = int((math.sqrt(8 * self.m * self.n + 1) - 1) / 2)
        self.gap = self.m * self.n - self.K * (self.K + 1) // 2
        self.P_val = len(list(primerange(self.n + 1, self.K + 1)))
        # Initialize game state variables
        self.reset()

    def get_extra_features(self):
        """
        Assemble a 7-dimensional feature vector.
        Constant features: m, n, gap, P_val.
        Dynamic features (per turn):
        exp_norm = expansion_turns / gap
        left_dist = (P_val - expansion_turns) / (K - P_val)
        right_dist = 1 - expansion_turns / (K - P_val)
        Handles division-by-zero by returning 0.
        Returns a tensor of shape [1, 7].
        """
        expan_norm = self.expansion_turns / self.gap if self.gap != 0 else self.expansion_turns
        # denom = (self.K - self.P_val) if (self.K - self.P_val) != 0 else 1.0
        denom = self.gap if self.gap != 0 else 1.0
        left_dist = (self.P_val - self.expansion_turns) / denom
        # right_dist = 1 - (self.expansion_turns / denom)
        right_dist = (self.K - self.P_val - self.expansion_turns) / denom
        features = torch.tensor([self.m, self.n, self.gap, self.P_val, expan_norm, left_dist, right_dist], dtype=torch.float32)
        return features.unsqueeze(0)
    
    def reset(self):
        """Reset the game state and return the initial observation."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.turn = 1  # Game starts on turn 1
        self.current_player = 1  # Player 1 starts
        self.expansion_turns = 0
        return self._get_obs()
    
    
    def _get_obs(self):
        """Return the current observation (board state)."""
        return self.board.copy()
    
    def get_binary_obs(self):
        """
        Returns a binary representation of the board.
        0 indicates a free cell, and 1 indicates an occupied cell.
        """
        obs = self._get_obs()  # original board (with 0, 1, and 2)
        return (obs != 0).astype(obs.dtype)
    
    def _tile_area(self, width, height):
        return width * height
    
    def _is_valid_move(self, row, col, width, height):
        """Check if placing a tile at (row, col) with dimensions width x height is valid."""
        # Check if the tile fits on the board
        if row < 0 or col < 0 or row + height > self.board_size or col + width > self.board_size:
            return False
        
        # Check if the tile overlaps existing tiles
        sub_board = self.board[row:row+height, col:col+width]
        if np.any(sub_board != 0):
            return False
        
        # Check if the tile area matches allowed area for this turn (i or i+1)
        area = self._tile_area(width, height)
        if area not in [self.turn, self.turn + 1]:
            return False
        
        return True

    

    def get_legal_moves(self):
        """Generate a list of all legal moves in the current state.
           Each move is represented as (row, col, width, height).
        """
        legal_moves = []
        # Consider all possible starting positions on the board.
        for row in range(self.board_size):
            for col in range(self.board_size):
                # Skip if the starting cell is already occupied.
                if self.board[row, col] != 0:
                    continue
                # Try all possible tile dimensions.
                for width in range(1, self.board_size - col + 1):
                    for height in range(1, self.board_size - row + 1):
                        if self._is_valid_move(row, col, width, height):
                            legal_moves.append((row, col, width, height))
        return legal_moves
    
    def step(self, action):
        """
        Execute one move in the game.
        Action should be a tuple: (row, col, width, height).
        Returns: observation, reward, done, info
        """
        row, col, width, height = action

        # Expansion turn counter update
        area = self._tile_area(width, height)
        if area == self.turn + 1:
            self.expansion_turns += 1
        
        if not self._is_valid_move(row, col, width, height):
            # Invalid move: end the game with a negative reward.
            return self._get_obs(), -1000, True, {"error": "Invalid move"}
        
        # Place the tile on the board.
        self.board[row:row+height, col:col+width] = self.current_player
        
        # Check for terminal condition: if no legal moves remain for the next turn.
        self.turn += 1
        self.current_player = 2 if self.current_player == 1 else 1
        legal_moves = self.get_legal_moves()
        done = len(legal_moves) == 0
        
        # Reward scheme:
        # - If the game ends and the current player (who just moved) wins, reward positively.
        # - Otherwise, give a small step reward.
        reward = 0.01
        if done:
            # When no moves remain, the player who just played wins.
            reward = 10
        
        
        return self._get_obs(), reward, done, {"legal_moves": legal_moves}
    
    def render(self, mode='human'):
        """Simple console render of the board."""
        print("Turn:", self.turn, "Current Player:", self.current_player)
        print(self.board)
        
if __name__ == "__main__":
    # Example test loop to interact with the environment.
    env = PackItEnv(board_size=6)
    obs = env.reset()
    env.render()
    
    done = False
    while not done:
        legal_moves = env.get_legal_moves()
        print("Legal moves:", legal_moves)
        if not legal_moves:
            print("No legal moves left!")
            break
        
        # For demonstration, select a random legal move.
        action = legal_moves[np.random.randint(len(legal_moves))]
        print("Taking action:", action)
        obs, reward, done, info = env.step(action)
        print(f"amount of moves for perfect pack: {env.K}")
        print(f"gap", env.gap)
        print(f'Extra features: {env.get_extra_features()}')
        env.render()
        
    print("Game over!")
