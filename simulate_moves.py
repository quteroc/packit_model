# language: python

import random
from packit_env import PackItEnv  # see [packit_env.py](packit_env.py)

def print_board(board):
    for row in board:
        print(" ".join(str(cell) for cell in row))
    print("-" * 20)

# Initialize environment with a 6x6 board.
board_size = 6
env = PackItEnv(board_size=board_size)
env.reset()

print("Initial board state:")
print_board(env._get_obs())

# Loop until no legal moves remain.
move_number = 1
while True:
    legal_moves = env.get_legal_moves()
    if not legal_moves:
        print("No legal moves left. Game over!")
        break
    # For this demonstration, choose a random legal move.
    move = random.choice(legal_moves)
    print(f"Move {move_number}: {move}")
    player_obs, reward, done, _ = env.step(move)
    obs = env.get_binary_obs()
    print(f"Board state after move {move_number}:")
    print("Player observation:")
    print_board(player_obs)
    print('Binary observation:')
    print_board(obs)
    move_number += 1
    if done:
        print("Game ended with reward:", reward)
        break