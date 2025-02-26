# language: python
import random
from packit_env import PackItEnv
from multiprocessing import Pool
from tqdm import tqdm  # Add tqdm for progress visualization

def simulate_single_game(board_size: int):
    """
    Simulate one game by choosing random legal moves on a board_size board.
    Returns the winner: 1 if player 1 wins, 2 otherwise.
    """
    env = PackItEnv(board_size=board_size)
    env.reset()
    done = False
    while not done:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        _, reward, done, _ = env.step(move)
    # Determine winner: winner is the opponent of the last mover.
    winner = 1 if env.current_player == 2 else 2
    return winner

def simulate_random_games(board_size: int, processes: int = 10):
    """
    Simulate (env.K ** 2 * 30) games on a board_size board in parallel,
    where in each game random legal moves are chosen.
    'processes' determines how many parallel processes (games) run concurrently.
    Returns a dictionary of winrates for each side.
    """
    env = PackItEnv(board_size=board_size)
    total_games = (env.K ** 2) * 80
    # Use Pool.imap with tqdm for showing progress.
    with Pool(processes=processes) as pool:
        winners = list(tqdm(
            pool.imap(simulate_single_game, [board_size] * total_games),
            total=total_games,
            desc=f"Simulating {total_games} games on {board_size}x{board_size} board"
        ))
    wins = {1: winners.count(1), 2: winners.count(2)}
    winrates = {player: wins[player] / total_games for player in wins}
    print(f"After {total_games} random games on a {board_size}x{board_size} board:")
    print(f"Winrate Player 1: {winrates[1]:.3f}, Winrate Player 2: {winrates[2]:.3f}")
    return winrates

if __name__ == "__main__":
    # Set the number of parallel processes for simulation here.
    num_processes = 10
    # Simulate random games on boards of different sizes.
    for i in range(1, 9):
        winrates = simulate_random_games(i, processes=num_processes)
        with open("winrates.txt", "a") as f:
            f.write(f"{i}x{i} Board: {winrates}\n")