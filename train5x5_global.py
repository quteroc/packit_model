import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from multiprocessing import Pool
import random
import math
from nn_model_extra import AlphaZeroNet  # see [nn_model_extra.py]
from packit_env import PackItEnv       # see [packit_env.py]
from MCTS_alphazero_extra import MCTS, action_to_index, index_to_action  # see [MCTS_alphazero_extra.py]

#### CHANGED: Added global shared_net and worker initializer for sharing the network in workers.
shared_net = None
def worker_init(state_dict, board_size, device):
    global shared_net
    shared_net = AlphaZeroNet(board_size, in_channels=1, num_actions=board_size**4, extra_features_dim=7)
    shared_net.load_state_dict(state_dict)
    shared_net.to(device)
    shared_net.train()
#### END CHANGED.

def self_play_game_parallel(args):
    """
    Run a single self-play game for training on a 6x6 board.
    Returns training_data, final board observation, and winner.
    """
    board_size, device, reward_multipliers, _ = args  # state_dict no longer needed here
    env = PackItEnv(board_size=board_size)
    num_actions = board_size ** 4
    #### CHANGED: Use global shared_net instead of creating new instance.
    global shared_net
    net = shared_net
    net.to(device)
    net.train()
    #### END CHANGED.
    state_history = []  # each record: (state, extra, mcts_policy, player)
    done = False
    while not done:
        mcts = MCTS(env, net, device=device, num_simulations=128)
        mcts_policy = mcts.get_move(temperature=1.1)
        # Sample a move according to the probabilities.
        moves, probs = zip(*mcts_policy.items())
        best_move = random.choices(moves, weights=probs, k=1)[0]
        state = env.get_binary_obs()  # e.g. a 6x6 board
        extra = env.get_extra_features() if hasattr(env, "get_extra_features") else torch.zeros(7, dtype=torch.float32)
        state_history.append((state, extra, mcts_policy, env.current_player))
        _, reward, done, _ = env.step(best_move)
    
    terminal_reward = 1
    # Determine winner: assume winner is the opponent of the last mover.
    winner = 1 if env.current_player == 2 else 2

    training_data = []
    for state, extra, policy, player in state_history:
        if player == winner:
            outcome = terminal_reward * reward_multipliers.get(player, 1.0)
        else:
            enemy = 1 if player == 2 else 2
            outcome = -terminal_reward * reward_multipliers.get(enemy, 1.0)
        training_data.append((state, extra, policy, outcome))
        
    return training_data, env.get_binary_obs(), winner

def simulate_model_game(net, board_size, device='cpu'):
    """
    Simulate a single game in evaluation mode on a board_size board.
    Returns the winner (1 or 2).
    """
    # board_size = 6
    env = PackItEnv(board_size=board_size)
    net.eval()  # disable exploration for deterministic play
    done = False
    while not done:
        mcts = MCTS(env, net, device=device, num_simulations=128)
        mcts_policy = mcts.get_move(temperature=1)
        moves, probs = zip(*mcts_policy.items())
        best_move = random.choices(moves, weights=probs, k=1)[0]
        _, reward, done, _ = env.step(best_move)
    winner = 1 if env.current_player == 2 else 2
    return winner

def eval_game(args):
    """
    Helper for evaluation: create a new net from the passed state and
    simulate one game.
    NOTE: For evaluation, we force computation on CPU.
    """
    board_size, _, state_dict = args  # ignore passed device, use CPU
    device_eval = 'cpu'
    num_actions = board_size**4
    net = AlphaZeroNet(board_size, in_channels=1, num_actions=num_actions, extra_features_dim=7)
    net.load_state_dict(state_dict)
    net.to(device_eval)
    return simulate_model_game(net, board_size, device_eval)

def evaluate_model(net, device, eval_games, board_size=6, parallel_processes=None):
    """
    Run evaluation games in parallel (on CPU) and print win counts.
    """
    state_dict = net.state_dict()
    args = [(board_size, device, state_dict)] * eval_games
    wins = {1: 0, 2: 0}
    processes = parallel_processes if parallel_processes is not None else eval_games
    with Pool(processes=processes) as pool:
        results = pool.map(eval_game, args)
    for winner in results:
        wins[winner] += 1
    print(f"Evaluation over {eval_games} games: {wins}")
    return wins

def rand_eval_game(args):
    """
    Helper for random evaluation.
    args = (board_size, device, state_dict, mode)
    mode: "model_first" means model is player 1,
          "model_second" means model is player 2.
    Returns 1 if the model wins, else 0.
    """
    board_size, _, state_dict, mode = args
    device_eval = 'cpu'
    target_player = 1 if mode == "model_first" else 2
    num_actions = board_size**4
    net = AlphaZeroNet(board_size, in_channels=1, num_actions=num_actions, extra_features_dim=7)
    net.load_state_dict(state_dict)
    net.to(device_eval)
    env = PackItEnv(board_size=board_size)
    done = False
    while not done:
        legal_moves = env.get_legal_moves()
        if env.current_player == target_player:
            net.eval()
            mcts = MCTS(env, net, device=device_eval, num_simulations=128)
            mcts_policy = mcts.get_move(temperature=1)
            moves, probs = zip(*mcts_policy.items())
            move = random.choices(moves, weights=probs, k=1)[0]
        else:
            move = random.choice(legal_moves)
        _, reward, done, _ = env.step(move)
    winner = 1 if env.current_player == 2 else 2
    return 1 if winner == target_player else 0

def rand_evaluate_model(net, device, eval_games, num_first, board_size, parallel_processes=None):
    """
    Run random evaluation games in parallel.
    Half the games: model plays as first mover (its opponent makes random moves),
    and the other half: model plays as second mover (first mover random).
    Returns a dict with the model's win counts in each mode.
    """
    state_dict = net.state_dict()
    wins = {"model_first": 0, "model_second": 0}
    num_second = eval_games - num_first
    args_first = [(board_size, device, state_dict, "model_first")] * num_first
    args_second = [(board_size, device, state_dict, "model_second")] * num_second
    all_args = args_first + args_second
    processes = parallel_processes if parallel_processes is not None else eval_games
    with Pool(processes=processes) as pool:
        results = pool.map(rand_eval_game, all_args)
    wins["model_first"] = sum(results[:num_first])
    wins["model_second"] = sum(results[num_first:])
    print(f"Random Evaluation over {eval_games} games: {wins}")
    return wins

if __name__ == "__main__":
    total_games = 3000
    board_size = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_actions = board_size**4

    net = AlphaZeroNet(board_size, in_channels=1, num_actions=num_actions, extra_features_dim=7)
    #!Load weights from previous training
    # net.load_state_dict(torch.load("alphazero5x5.pth"))
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    print("Testing evaluation function on CPU...")
    test_wins = evaluate_model(net, device, eval_games=6, board_size=board_size, parallel_processes=3)
    print(f"Test evaluation wins: {test_wins}")

    # Initialize adaptive reward multipliers.
    adaptive_reward = {1: 1.0, 2: 1.0}
    loss_stats = []
    winner_stats = []
    parallel_games = 10  # number of parallel self-play games per training batch

    for game in range(0, total_games, parallel_games):
        # Get the current state dict.
        current_state_dict = net.state_dict()
        #### CHANGED: Use pool initializer to set up shared_net once per training batch.
        with Pool(processes=parallel_games, initializer=worker_init, initargs=(current_state_dict, board_size, device)) as pool:
            args_list = [(board_size, device, adaptive_reward, current_state_dict)] * parallel_games
            results = pool.map(self_play_game_parallel, args_list)
        #### END CHANGED.

        block_loss = 0.0
        #NEW
        # total_loss = 0.0 
        # sample_count = 0
        ### CHANGED: Use optimizer.zero_grad() and optimizer.step() outside the loop.
        optimizer.zero_grad()  # accumulate gradients over the batch
        for training_data, final_board, winner in results:
            winner_stats.append(winner)
            loss = None
            for state, extra, mcts_policy, outcome in training_data:
                # sample_count += 1
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                if not torch.is_tensor(extra):
                    extra = torch.tensor(extra, dtype=torch.float32)
                extra = extra.to(device)
                if extra.dim() == 1:
                    extra = extra.unsqueeze(0)
                log_probs, value = net(state_tensor, extra)
                
                target_policy = torch.zeros(num_actions, dtype=torch.float32).to(device)
                for action, prob in mcts_policy.items():
                    idx = action_to_index(action, board_size)
                    target_policy[idx] = prob
                policy_loss = -torch.sum(target_policy * log_probs[0])
                value_loss = (value[0, 0] - outcome) ** 2
                if loss == None:
                    loss = policy_loss + value_loss
                else:
                    loss += policy_loss + value_loss
                #OLD
                # loss.backward()
                #NEW
                # total_loss += loss
                # block_loss += loss.item()
            loss.backward()  # apply gradients for the game
            block_loss += loss.item()
        #NEW
        # avg_loss = total_loss / sample_count      #### CHANGED: Average over all samples
        # avg_loss.backward()                         #### CHANGED: Call backward once on the average loss

        optimizer.step()  # update once per batch

        if game % 20 == 0:
            avg_loss = block_loss / parallel_games if parallel_games > 0 else 0
            # avg_loss = avg_loss.item()
            loss_stats.append({"game": game, "avg_loss": avg_loss})
            if game in {0, 20, 100, 200, 300, 400, 500, 700, 1000, 1300, 1500, 2000, 2300, 2500, 3000}:
                print(f"Game: {game}, Average Loss: {avg_loss}")

        # Update adaptive reward multipliers every 100 games.
        if len(winner_stats) >= 100 and len(winner_stats) % 100 == 0:
            last_100 = winner_stats[-100:]
            wins_p1 = last_100.count(1)
            wins_p2 = last_100.count(2)
            win_rate_p1 = wins_p1 / 100
            win_rate_p2 = wins_p2 / 100
            # adaptive_reward = {
            #     1: 1.0 / math.sqrt(win_rate_p1) if win_rate_p1 > 0 else 50,
            #     2: 1.0 / math.sqrt(win_rate_p2) if win_rate_p2 > 0 else 50
            # }
            adaptive_reward = {
                1: 1.0,
                2: 1.0,
            }
            print(f"Adaptive multipliers updated (last 100 games): {adaptive_reward}")
            print(f"Wins Player 1: {wins_p1}, | Player 2: {wins_p2}")

        # Every 300 training games, run evaluation in parallel on CPU.
        if game > 0 and game % 300 == 0:
            eval_games = 6 * parallel_games  # evaluation games count
            num_first = 2 * parallel_games
            num_second = eval_games - num_first
            print(f"\n--- Evaluation at {game} training games ---")
            wins = rand_evaluate_model(net, device, eval_games, num_first, board_size=board_size, parallel_processes=parallel_games)
            total_wins = wins["model_first"] + wins["model_second"]
            print(f"Random Evaluation Results: model_first wins: {wins['model_first']}/{num_first}, " +
                  f"model_second wins: {wins['model_second']}/{num_second}, " +
                  f"total wins: {total_wins}/{eval_games}")
            if wins["model_first"] == num_first or wins["model_second"] == num_second:
                print(f"Breaking training early due to homogeneous evaluation results: {wins}")
                break

    with open("training_stats.json", "w") as f:
        json.dump({"loss_stats": loss_stats, "winner_stats": winner_stats}, f)
    torch.save(net.state_dict(), "alphazero5x5.pth")