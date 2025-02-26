# PackIt! Extra: MCTS and AlphaZero-inspired Combinatorial Game

This repository contains an implementation of a combinatorial game called **PackIt!** and its corresponding reinforcement learning (RL) framework inspired by AlphaZero. The project uses Monte Carlo Tree Search (MCTS) enhanced with a neural network to guide self-play and evaluate game states.

## Project Structure

- **packit_env.py**  
  Implements the game environment (rules, legal moves, state observations, etc.).

- **nn_model_extra.py**  
  Defines the neural network architecture (AlphaZeroNet) with convolutional layers, policy/value outputs, and extra features.

- **MCTS_alphazero_extra.py**  
  Implements the MCTS algorithm along with simulation, selection, expansion and backpropagation. This module evaluates nodes via the neural network and guides the search process.

- **get_model_architecture.py**  
  Provides utilities in order to visualize the neural network architecture and generate high-level diagrams for MCTS flow, as well as run SHAP analysis on extra features.

- **train6x6_extra.py**  
  Contains training routines for the network using self-play data.  
  **Training Command:**  
  On Windows and Linux, run with:
  set CUDA_LAUNCH_BLOCKING=1 && python train6x6_extra.py

- **get_winrates.py**  
Script to simulate a large number of random games in parallel (using multiprocessing) to evaluate winrates of players.

- **play_vs_model.py**  
Implements a GUI (using Tkinter) for human vs. model play. This allows you to interactively challenge the trained model.

## Installation

1. Clone this repository:
git clone <repo_url> cd PackItExtra

2. Create and activate the conda environment (if using Anaconda):
conda create -n assignment_3_env python=3.8 conda activate assignment_3_env

3. pip install torch torchvision tensorflow numpy matplotlib tqdm shap graphviz torchsummary
