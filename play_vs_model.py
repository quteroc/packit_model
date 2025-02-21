# language: python
# filepath: /c:/Users/butel/General_PackIt!_Extra/play_vs_model.py

import tkinter as tk
import torch
from tkinter import messagebox
from main_gui import PackItGUI  # see [main_gui.py](main_gui.py)
from packit_env import PackItEnv  # see [packit_env.py](packit_env.py)
from MCTS_alphazero_extra import MCTS  # see [MCTS_alphazero_extra.py](MCTS_alphazero_extra.py)
from nn_model_extra import AlphaZeroNet  # see [nn_model_extra.py](nn_model_extra.py)


class PackItGUIvsModel(PackItGUI):
    def __init__(self, master, board_size=6):
        self.board_size = board_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the trained model.
        num_actions = board_size**4
        self.model_net = AlphaZeroNet(board_size, in_channels=1, num_actions=num_actions, extra_features_dim=7)
        try:
            self.model_net.load_state_dict(torch.load("alphazero6x6.pth", map_location=self.device))
        except Exception as e:
            print(f"Error loading model: {e}")
        self.model_net.to(self.device)
        self.model_net.eval()
        # Initialize environment with board_size = 5.
        self.env = PackItEnv(board_size=board_size)
        # Initialize GUI variables.
        self.moves = []
        self.move_counter = 0
        self.first_click = None
        # We'll set human_player based on user's choice (1 or 2)
        self.human_player = None
        super().__init__(master, Board_Size=board_size)
        self.show_role_selection()

    def show_role_selection(self):
        # Create a frame overlay to choose role.
        self.role_frame = tk.Frame(self.master)
        tk.Label(self.role_frame, text="Choose Your Role:", font=("Arial", 14)).pack(pady=10)
        tk.Button(self.role_frame, text="Play as First", font=("Arial", 12),
                  command=lambda: self.set_human_role(1)).pack(pady=5)
        tk.Button(self.role_frame, text="Play as Second", font=("Arial", 12),
                  command=lambda: self.set_human_role(2)).pack(pady=5)
        self.role_frame.place(relx=0.5, rely=0.5, anchor="center")

    def set_human_role(self, role):
        self.human_player = role
        self.role_frame.destroy()
        self.draw_board()
        # If human plays second, let the model make the first move.
        if self.env.current_player != self.human_player:
            self.after_human_move()

    def on_human_move(self, action):
        # Execute the human move only when it's their turn.
        if self.env.current_player != self.human_player:
            print("Not your turn!")
            return
        try:
            obs, reward, done, info = self.env.step(action)
        except Exception as e:
            print(f"Error during human move: {e}")
            return
        self.moves.append(action)
        self.move_counter += 1
        self.draw_board()
        if done:
            messagebox.showinfo("Game Over", "Game over after your move!")
            return
        self.after_human_move()

    def after_human_move(self):
        # Let the model compute its move if it's not the human's turn.
        if self.env.current_player == self.human_player:
            return  # It is human's turn.
        mcts = MCTS(self.env, self.model_net, device=self.device)
        mcts_policy = mcts.get_move(temperature=1)
        best_move = max(mcts_policy.items(), key=lambda x: x[1])[0]
        try:
            obs, reward, done, info = self.env.step(best_move)
        except Exception as e:
            print(f"Error during model move: {e}")
            return
        self.moves.append(best_move)
        self.move_counter += 1
        self.draw_board()
        if done:
            messagebox.showinfo("Game Over", "Game over after model move!")

    def on_click(self, event):
        # Do not react to clicks until the user chooses a role.
        if self.human_player is None:
            return
        # Only accept clicks when it's the human's turn.
        if self.env.current_player != self.human_player:
            print("Wait for the model to move!")
            return
        
        cell_size = 400 // self.board_size
        col = event.x // cell_size
        row = event.y // cell_size

        if self.first_click is None:
            self.canvas.delete("temp")
            self.first_click = (row, col)
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=3, tag="temp")
        else:
            self.canvas.delete("temp")
            first_row, first_col = self.first_click
            second_row, second_col = row, col
            self.first_click = None
            
            start_row = min(first_row, second_row)
            start_col = min(first_col, second_col)
            width = abs(first_col - second_col) + 1
            height = abs(first_row - second_row) + 1
            action = (start_row, start_col, width, height)
            # Validate move.
            legal_moves = self.env.get_legal_moves()
            if action not in legal_moves:
                print(f"Illegal move: {action}")
                messagebox.showerror("Invalid Move", f"The move {action} is not legal!")
                return
            self.on_human_move(action)

if __name__ == "__main__":
    root = tk.Tk()
    # Ask the user for the board size via a simple dialog.
    board_size_str = tk.simpledialog.askstring("Board Size", "Enter board size (e.g., 5 or 6):")
    try:
        board_size = int(board_size_str)
    except (ValueError, TypeError):
        board_size = 6  # default value if user input is invalid
    gui = PackItGUIvsModel(root, board_size=board_size)
    root.mainloop()