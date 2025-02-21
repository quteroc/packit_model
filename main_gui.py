import tkinter as tk
from tkinter import messagebox
from packit_env import PackItEnv
import tkinter.simpledialog as simpledialog

class PackItGUI:
    def __init__(self, master, Board_Size):
        self.master = master
        self.master.title("PackIt! Game")
        
        self.board_size = Board_Size  # Set the board size
        self.env = PackItEnv(board_size=self.board_size)
        self.canvas = tk.Canvas(master, width=400, height=400)
        self.canvas.pack()
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.move_counter = 0  # Counter for the moves
        self.moves = []        # List to store executed moves (each as a tuple)
        self.first_click = None
        self.reset_game()

    def reset_game(self):
        self.env.reset()
        self.move_counter = 0
        self.moves = []
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        cell_size = 400 // self.board_size
        
        # Draw the board cells.
        for row in range(self.board_size):
            for col in range(self.board_size):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                # Determine fill color based on env.board value.
                if self.env.board[row, col] == 0:
                    color = "white"
                elif self.env.board[row, col] == 1:
                    color = "#8A2BE2"  # Retro blue
                else:
                    color = "#FF6347"  # Retro red
                # Use a thicker outline for more visible borders.
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black", width=2)
        
        # Draw a thin outline for each executed move.
        for action in self.moves:
            start_row, start_col, width, height = action
            x1 = start_col * cell_size
            y1 = start_row * cell_size
            x2 = (start_col + width) * cell_size
            y2 = (start_row + height) * cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray", width=1)
        
        # Display the move counter in the top left corner.
        # Showing move number as move_counter+1 so that it's 1-indexed for display.
        self.canvas.create_text(5, 5, anchor="nw", text=f"Move: {self.move_counter + 1}", fill="black")
        self.master.update()


    def on_click(self, event):
        cell_size = 400 // self.board_size
        col = event.x // cell_size
        row = event.y // cell_size

        if self.first_click is None:
            # First click: clear any previous temporary highlights and record the corner.
            self.canvas.delete("temp")
            self.first_click = (row, col)
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=3, tag="temp")
            return
        else:
            # Second click: clear the temporary highlight.
            self.canvas.delete("temp")
            first_row, first_col = self.first_click
            second_row, second_col = (row, col)
            self.first_click = None
            
            # Compute move.
            start_row = min(first_row, second_row)
            start_col = min(first_col, second_col)
            height = abs(first_row - second_row) + 1
            width = abs(first_col - second_col) + 1
            action = (start_row, start_col, width, height)
            
            # Check if move is legal.
            legal_moves = self.env.get_legal_moves()
            if action not in legal_moves:
                messagebox.showerror("Invalid Move", f"The move {action} is not legal!")
                # Reset so user can choose a new first corner.
                self.first_click = None
                return
            
            # Execute the move.
            try:
                obs, reward, done, info = self.env.step(action)
            except ValueError as e:
                messagebox.showerror("Move Error", str(e))
                return

            # Record the move and update counter.
            self.moves.append(action)
            self.move_counter += 1

            self.draw_board()
            
            if done:
                # Determine winner based on move counter.
                # When done is reached, the current turn has no legal moves.
                # So the player who just moved wins.
                if self.move_counter % 2 == 1:
                    winner = "Blue (First player) wins!"
                else:
                    winner = "Red (Second player) wins!"
                messagebox.showinfo("Game Over", f"No legal moves left for the opponent!\n{winner}")
                self.reset_game()

if __name__ == "__main__":
    root = tk.Tk()
    app = PackItGUI(root)
    root.mainloop()