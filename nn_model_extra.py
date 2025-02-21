# language: python
# filepath: nn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from packit_env import PackItEnv

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size: int, in_channels: int, num_actions: int, extra_features_dim: int):
        super(AlphaZeroNet, self).__init__()
        self.extra_features_dim = extra_features_dim  # This line must be added.
        self.board_size = board_size
        self.num_actions = num_actions
        
        # Convolutional backbone.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        # Policy head.
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc1 = nn.Linear(2 * board_size * board_size + extra_features_dim, 128)
        self.policy_fc2 = nn.Linear(128, num_actions)

        # Value head.
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size + extra_features_dim, 128)
        self.value_fc2 = nn.Linear(128, 64)
        self.value_fc3 = nn.Linear(64, 1)
    
    def forward(self, x, extra_features):
        """
        x: Tensor [batch, in_channels, board_size, board_size]
        extra_features: Tensor [batch, 7]
        """
        x = self.conv(x)
        
        # Policy head.
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        # Use Mish activation
        p = F.mish(p)
        p = p.view(p.size(0), -1)  # flatten convolution features
        p = torch.cat([p, extra_features], dim=1)
        p = F.leaky_relu(self.policy_fc1(p))
        p = self.policy_fc2(p)
        log_probs = F.log_softmax(p, dim=1)

        # if random.random() < 0.01:
        #     # DEBUG: Print the policy distribution.
        #     probs = torch.exp(log_probs)
        #     print("DEBUG: Policy distribution:", probs.cpu().detach().numpy())

        # Value head.
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.mish(v)
        v = v.view(v.size(0), -1)
        v = torch.cat([v, extra_features], dim=1)
        v = F.tanh(self.value_fc1(v))
        v = F.leaky_relu(self.value_fc2(v))
        value = torch.tanh(self.value_fc3(v))
        
        return log_probs, value

if __name__ == "__main__":
    # Example usage.
    board_size = 4
    in_channels = 1
    env = PackItEnv(board_size=board_size)
    num_actions = board_size ** 4
    print("Number of actions:", num_actions)
    extra_features_dim = 7

    model = AlphaZeroNet(board_size, in_channels, num_actions, extra_features_dim)
    dummy_board = torch.randn(1, in_channels, board_size, board_size)
    dummy_extra = torch.randn(1, extra_features_dim)  # Replace with actual computed features.
    
    log_probs, value = model(dummy_board, dummy_extra)
    print("Log probs shape:", log_probs.shape)
    print("Value shape:", value.shape)