import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BassetBranched(nn.Module):
    def __init__(self, input_len=200):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(4, 300, kernel_size=19, padding=9),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(300, 200, kernel_size=11, padding=5),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(200, 200, kernel_size=7, padding=3),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        conv_output_len = input_len
        conv_output_len = conv_output_len // 3  # First pool
        conv_output_len = conv_output_len // 4  # Second pool
        conv_output_len = conv_output_len // 4  # Third pool

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200 * conv_output_len, 1000),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1000, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x