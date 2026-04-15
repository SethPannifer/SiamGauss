import torch
import torch.nn as nn
import torch.nn.functional as F

class basicNN(nn.Module):
    def __init__(self):
        super(basicNN, self).__init__()
        
        # The input dimension is 160 (two 4x4x5 matrices flattened and concatenated)
        self.fc1 = nn.Linear(160, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x1, x2, device = 'cpu'):
        # Flatten and concatenate the two inputs
        x1 = x1.reshape(1, 80)  # Flatten 4x4x5 to 80
        x2 = x2.reshape(1, 80)  # Flatten 4x4x5 to 80
        x = torch.cat((x1, x2), dim=1)  # Concatenate to get 160
        
        # Pass through hidden layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Final layer with tanh to get output in range (-1, 1)
        x = torch.tanh(self.fc4(x))
        return x
