import torch
import torch.nn as nn
import torch.nn.functional as F

class DominanceNN(nn.Module):
    def __init__(self, input_size):
        super(DominanceNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1) 
    
    def forward(self, v1, v2):
        diff = v1 - v2
        x = F.relu(self.fc1(diff))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class DominancePlane(nn.Module):
    def __init__(self, input_size):
        super(DominancePlane, self).__init__()
    
    def forward(self, v1, v2):
        diff = v1 - v2
        return diff


class DominanceGaussian(nn.Module):
    def __init__(self, input_size):
        super(DominanceGaussian, self).__init__()

    def forward(self, v1, v2, device = 'cpu'):
        diff = (v1 - v2).squeeze()
        positive_likelihood = torch.ones(1, requires_grad=True).to(device)
        negative_likelihood = torch.ones(1, requires_grad=True).to(device)
        for dominance_value in diff:
            normal_dist = torch.distributions.Normal(0, 1)
            cdf_value_positive = normal_dist.cdf(dominance_value)
            
            # Use PyTorch's in-place multiplication to keep gradient flow
            positive_likelihood = positive_likelihood * cdf_value_positive
            negative_likelihood = negative_likelihood * (1 - cdf_value_positive)
        return positive_likelihood - negative_likelihood


class Dominance(nn.Module):
    def __init__(self, input_size):
        super(Dominance, self).__init__()

    def forward(self, v1, v2):
        pass
