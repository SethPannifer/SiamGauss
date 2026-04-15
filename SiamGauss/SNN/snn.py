import torch
import torch.nn as nn
import torch.nn.functional as F

from SNN.SNN_tools.comparison_functions import DominanceNN, DominancePlane, DominanceGaussian

class SetConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=16, pool="sum"):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.rho = nn.Linear(hidden_dim, out_channels)
        self.pool = pool

    def forward(self, x):
        # x shape: [batch, channels, set_size]

        x = x.permute(0, 2, 1)      # [batch, set_size, channels]
        x = self.phi(x)             # apply φ

        if self.pool == "sum":
            x = x.sum(dim=1)
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "max":
            x = torch.max(x, dim=1).values

        x = self.rho(x)             # apply ρ

        return x.unsqueeze(-1)      # return shape like Conv1D: [batch, out_channels, 1]


class SiameseNetwork_dominance(nn.Module):
    def __init__(self, 
                input_size = 4 * 4,
                hidden_size_1 =  256,
                num_repeated_hidden = 1,
                hidden_size_2 =  128,
                fc_size =  6,
                convD = 2):

        super(SiameseNetwork_dominance, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_repeated_hidden = num_repeated_hidden
        self.fc_size =fc_size

        # dual model layers
        if convD == -1:
            # permutation invariant "set convolution"
            in_channels = 1
            out_channels = 8
            self.conv1 = SetConv1D(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_dim=16,
                pool="sum")
            conv_out_size = 1
            self.fc1 = nn.Linear(conv_out_size * out_channels, self.hidden_size_1)

        elif convD == 0:
            self.conv1 =  nn.Identity()
            self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)

        elif convD == 1:
            padding = 1
            kernel_size = 3
            stride = 1
            dilation = 1
            in_channels = 1
            out_channels = 8 
            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=kernel_size, padding=padding)
            conv_out_size = int(((self.input_size + 2*padding - (dilation*(kernel_size - 1)) - 1)/stride) + 1)
            self.fc1 = nn.Linear(conv_out_size * out_channels, self.hidden_size_1)

        else:
            self.conv1 = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(8 * self.input_size, self.hidden_size_1)

        self.fc_repeat = nn.Linear(self.hidden_size_1, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc_out = nn.Linear(self.hidden_size_2, self.fc_size)

        self.featurecompare = DominanceGaussian(self.fc_size)

    #     self.sigmoid = nn.Sigmoid()
    #     self.featurecompare.apply(self.init_weights)
        
        # self.apply(self.init_weights)

    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         m.bias.data.fill_(0.1)


    def forward_dual(self, x):
        if len(x.shape) == 2:   # shape (batch_size, seq_len)
            x = x.unsqueeze(1)   # add channel dimension -> (batch_size, 1, seq_len)
        output = x
        output = F.relu(self.conv1(output))
        # output = F.dropout(output, p=0.3) 
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))

        for _ in range (0,self.num_repeated_hidden):
            output = F.relu(self.fc_repeat(output))
        output = F.leaky_relu(self.fc2(output), negative_slope=0.01)  
        output = self.fc_out(output)
        return output

    def forward(self, input1, input2, device = 'cpu'):
        output1 = self.forward_dual(input1)
        output2 = self.forward_dual(input2)
        output = self.featurecompare(output1,output2)
        return output



