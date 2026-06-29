import torch
import torch.nn.functional as F
import numpy as np
import random
import pathlib
import csv
import torch.nn as nn

# from SiamGauss.SNN.SNN_tools.loss_functions import QuadrupletLossbatch, QuadrupletLoss
# from SiamGauss.SNN.snn import SiameseNetwork_dominance
from SiamGauss.SNN.Resources.MOO_functions import mooDataset
from SiamGauss.Benchmarks.mpbench.problems import *
from SiamGauss.Benchmarks.train import *



class conv_neural_network(nn.Module):
    def __init__(self, 
                input_size = 4 * 4,
                hidden_size_1 =  256,
                hidden_size_2 =  128,
                convD = 2):

        super(neural_network, self).__init__()
        self.input_size = input_size * 2
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        self.fc1 = nn.Linear(conv_out_size * out_channels, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc_out = nn.Linear(self.hidden_size_2, self.fc_size)


    def forward_dual(self, x):
        if len(x.shape) == 2:   # shape (batch_size, seq_len)
            x = x.unsqueeze(1)   # add channel dimension -> (batch_size, 1, seq_len)
        output = x
        output = F.relu(self.conv1(output))
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))

        for _ in range (0,self.num_repeated_hidden):
            output = F.relu(self.fc_repeat(output))
        output = F.leaky_relu(self.fc2(output), negative_slope=0.01)  
        output = self.fc_out(output)
        return output

    def forward(self, input1, input2, device = 'cpu'):
        x = torch.cat((input1, input2), dim=0)
        if len(x.shape) == 2:   # shape (batch_size, seq_len)
            x = x.unsqueeze(1)   # add channel dimension -> (batch_size, 1, seq_len)
        output = x
        output = F.relu(self.conv1(output))
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = torch.tanh(output)
        return output





if __name__ == '__main__':
    

    args = training_args(lr = 0.001)
    args.epochs = 20

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # device = torch.device('cpu')
    print('device:', device)
    

    input_sizes = [10,30,50]
    problems = []
    results = {}

    for n in input_sizes:
        problems.extend([MPMOP1(n), MPMOP2(n), MPMOP3(n), MPMOP4(n), MPMOP5(n), MPMOP6(n), MPMOP7(n), MPMOP8(n), MPMOP9(n), MPMOP10(n), MPMOP11(n)])

    for seed in range(0,5):
        np.random.seed(seed)
        for problem in problems:
            popsize = 100
            args.log_interval = popsize
            val_popsize = 100
            mooDataset_train = make_data(problem, popsize = popsize)
            mooDataset_val = make_data(problem, popsize = popsize)

            model = SiameseNetwork_dominance(input_size = problem.D,num_repeated_hidden = 1, hidden_size_1 = problem.D*8*8, hidden_size_2 =  problem.D, fc_size =  4, convD = 0).to(device)
            model_name = 'Benchmarks/Models/siamese_network_benchmark.pt'
            if args.load_model:
                try:
                    model.load_state_dict(torch.load(model_name))
                    print('model loaded')
                except:
                    # print('model could not be loaded')
                    pass

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr_start,
                weight_decay=1e-4
                )

            _ = train(args, model, device, mooDataset_train, mooDataset_val, optimizer, fast_run = True)

            mooDataset_val = make_data(problem, popsize = val_popsize)
            testing(model, device, mooDataset_train, Threshold = 0.5)
            results[str(seed) + problem.__class__.__name__ + "_" + str(problem.D)] = testing(model, device, mooDataset_val, Threshold = 0.5)

    with open("SiamGauss/Benchmarks/results_cnn.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "acc", "f1"]) 

        for name, (acc, f1) in results.items():
            writer.writerow([name, acc, f1])

    print(results)
