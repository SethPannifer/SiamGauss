import torch
import torch.nn.functional as F
import numpy as np
import random
import pathlib
import csv

from nats_bench import create

from SiamGauss.SNN.SNN_tools.loss_functions import QuadrupletLossbatch, QuadrupletLoss
from SiamGauss.SNN.snn import SiameseNetwork_dominance
from SiamGauss.SNN.Resources.MOO_functions import mooDataset
from SiamGauss.Benchmarks.mpbench.problems import *
from SiamGauss.Benchmarks.train import *

np.random.seed(1)


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
    device = torch.device('cpu')
    print('device:', device)
    

    input_sizes = [10,30,50]
    problems = []
    results = {}

    for n in input_sizes:
        problems.extend([MPMOP1(n), MPMOP2(n), MPMOP3(n), MPMOP4(n), MPMOP5(n), MPMOP6(n), MPMOP7(n), MPMOP8(n), MPMOP9(n), MPMOP10(n), MPMOP11(n)])


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
        results[problem.__class__.__name__ + "_" + str(problem.D)] = testing(model, device, mooDataset_val, Threshold = 0.5)

    with open("Benchmarks/results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "acc", "f1"]) 

        for name, (acc, f1) in results.items():
            writer.writerow([name, acc, f1])

    print(results)
    