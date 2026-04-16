from nats_bench import create
import torch
import random
from SiamGauss.SNN.Resources.NATS_prep import NATS_neuralNetwork, str2matrix, matrix2str
from SiamGauss.SNN.snn import SiameseNetwork_dominance
from SiamGauss.SNN.Resources.MOO_functions import mooDataset
from SiamGauss.paths import model_name, api, dataset
from SiamGauss.train import train
from SiamGauss.paths import model_name, api, dataset

import pathlib
path_dir = str(pathlib.Path().resolve())

              
class training_args():
    def __init__ (self,batch_size = 64,test_batch_size = 1000,epochs = 20,lr = 0.001,gamma =0.7,
                no_cuda=False,no_mps=False,dry_run=False,seed=1,log_interval=100,load_model = True, save_model=True):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.no_cuda = no_cuda
        self.no_mps = no_mps
        self.dry_run = dry_run
        self.seed = seed
        self.log_interval = log_interval
        self.load_model = load_model
        self.save_model = save_model
args = training_args()


def fine_tune(data_points_idx,model_name, api = api):
    args = training_args()
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

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    intial_dataset_size = 10
    dataset_matrix = [[],[]]
    for i in (data_points_idx):
        
        str_arch = api.arch(i)
        input = str2matrix(str_arch)
        network = NATS_neuralNetwork(input)
        if network.check_exists() == '200':
            network.simulate_train()
            objectives = [network.one_over_accuracy,network.flops]
            dataset_matrix[0].append(network.encoded_matrix)
            dataset_matrix[1].append(objectives)

        
    mooDataset_train = mooDataset(dataset_matrix[0], dataset_matrix[1])

    model = SiameseNetwork_dominance()
    # args.load_model = False
    if args.load_model:
        try:
            model.load_state_dict(torch.load(model_name))
            print('model loaded')
        except:
            pass
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, mooDataset_train, optimizer, epoch)
        # test(model, device, test_loader)
    if args.save_model:
        torch.save(model.state_dict(), model_name)
        print('model saved')

if __name__ == '__main__':   
    data_points_idx = [i for i in range(0,10)]
    fine_tune(data_points_idx, model_name)

