
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nats_bench import create
from SNN.snn import SiameseNetwork_dominance
from SNN.basic_NN import basicNN
from SNN.SNN_tools.loss_functions import QuadrupletLossbatch, QuadrupletLoss
from SNN.Resources.NATS_prep import NATS_neuralNetwork, str2matrix
from SNN.Resources.MOO_functions import domination_check, mooDataset
from testing import Run_test
import csv
from itertools import product

import pathlib
path_dir = str(pathlib.Path().resolve())

def train(args, model, device, dataset, optimizer, epoch, lossfunc = 'quadlossbatch'):
    model.train()
    if lossfunc == 'quadlossbatch':
        criterion = QuadrupletLossbatch().to(device)
        for i in range (0,dataset.__len__()):

            anchor,positive,neutral,negative  =dataset.__getitem__(i)
            anchor = anchor.to(device)
            positive = [p.to(device) for p in positive]
            neutral = [n.to(device) for n in neutral]
            negative = [n.to(device) for n in negative]
            output_positive = [model.forward(anchor, positive[j]).squeeze() for j in range (0,len(positive))]
            output_neutral = [model.forward(anchor, neutral[j]).squeeze() for j in range (0,len(neutral))]
            output_negative = [model.forward(anchor, negative[j]).squeeze() for j in range (0,len(negative))]
            loss = criterion(output_positive, output_neutral,output_negative).squeeze()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i , (dataset.__len__()),
                    100. * i / (dataset.__len__()), loss.item()))
                if args.dry_run:
                    break
        

    elif lossfunc == 'quadloss':
        
        criterion = QuadrupletLoss().to(device)
        running_loss = 0
        for i in range (0,dataset.__len__()):
            anchor,positive,neutral,negative = dataset.__getitem__(i)
            anchor = anchor.to(device)
            positive = [p.to(device) for p in positive]
            neutral = [n.to(device) for n in neutral]
            negative = [n.to(device) for n in negative]

            len_positive = len(positive)
            len_neutral = len(neutral)
            len_negative = len(negative)
            max_class_size = max(len_positive ,len_neutral ,len_negative)
            anchor_loss = 0
            for j in range (0,max_class_size):
                if len_positive == 0:
                    output_positive = 1
                else:
                    output_positive = model.forward(anchor, positive[j%len_positive], device = device).squeeze()

                if len_neutral == 0:
                    len_neutral = 0
                else:
                    output_neutral = model.forward(anchor, neutral[j%len_neutral], device = device).squeeze()

                if len_negative == 0:
                    output_negative = -1
                else:
                    output_negative = model.forward(anchor, negative[j%len_negative], device = device).squeeze()

                loss = criterion(output_positive, output_neutral, output_negative)
                anchor_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            running_loss += anchor_loss

            if i % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i , (dataset.__len__()),
                    100. * i / (dataset.__len__()), loss.item()))
                if args.dry_run:
                    break
    return(loss.item())

class training_args():
        def __init__ (self,batch_size = 64,test_batch_size = 1000,epochs = 14,lr = 0.0001,gamma =0.7,
                    no_cuda=False,no_mps=True,dry_run=False,seed=1,log_interval=10,load_model = False, save_model=False, plot = False):
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
            self.plot = plot


def make_dataset(intial_dataset_size):
    api = create(path_dir +'/SNN/Resources/NATS_DATASETS/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
    dataset_matrix = [[],[]]
    for i in range (0,intial_dataset_size):
        str_arch = api.arch(i)
        input = str2matrix(str_arch)
        network = NATS_neuralNetwork(input)
        if network.check_exists() == '200':
            network.simulate_train()
            objectives = [network.one_over_accuracy,network.flops]
            dataset_matrix[0].append(network.encoded_matrix)
            dataset_matrix[1].append(objectives)
            network_found = True 
    return mooDataset(dataset_matrix[0], dataset_matrix[1])

if __name__ == '__main__': 

    def train_run(args, input_size = None, hidden_size_1 = None, num_repeated_hidden = None, hidden_size_2 = None, fc_size = None, best_f1 = 0, save_best = True   ):               

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        use_mps = not args.no_mps and torch.backends.mps.is_available()
        torch.manual_seed(args.seed)
        if use_cuda:
            device = torch.device("cuda")
        elif use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print('device:', device)

        
        mooDataset_train = make_dataset(intial_dataset_size)

        # model = basicNN().to(device)
        if hidden_size_1 != None:
            model = SiameseNetwork_dominance(
                input_size=input_size,
                hidden_size_1=hidden_size_1,
                num_repeated_hidden=num_repeated_hidden,
                hidden_size_2=hidden_size_2,
                fc_size=fc_size
            ).to(device)
        else: 
            model = SiameseNetwork_dominance().to(device)

        model_name = 'siamese_network.pt'
        if args.load_model:
            try:
                model.load_state_dict(torch.load(model_name))
                print('model loaded')
            except:
                print('model could not be loaded')
                pass

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_data = {}
        val_data = {}

        for epoch in range(1, args.epochs + 1):
            train_data[epoch] = train(args, model, device, mooDataset_train, optimizer, epoch)
            val_data[epoch] = Run_test(model)

            save_best = True   
            if save_best:
                print(val_data[epoch][5], val_data[epoch][5] > best_f1)
                if val_data[epoch][5] > best_f1:
                    best_f1 = val_data[epoch][5]
                    torch.save(model.state_dict(), model_name)
                    with open('best_arch.csv', 'w', newline='') as csvfile:
                        best_arch = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        best_arch.writerow(["Input Size", "Hidden Size 1", "Num Repeated Hidden", 
                            "Hidden Size 2", "FC Size", "Compare Size", "Epoch", 
                            "Validation Loss", "Train Loss", "Accuracy", "F1 Score"])

                        best_arch.writerow([input_size, hidden_size_1, num_repeated_hidden, 
                                        hidden_size_2, fc_size, epoch, 
                                        val_data[epoch][1], train_data[epoch], val_data[epoch][3], val_data[epoch][5]])
                    


        # if args.save_model:
        #     torch.save(model.state_dict(), model_name)

        for key in val_data:
            print(key, val_data[key], 'train loss:', train_data[key])


        epochs = []
        validation_loss = []
        train_loss = []
        accuracy = []
        f1_score = []
        
        for key in val_data:
            epochs.append(key)
            validation_loss.append(val_data[key][1])
            train_loss.append(train_data[key])
            accuracy.append(val_data[key][3])
            f1_score.append(val_data[key][5])

        if args.plot:
            
            

            plt.figure(figsize=(10, 6))

            plt.plot(epochs, validation_loss, label="Validation Loss", color='blue', linestyle='--')
            plt.plot(epochs, train_loss, label="Training Loss", color='orange', linestyle='--')
            plt.plot(epochs, accuracy, label="Accuracy", color='green', marker='o')
            plt.plot(epochs, f1_score, label="F1 Score", color='purple', marker='x')

            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.title("Model Performance Metrics Over Epochs")
            plt.legend()
            plt.grid(True)

            plt.show()
        
        return ([epochs, validation_loss, train_loss, accuracy, f1_score]), best_f1


    args = training_args()
    args.lr = 0.0001
    args.epochs = 20
    intial_dataset_size = 50
    args.log_interval = intial_dataset_size

    input_size_values = [4 * 4 * 5]
    hidden_size_1_values = [64, 128, 256]
    num_repeated_hidden_values = [1, 2]
    hidden_size_2_values = [64, 128, 256]
    fc_size_values = [2, 4, 8]

    param_combinations = list(product(input_size_values, hidden_size_1_values, num_repeated_hidden_values,
                                    hidden_size_2_values, fc_size_values))
               
    
    model_arch_search = False
    dataset_size_search = False
    learning_rate_search = True

    best_f1 = 0


    with open('search_results.csv', 'w', newline='') as csvfile:
        results_csv = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if model_arch_search:
            results_csv.writerow(["Model ID", "Input Size", "Hidden Size 1", "Num Repeated Hidden", 
                        "Hidden Size 2", "FC Size", "Compare Size", "Epoch", 
                        "Validation Loss", "Train Loss", "Accuracy", "F1 Score"])

            for i, (input_size, hidden_size_1, num_repeated_hidden, hidden_size_2, fc_size) in enumerate(param_combinations):
                model_metrics = train_run(args, input_size, hidden_size_1, num_repeated_hidden, hidden_size_2, fc_size, best_f1)
                for epoch, val_loss, tr_loss, acc, f1 in zip(*model_metrics):
                    results_csv.writerow([i, input_size, hidden_size_1, num_repeated_hidden, 
                                    hidden_size_2, fc_size, epoch, 
                                    val_loss, tr_loss, acc, f1])
                    

        if dataset_size_search:
            results_csv.writerow(["Dataset_size", "Epoch", 
                        "Validation Loss", "Train Loss", "Accuracy", "F1 Score"])
            sizes_to_search = [10,20,50]
            for dataset_size in sizes_to_search:
                intial_dataset_size = dataset_size
                args.log_interval = intial_dataset_size
                model_metrics, best_f1 = train_run(args, input_size = 16, hidden_size_1 =64, num_repeated_hidden = 1, hidden_size_2 = 256, fc_size = 2, best_f1 = best_f1)
                for epoch, val_loss, tr_loss, acc, f1 in zip(*model_metrics):
                    results_csv.writerow([dataset_size, epoch, 
                                        val_loss, tr_loss, acc, f1])


        
        if learning_rate_search:
            results_csv.writerow(["Dataset_size", "Epoch", 
                        "Validation Loss", "Train Loss", "Accuracy", "F1 Score"])
            learning_rate_to_search = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
            for learning_rate in learning_rate_to_search:
                args.lr = learning_rate
                model_metrics, best_f1 = train_run(args, best_f1 = best_f1)
                for epoch, val_loss, tr_loss, acc, f1 in zip(*model_metrics):
                    results_csv.writerow([learning_rate, epoch, 
                                        val_loss, tr_loss, acc, f1])