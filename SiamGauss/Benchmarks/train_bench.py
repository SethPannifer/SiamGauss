from nats_bench import create
import torch
import random
from SNN.Resources.NATS_prep import NATS_neuralNetwork, str2matrix, matrix2str
from SNN.SNN_tools.loss_functions import QuadrupletLossbatch, QuadrupletLoss
from SNN.snn import SiameseNetwork_dominance
from SNN.Resources.MOO_functions import mooDataset

import pathlib
from Benchmarks.mpbench.problems.mpmop1 import MPMOP1
import numpy as np
import torch.nn.functional as F

class training_args():
        def __init__ (self,batch_size = 64,test_batch_size = 1000,epochs = 14,lr = 0.0001,gamma =0.7,
                    no_cuda=False,no_mps=False,dry_run=False,seed=1,log_interval=100,load_model = True, save_model=True, plot = True):
            self.batch_size = batch_size
            self.test_batch_size = test_batch_size
            self.epochs = epochs
            self.lr = lr
            self.lr_start = lr
            self.gamma = gamma
            self.no_cuda = no_cuda
            self.no_mps = no_mps
            self.dry_run = dry_run
            self.seed = seed
            self.log_interval = log_interval
            self.load_model = load_model
            self.save_model = save_model
            self.plot = plot

def make_data(problem, popsize = 100):
    X = problem.lower + (problem.upper - problem.lower) * np.random.rand(popsize, problem.D)
    Y = problem.evaluate(X)
    x = torch.tensor(X).float()
    y = torch.tensor(Y).float()
    mooDataset_ = mooDataset(x, y)
    return mooDataset_


def train_step_slow(args, model, device, dataset, optimizer, epoch, lossfunc = 'quadlossbatch', fast_run = False):
    model.train()
    running_loss = 0
    if lossfunc == 'quadlossbatch':
        criterion = QuadrupletLossbatch().to(device)
        dataset_length = dataset.__len__()
        if fast_run:
            dataset_length -= 1
        for i in range (0,dataset_length):
            if fast_run:
                anchor,positive,neutral,negative  =dataset.__getitem_vectorised__(i)
            else:
                anchor,positive,neutral,negative  =dataset.__getitem__(i)
            
            anchor = anchor.to(device).unsqueeze(0)
            positive = [p.to(device) for p in positive]
            neutral = [n.to(device) for n in neutral]
            negative = [n.to(device) for n in negative]
            output_positive = [model.forward(anchor, positive[j].unsqueeze(0)) for j in range (0,len(positive))]
            output_neutral = [model.forward(anchor, neutral[j].unsqueeze(0)) for j in range (0,len(neutral))]
            output_negative = [model.forward(anchor, negative[j].unsqueeze(0)) for j in range (0,len(negative))]
            loss = criterion(output_positive, output_neutral,output_negative)
            loss = torch.as_tensor(loss, device=device).squeeze()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, i , (dataset.__len__()),
            #         100. * i / (dataset.__len__()), loss.item()))
            #     if args.dry_run:
            #         break
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch,  running_loss))
def train_step_vector(args, model, device, dataset, optimizer, epoch, lossfunc='quadlossbatch', fast_run=False):

    model.train()

    if lossfunc == 'quadlossbatch':
        criterion = QuadrupletLossbatch().to(device)
        dataset_length = len(dataset)

        if fast_run:
            dataset_length -= 1

        loss = 0

        Threshold = 0.5
        conf_matrix = [[0,0,0],[0,0,0],[0,0,0]]

        # --- helper functions defined once ---
        def ensure_batch(x):
            if isinstance(x, list):
                return torch.stack(x).to(device)
            elif isinstance(x, torch.Tensor):
                return x.to(device)
            else:
                raise TypeError(f'Unexpected type {type(x)}')

        def safe_forward(anchor, examples):
            if examples is None or examples.size(0) == 0:
                return None
            batch_anchor = anchor.expand(examples.size(0), -1, -1)  # faster than repeat
            return model(batch_anchor, examples)

        def classify(x):
            if x >= Threshold:
                return 2
            elif x <= -Threshold:
                return 0
            else:
                return 1

        for i in range(dataset_length):

            if fast_run:
                anchor, positive, neutral, negative = dataset.__getitem_vectorised__(i)
            else:
                anchor, positive, neutral, negative = dataset.__getitem__(i)

            anchor = anchor.to(device).unsqueeze(0).unsqueeze(1)

            positive = ensure_batch(positive).unsqueeze(1)
            neutral = ensure_batch(neutral).unsqueeze(1)
            negative = ensure_batch(negative).unsqueeze(1)

            output_positive = safe_forward(anchor, positive)
            output_neutral = safe_forward(anchor, neutral)
            output_negative = safe_forward(anchor, negative)

            loss += criterion(output_positive, output_neutral, output_negative)

            # --- metrics ---
            if output_positive is not None:
                for x in output_positive.detach():
                    conf_matrix[classify(float(x))][2] += 1

            if output_neutral is not None:
                for x in output_neutral.detach():
                    conf_matrix[classify(float(x))][1] += 1

            if output_negative is not None:
                for x in output_negative.detach():
                    conf_matrix[classify(float(x))][0] += 1

        loss = loss / dataset_length

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- compute metrics ---
        precision = []
        recall = []
        f1_score = []

        for i in range(3):
            tp = conf_matrix[i][i]
            precision_i = tp / (sum(conf_matrix[i]) + 1e-6)
            recall_i = tp / (conf_matrix[0][i] + conf_matrix[1][i] + conf_matrix[2][i] + 1e-6)
            f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + 1e-6)

            precision.append(precision_i)
            recall.append(recall_i)
            f1_score.append(f1_i)

        ave_f1_score = sum(f1_score) / 3

        print(f'Train Epoch: {epoch} \tLoss: {loss.item():.6f} \tF1: {ave_f1_score:.6f}')

        return loss.item(), ave_f1_score

def train(args, model, device, mooDataset_train, mooDataset_val,  optimizer, fast_run = False):
    train_loss = {}
    val_data = {}
    for epoch in range(1, args.epochs + 1):
        train_loss[epoch] = train_step_slow(args, model, device, mooDataset_train, optimizer, epoch, fast_run = fast_run)
        # val_data[epoch] = testing(model, device, mooDataset_val, print_vals = False)
        # args.lr = lr_scheduler(args.lr_start, epoch)
    
    return train_loss

# def testing(model, device, dataset, Threshold = 0.5, print_vals = True):
#     accuracy = 0
#     precision =[]
#     recall =[]
#     f1_score=[]
#     conf_matrix = [[0,0,0],[0,0,0],[0,0,0]]
#     num_tests = dataset.__len__()**2

#     for i in range (0,dataset.__len__()):
#         anchor,positive,neutral,negative  =dataset.__getitem__(i)
#         anchor = anchor.to(device).unsqueeze(0)
#         positive = [p.to(device) for p in positive]
#         neutral = [n.to(device) for n in neutral]
#         negative = [n.to(device) for n in negative]
#         output_positive = [model.forward(anchor, positive[j].unsqueeze(0)) for j in range (0,len(positive))]
#         output_neutral = [model.forward(anchor, neutral[j].unsqueeze(0)) for j in range (0,len(neutral))]
#         output_negative = [model.forward(anchor, negative[j].unsqueeze(0)) for j in range (0,len(negative))]

#         # criterion = QuadrupletLossbatch().to(device)
#         # loss = criterion(output_positive, output_neutral,output_negative).squeeze()
        
#         acc_positive = sum(x >= Threshold for x in output_positive) 
#         acc_neutral = sum(-Threshold <= x <= Threshold for x in output_neutral)
#         acc_negative = sum(x <= -Threshold for x in output_negative) 
#         accuracy += (acc_positive + acc_neutral + acc_negative)

#         for x in output_positive:
#             conf_matrix[int(round(float(x), 0)) + 1][2] += 1
#         for x in output_neutral:
#             conf_matrix[int(round(float(x), 0)) + 1][1] += 1
#         for x in output_negative:
#             conf_matrix[int(round(float(x), 0)) + 1][0] += 1

#     for i in range(0,2):
#         precision.append(conf_matrix[i][i]/(sum(conf_matrix[i])+0.000001))
#         recall.append(conf_matrix[i][i]/(conf_matrix[0][i]+conf_matrix[1][i]+conf_matrix[2][i]+0.000001))
#         f1_score.append(2 * ((precision[i]*recall[i]) / (precision[i]+recall[i]+0.000001)))
#     ave_f1_score = sum(f1_score)/len(f1_score)

#     if print_vals:
#         print(conf_matrix)
#         print('acc:',accuracy/num_tests,'f1:',ave_f1_score)

#     return accuracy/num_tests, ave_f1_score

def testing(model, device, dataset, Threshold=0.5, print_vals=True):

    model.eval()

    conf_matrix = torch.zeros((3,3), dtype=torch.int64)
    accuracy = 0
    num_tests = len(dataset)

    def classify(x):
        if x >= Threshold:
            return 2   # positive
        elif x <= -Threshold:
            return 0   # negative
        else:
            return 1   # neutral

    with torch.no_grad():

        for i in range(len(dataset)):

            anchor, positive, neutral, negative = dataset.__getitem__(i)

            anchor = anchor.to(device).unsqueeze(0)

            positive = [p.to(device) for p in positive]
            neutral = [n.to(device) for n in neutral]
            negative = [n.to(device) for n in negative]

            output_positive = [model(anchor, p.unsqueeze(0)).item() for p in positive]
            output_neutral = [model(anchor, n.unsqueeze(0)).item() for n in neutral]
            output_negative = [model(anchor, n.unsqueeze(0)).item() for n in negative]

            # --- accuracy ---
            accuracy += sum(x >= Threshold for x in output_positive)
            accuracy += sum(-Threshold <= x <= Threshold for x in output_neutral)
            accuracy += sum(x <= -Threshold for x in output_negative)

            # --- confusion matrix ---
            for x in output_positive:
                pred = classify(x)
                conf_matrix[pred][2] += 1

            for x in output_neutral:
                pred = classify(x)
                conf_matrix[pred][1] += 1

            for x in output_negative:
                pred = classify(x)
                conf_matrix[pred][0] += 1

    # --- metrics ---
    precision = []
    recall = []
    f1_score = []

    for i in range(3):

        tp = conf_matrix[i][i].item()

        precision_i = tp / (conf_matrix[i].sum().item() + 1e-6)
        recall_i = tp / (conf_matrix[:,i].sum().item() + 1e-6)

        f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + 1e-6)

        precision.append(precision_i)
        recall.append(recall_i)
        f1_score.append(f1_i)

    ave_f1_score = sum(f1_score) / 3
    acc = conf_matrix.diag().sum().item() / conf_matrix.sum().item()

    if print_vals:
        print(conf_matrix.tolist())
        print("acc:", acc, "f1:", ave_f1_score)

    return acc, ave_f1_score





def lr_scheduler(lr, epoch):
    return max(lr * (1 - ((0.1*epoch)**2)), 0.00005)

if __name__ == '__main__':   
    args = training_args(lr = 0.0005)
    args.epochs = 20
    intial_dataset_size = 200
    args.log_interval = intial_dataset_size



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
    
    # mooDataset_train = make_dataset(intial_dataset_size)    
    problem = MPMOP1(30)
    popsize = 100

    mooDataset_train = make_data(problem, popsize = popsize)
    mooDataset_val = make_data(problem, popsize = popsize)

    model = SiameseNetwork_dominance(input_size = 30,num_repeated_hidden = 1, hidden_size_2 =  128, fc_size =  6, convD = 1).to(device)

    model_name = 'Benchmarks/Models/siamese_network_benchmark.pt'
    if args.load_model:
        try:
            model.load_state_dict(torch.load(model_name))
            print('model loaded')
        except:
            print('model could not be loaded')
            pass

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_data = train(args, model, device, mooDataset_train, mooDataset_val, optimizer)

    mooDataset_val = make_data(problem, popsize = popsize)
    testing(model, device, mooDataset_val)

    if args.save_model:
        torch.save(model.state_dict(), model_name)
    

    # for key in val_data:
    #     print(key, val_data[key], 'train loss:', train_data[key])

