import random
import torch
from nats_bench import create
from SiamGauss.SNN.Resources.NATS_prep import NATS_neuralNetwork, str2matrix
from SiamGauss.SNN.snn import SiameseNetwork_dominance
from SiamGauss.SNN.Resources.MOO_functions import domination_check
from SiamGauss.paths import model_name, api, dataset
import pathlib

path_dir = str(pathlib.Path().resolve()) 

def Run_test(model, dataset_index_offset = 200):

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device = torch.device('cpu')

    model.to(device)
    model.eval()
    loss_accuracy = 0
    accuracy = 0 
    num_tests = 200
    conf_matrix = [[0,0,0],[0,0,0],[0,0,0]]

    

    for i in range(0+ dataset_index_offset, num_tests + dataset_index_offset):   
        str_arch = api.arch(i)
        input = str2matrix(str_arch)
        network = NATS_neuralNetwork(input)
        if network.check_exists() == '200':
            
            network.simulate_train()
            objectives = [network.one_over_accuracy,network.flops]

        str_arch2 = api.arch(i+1)
        input2 = str2matrix(str_arch2)
        network2 = NATS_neuralNetwork(input2)
        if network2.check_exists() == '200':
            network2.simulate_train()
            objectives2 = [network2.one_over_accuracy,network2.flops]

        actual = domination_check(objectives,objectives2)
        pred = model.forward(torch.tensor(network.encoded_matrix, dtype=torch.float),(torch.tensor(network2.encoded_matrix, dtype=torch.float)))
        loss_accuracy += (float(pred)  - actual)**2
        if round(float(pred),0) == actual:
            accuracy  += 1
        conf_matrix[int(round(float(pred),0))+1][actual+1]+=1

    precision =[]
    recall =[]
    f1_score=[]
    print(conf_matrix)
    for i in range(0,2):
        precision.append(conf_matrix[i][i]/(sum(conf_matrix[i])+0.000001))
        recall.append(conf_matrix[i][i]/(conf_matrix[0][i]+conf_matrix[1][i]+conf_matrix[2][i]+0.000001))
        f1_score.append(2 * ((precision[i]*recall[i]) / (precision[i]+recall[i]+0.000001)))

    ave_f1_score = sum(f1_score)/len(f1_score)
    print('Results:')
    print('loss:',loss_accuracy/num_tests,'acc:',accuracy/num_tests,'f1:',ave_f1_score)
    return ('loss:',loss_accuracy/num_tests,'acc:',accuracy/num_tests,'f1:',ave_f1_score)

if __name__ == '__main__':  

    model = SiameseNetwork_dominance()
    model.load_state_dict(torch.load(model_name))
    Run_test(model)



