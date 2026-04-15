import random
import torch
from nats_bench import create
from SNN.Resources.NATS_prep import NATS_neuralNetwork, str2matrix, matrix2str
from SNN.Resources.MOO_functions import domination_check
from SNN.snn import SiameseNetwork_dominance
import pathlib
import contextlib
import matplotlib.pyplot as plt
from fine_tuning import fine_tune
path_dir = str(pathlib.Path().resolve()) 
api = create(path_dir +'/SNN/Resources/NATS_DATASETS/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)

def create_search_space(size =1000):
    networks = []
    for i in range (0+7000,size+7000):
        str_arch = api.arch(i)
        input = str2matrix(str_arch)
        network = NATS_neuralNetwork(input)
        if network.check_exists() == '200':
            network.simulate_train()
            networks.append(network)
    return networks

def find_pareto_front(networks):
    pareto_front = []
    for network_test in networks:
        print(network_test.validation_accuracy)
        in_pareto = True
        for network_check in networks:
            
            if network_test.one_over_accuracy > network_check.one_over_accuracy and network_test.flops > network_check.flops:
                in_pareto = False
        if in_pareto:
            pareto_front.append(network_test)

    return pareto_front

def run_search(networks, model_name):
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device = torch.device('cpu')

    model = SiameseNetwork_dominance()
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()

    scores = []
    for network1 in networks:
        score = 0
        for network2 in networks:
            pred = float(model.forward(torch.tensor(network1.encoded_matrix, dtype=torch.float),(torch.tensor(network2.encoded_matrix, dtype=torch.float))))
            score += pred
        scores.append(score)
    print('scores:', scores)
    scored_networks = list(zip(scores, networks))
    sorted_scored_networks = sorted(scored_networks, key=lambda x: x[0], reverse=True)

    print('sorted scores:', sorted_scored_networks)
    sorted_scores, sorted_networks = zip(*sorted_scored_networks)

    return sorted_scores, sorted_networks

if __name__ == '__main__':  

    model_name = 'siamese_network_search.pt'
    networks = create_search_space(size =500)
    pareto_front = find_pareto_front(networks)
    high_acc_pareto_front = []
    for network in pareto_front:
        if network.validation_accuracy > 50:
            high_acc_pareto_front.append(network)
            print(network.validation_accuracy, network.flops)
    pareto_front = high_acc_pareto_front
    print(len(pareto_front))
    
    sorted_scores, sorted_networks = run_search(networks, model_name)
    for idx in range(0,30):
        network = networks[idx]
        print(network.validation_accuracy, network.flops)

    plot = True
    if plot:
        sorted_scores_accuracy = [1 / network.validation_accuracy for network in sorted_networks[:10]]
        sorted_scores_flops = [network.flops for network in sorted_networks[:10]]

        pareto_front_sorted = sorted(pareto_front, key=lambda network: network.flops)
        pareto_front_sorted = sorted(pareto_front, key=lambda network: network.validation_accuracy)
        pareto_front_accuracy = [1 / network.validation_accuracy for network in pareto_front_sorted]
        pareto_front_flops = [network.flops for network in pareto_front_sorted]

        plt.figure(figsize=(10, 6))

        # Use plt.scatter() for the sorted scores (point plot)
        # plt.scatter(sorted_scores_flops[0], sorted_scores_accuracy[0], color='purple', label='SNN top pick')
        plt.scatter(sorted_scores_flops, sorted_scores_accuracy, color='blue', label='SNN top 10')

        # Use plt.plot() for the Pareto Front (line plot)
        plt.plot(pareto_front_flops, pareto_front_accuracy, marker='s', linestyle='-', color='red', label='Pareto Front')

        plt.xlabel('FLOPS')
        plt.ylabel('1/Accuracy')
        plt.title('Pareto Front Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    fine_tune_data_idxs = [network.arch_idx for network in sorted_networks[:20]]
    fine_tune(fine_tune_data_idxs, model_name)

    fine_tuned_sorted_scores, fine_tuned_sorted_networks = run_search(networks, model_name)
    for idx in range(0,30):
        network = networks[idx]
        print(network.validation_accuracy, network.flops)

    plot = True
    if plot:
        fine_tuned_sorted_scores_accuracy = [1 / network.validation_accuracy for network in fine_tuned_sorted_networks[:10]]
        fine_tuned_sorted_scores_flops = [network.flops for network in fine_tuned_sorted_networks[:10]]

        pareto_front_sorted = sorted(pareto_front, key=lambda network: network.flops)
        pareto_front_sorted = sorted(pareto_front, key=lambda network: network.validation_accuracy)
        pareto_front_accuracy = [1 / network.validation_accuracy for network in pareto_front_sorted]
        pareto_front_flops = [network.flops for network in pareto_front_sorted]

        plt.figure(figsize=(10, 6))

        # plt.scatter(sorted_scores_flops[0], sorted_scores_accuracy[0], color='purple', label='SNN top pick')
        plt.scatter(fine_tuned_sorted_scores_flops, fine_tuned_sorted_scores_accuracy, color='blue', label='SNN top 10')

        plt.plot(pareto_front_flops, pareto_front_accuracy, marker='s', linestyle='-', color='red', label='Pareto Front')

        plt.xlabel('FLOPS')
        plt.ylabel('1/Accuracy')
        plt.title('Pareto Front Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

