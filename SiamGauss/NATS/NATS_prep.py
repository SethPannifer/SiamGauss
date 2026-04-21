import pathlib
from nats_bench import create
path_dir = str(pathlib.Path().resolve())

import os, copy, random, torch, numpy as np
from typing import List, Text, Union, Dict, Optional
import torch.nn.functional as F

from SiamGauss.NATS.paths import model_name, api, dataset

#the str2matrix function is taken from https://github.com/D-X-Y/NAS-Bench-201/blob/master/nas_201_api/api_201.py
def str2matrix(arch_str: Text,
                 search_space: List[Text] = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']) -> np.ndarray:
    """
    This func shows how to convert the string-based architecture encoding to the encoding strategy in NAS-Bench-101.

    :param
      arch_str: the input is a string indicates the architecture topology, such as
                    |nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|
      search_space: a list of operation string, the default list is the search space for NAS-Bench-201
        the default value should be be consistent with this line https://github.com/D-X-Y/AutoDL-Projects/blob/master/lib/models/cell_operations.py#L24
    :return
      the numpy matrix (2-D np.ndarray) representing the DAG of this architecture topology
    :usage
      matrix = api.str2matrix( '|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|' )
      This matrix is 4-by-4 matrix representing a cell with 4 nodes (only the lower left triangle is useful).
         [ [0, 0, 0, 0],  # the first line represents the input (0-th) node
           [2, 0, 0, 0],  # the second line represents the 1-st node, is calculated by 2-th-op( 0-th-node )
           [0, 0, 0, 0],  # the third line represents the 2-nd node, is calculated by 0-th-op( 0-th-node ) + 0-th-op( 1-th-node )
           [0, 0, 1, 0] ] # the fourth line represents the 3-rd node, is calculated by 0-th-op( 0-th-node ) + 0-th-op( 1-th-node ) + 1-th-op( 2-th-node )
      In NAS-Bench-201 search space, 0-th-op is 'none', 1-th-op is 'skip_connect',
         2-th-op is 'nor_conv_1x1', 3-th-op is 'nor_conv_3x3', 4-th-op is 'avg_pool_3x3'.
    :(NOTE)
      If a node has two input-edges from the same node, this function does not work. One edge will be overlapped.
    """
    node_strs = arch_str.split('+')
    num_nodes = len(node_strs) + 1
    matrix = np.zeros((num_nodes, num_nodes))
    for i, node_str in enumerate(node_strs):
      inputs = list(filter(lambda x: x != '', node_str.split('|')))
      for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
      for xi in inputs:
        op, idx = xi.split('~')
        if op not in search_space: raise ValueError('this op ({:}) is not in {:}'.format(op, search_space))
        op_idx, node_idx = search_space.index(op), int(idx)
        matrix[i+1, node_idx] = op_idx
    return matrix

def matrix2str(arch_matrix: List,
                 search_space: List[Text] = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']) -> np.ndarray: 
    output_str = ''
    string_started = False
    for row_idx, matrix_row in enumerate(arch_matrix):
        #convert matrix row to string
        row = ''
        row_started = 0
        print()
        for idx, entry in enumerate(matrix_row):
            # print('test',idx, entry)
            if entry not in [0,1,2,3,4]:
                print("Error, Invalid matrix index:" + str(entry))  
            if entry != 0:

                for none_position in range(row_started,idx):
                    row += '|none~'+str(none_position)
                row +=  '|'+search_space[int(entry)]+'~'+str(idx)
                row_started = idx +1
            
            if  row_idx == 2:
                if idx ==1:
                    if  entry == 0:
                        if row != '':
                            row += '|none~1'
                
            if row_idx == 3:
                if idx ==2:
                    if  entry == 0:
                        if row != '':
                            if matrix_row[idx-1] == 0:
                                row += '|none~1|none~2'
                            else:
                                row += '|none~2'
                    
        if row != '':
            row = row+ '|'
        
        if string_started:
            output_str += '+'
        else:
            pass 
        if row == '':
            for none_position in range(0,row_idx):
                output_str += '|none~'+str(none_position)
            if row_idx != 0:
                output_str += '|'
        else:    
            output_str += row 
        if output_str != '':
            string_started = True
    return output_str




# datasets:
# cifar10
# cifar100
# ImageNet16-120

class NATS_neuralNetwork():
    def __init__(self,matrix, dataset = 'cifar100', api = api):
        self.dataset = dataset
        self.matrix = matrix
        self.encoded_matrix = F.one_hot(torch.tensor(self.matrix, dtype=torch.int64), num_classes=5).float()  # Shape: (4, 4, 5)
        self.encoded_matrix = self.encoded_matrix.permute(2, 0, 1).unsqueeze(0)
        self.string = matrix2str(self.matrix)
        self.arch_idx = api.query_index_by_arch(self.string)
        self.api = api
    def check_exists(self):
        if self.arch_idx == -1:
            return('Error, Matrix is not associated to a pre-trained model')
        self.info = self.api.get_cost_info(self.arch_idx, self.dataset)
        self.flops = self.info['flops']
        return '200'
    def simulate_train(self, hp ='12'):
        self.validation_accuracy, self.latency, self.time_cost, self.current_total_time_cost = self.api.simulate_train_eval(self.arch_idx, dataset=self.dataset, hp=hp)
        self.one_over_accuracy = 1/self.validation_accuracy
if __name__ == "__main__":
  
  testNN =NATS_neuralNetwork()
  if testNN.check_exists() == '200':
      testNN.simulate_train()
      print(testNN.validation_accuracy)
      print(testNN.flops)


