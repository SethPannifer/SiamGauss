import pathlib
from nats_bench import create
path_dir = str(pathlib.Path().resolve())

import os, copy, random, torch, numpy as np
from typing import List, Text, Union, Dict, Optional
from SiamGauss.SNN.NATS_prep import matrix2str, str2matrix
from SiamGauss.paths import model_name, api, dataset

for i in range (0,10):
    architecture_matrix = str2matrix(api.arch(i))
    print(architecture_matrix)