import pathlib
from nats_bench import create
path_dir = str(pathlib.Path().resolve())

import os, copy, random, torch, numpy as np
from typing import List, Text, Union, Dict, Optional
from NATS_prep import matrix2str, str2matrix

api =create(path_dir +'/SNN/Resources/NATS_DATASETS/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)


for i in range (0,10):
    architecture_matrix = str2matrix(api.arch(i))
    print(architecture_matrix)