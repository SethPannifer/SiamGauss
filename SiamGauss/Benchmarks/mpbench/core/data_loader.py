import os
from scipy.io import loadmat


def load_mat_file(filename):
    base_path = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_path, filename)
    return loadmat(file_path)
