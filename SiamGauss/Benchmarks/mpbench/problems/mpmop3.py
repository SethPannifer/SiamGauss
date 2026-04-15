import numpy as np
import torch
from ..core.mpmop_value import MPMOP_Value


class MPMOP3:
    def __init__(self, D):
        self.M = 4
        self.DM = 2
        self.D = D
        self.maxFE = 1000 * D * self.DM

        self.lower = -1 * np.ones(D)
        self.upper = np.ones(D)
        self.lower[0] = 0

        self.calcount = 0

        self.base_problem = 'MPMOP3'
        self.t1 = 0
        self.t2 = np.pi / 2

    def evaluate(self, X):
        input_is_torch = isinstance(X, torch.Tensor)
        if input_is_torch:
            device = X.device
            X = X.detach().cpu().numpy()

        if self.calcount > self.maxFE:
            raise RuntimeError("Maximum evaluations exceeded.")

        N = X.shape[0]
        PopObj = np.zeros((N, self.M))

        PopObj[:, 0:2] = MPMOP_Value(self.base_problem, X, self.t1)
        PopObj[:, 2:4] = MPMOP_Value(self.base_problem, X, self.t2)

        self.calcount += N

        if input_is_torch:
            return torch.tensor(PopObj, dtype=torch.float32, device=device)

        return PopObj
