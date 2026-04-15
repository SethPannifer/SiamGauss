import numpy as np
import torch
from ..core.mpmop_value import MPMOP_Value


class MPMOP1:
    """
    Exact Python translation of MATLAB MPMOP1.m
    """

    def __init__(self, D):
        self.M = 4
        self.DM = 2
        self.D = D
        self.maxFE = 1000 * D * self.DM

        self.lower = np.zeros(D)
        self.upper = np.ones(D)
        self.lower[0] = 1
        self.upper[0] = 4

        self.calcount = 0

        # CRITICAL: These must match MATLAB file
        self.base_problem = 'MPMOP1'
        self.t1 = 1
        self.t2 = 2

    def evaluate(self, X):
        """
        X: (N, D)
        Returns: (N, 4)
        """

        input_is_torch = isinstance(X, torch.Tensor)

        if input_is_torch:
            device = X.device
            X = X.detach().cpu().numpy()

        if self.calcount > self.maxFE:
            raise RuntimeError("Maximum number of evaluations exceeded.")

        N = X.shape[0]
        M = self.M

        PopObj = np.zeros((N, M))

        # First task
        PopObj[:, 0:M//2] = MPMOP_Value(
            self.base_problem,
            X,
            self.t1
        )

        # Second task
        PopObj[:, M//2:M] = MPMOP_Value(
            self.base_problem,
            X,
            self.t2
        )

        self.calcount += N

        if input_is_torch:
            return torch.tensor(PopObj, dtype=torch.float32, device=device)

        return PopObj



if __name__ == '__main__':  
    problem = MPMOP1(30)
    popsize = 100
    X0 = problem.lower + (problem.upper - problem.lower) * np.random.rand(popsize, problem.D)
    Y0 = problem.evaluate(X0)

    print(X0.shape)
    print(Y0.shape)

