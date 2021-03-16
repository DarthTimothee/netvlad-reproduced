import torch.nn as nn
import torch
import numpy as np


class VladCore(nn.Module):

    def __init__(self, K):
        super(VladCore, self).__init__()
        self.K = K

    def forward(self, x, a_bar):
        """
        input:  x     <- (D x N)
                a_bar <- (1 x K x H x W)  -> moet iig N en K hebben voor equation 1
                c     <- TODO: I think c is also a parameter?
        output: V     <- (D x K)
        """

        print(f"VladCore->forward: x.shape     {x.shape}")
        print(f"VladCore->forward: a_bar.shape {a_bar.shape}")

        D = x.shape[0]
        N = x.shape[1]
        K = self.K
        c = np.zeros((K, D))  # TODO: get actual c (parameter) used zeros for now

        a_bar = a_bar.flatten(0, 1).flatten(1, 2).detach().numpy()
        x = x.detach().numpy().T
        # TODO: make it so we don't have to go to numpy for this

        print(f"a_bar.shape = {a_bar.shape}, required: ({K}, {N})")
        print(f"x.shape     = {x.shape}, required: ({N}, {D})")

        V = np.zeros((D, K))
        # for j in range(D): N.B. I used : notation to vectorize the j loop
        for k in range(K):
            print(f"k={k} / {K}")
            for i in range(N):
                V[:, k] = a_bar[k, i] * (x[i, :] - c[k, :])
        # TODO: do with tensor operations instead of dumb looping
        return torch.tensor(V)

    def backward(self, dx):
        return dx  # TODO: backward pass???
