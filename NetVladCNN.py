import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NetVladCNN(torch.nn.Module):

    def __init__(self, base_cnn, K):
        # TODO: input params
        super(NetVladCNN, self).__init__()

        self.base_cnn = base_cnn
        # TODO: get D from base_cnn
        D = 256
        N = 225
        self.netvlad_layer = NetVladLayer(K=K, D=D, N=N)

    def forward(self, x):
        # TODO: make it batch proof, right now it can handle only batches of size 1
        feature_map = self.base_cnn(x)
        # feature_map is now a (D x N) tensor
        y = self.netvlad_layer(feature_map)
        return y


class NetVladLayer(nn.Module):

    def __init__(self, K, D, N):
        super(NetVladLayer, self).__init__()
        self.K = K
        self.D = D
        self.N = N

        # TODO: padding based on kernel size
        self.conv = nn.Conv2d(in_channels=D, out_channels=K, kernel_size=(1, 1), bias=True)  # TODO: bias?
        self.vlad_core = VladCore(K, D, N)

    def forward(self, x):
        """
        input: x <- (1 x D x H x W) map interpreted as N x D local descriptors x
        output:             (K x D) x 1, VLAD vector TODO: although now the 1 x dimension is not there
        """
        K, D, N = self.K, self.D, self.N
        W, H = x.shape[2:]
        assert W * H == N
        assert x.shape == (1, D, W, H)

        # Soft assignment
        a_bar = F.softmax(self.conv(x))  # TODO: softmax dim?
        assert a_bar.shape == (1, K, W, H)

        x = x.view(D, N)  # "interpret as N x D local descriptors"
        a_bar = a_bar.view(K, N)

        # VLAD vector
        V = self.vlad_core(x, a_bar)
        assert V.shape == (K, D)

        # TODO: double check the normalization here, because I don't trust it
        # NOTE: intra_normalization should be column-wise. Assumed K are the columns here
        # Intra normalization (column-wise L2 normalization):
        V = F.normalize(V, p=2, dim=0)

        # L2 normalization
        y = F.normalize(V, p=2)

        return y


class VladCore(nn.Module):

    def __init__(self, K, D, N):
        super(VladCore, self).__init__()
        self.K = K
        self.D = D
        self.N = N

    def forward(self, x, a_bar):
        """
        input:  x     <- (D x N)
                a_bar <- (1 x K x H x W)  -> moet iig N en K hebben voor equation 1
                c     <- TODO: I think c is also a parameter? what dimensions does it have?
        output: V     <- (D x K)
        """
        K, D, N = self.K, self.D, self.N
        assert x.shape == (D, N)
        assert a_bar.shape == (K, N)

        c = np.zeros((K, D))  # TODO: get actual c (parameter) used zeros for now

        V = torch.zeros((K, D))
        with torch.no_grad():  # TODO: this fixes an error so we don't need to go to numpy, but will this cause issues when optimizing?
            for k in range(K):
                V[k] = torch.mv((x.T - c[k]).T, a_bar[k].double())

        return V
