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
        self.netvlad_layer = NetVladLayer(K=K, D=D)

    def forward(self, x, c):
        # TODO: make it batch proof, right now it can handle only batches of size 1
        feature_map = self.base_cnn(x)
        # feature_map is now a (D x N) tensor
        y = self.netvlad_layer(feature_map, c)
        return y


class NetVladLayer(nn.Module):

    def __init__(self, K, D):
        super(NetVladLayer, self).__init__()
        self.K = K
        self.D = D

        # TODO: padding based on kernel size
        self.conv = nn.Conv2d(in_channels=D, out_channels=K, kernel_size=(1, 1), bias=True)  # TODO: bias?
        self.vlad_core = VladCore(K, D)

    def forward(self, x, c):
        """
        input: x <- (1 x D x H x W) map interpreted as N x D local descriptors x
        output:             (K x D) x 1, VLAD vector TODO: although now the 1 x dimension is not there
        """
        K, D = self.K, self.D
        H, W = x.shape[2:]
        N = W * H
        assert x.shape == (1, D, H, W)

        # Soft assignment
        features = self.conv(x)
        assert features.shape == (1, K, H, W)

        a_bar = F.softmax(features, dim=1)
        assert a_bar.shape == (1, K, H, W)

        x = x.view(D, N)  # "interpret as N x D local descriptors"
        a_bar = a_bar.view(K, N)

        # VLAD vector
        V = self.vlad_core(x, a_bar, c)
        assert V.shape == (K, D)

        # TODO: double check the normalization here, because I don't trust it
        # NOTE: intra_normalization should be column-wise. Assumed K are the columns here
        # Intra normalization (column-wise L2 normalization):
        V = F.normalize(V, p=2, dim=0)

        # L2 normalization
        y = F.normalize(V, p=2)

        return y


class VladCore(nn.Module):

    def __init__(self, K, D):
        super(VladCore, self).__init__()
        self.K = K
        self.D = D

    def forward(self, x, a_bar, c):
        """
        input:  x     <- (D x N)
                a_bar <- (1 x K x H x W)  -> moet iig N en K hebben voor equation 1
                c     <- TODO: I think c is also a parameter? what dimensions does it have?
        output: V     <- (D x K)
        """
        K, D, N = self.K, self.D, x.shape[1]
        assert x.shape == (D, N)
        assert a_bar.shape == (K, N)

        V = torch.zeros((K, D))
        x_numpy = x.detach().numpy()
        #with torch.no_grad():  # TODO: this fixes an error so we don't need to go to numpy, but will this cause issues when optimizing?
        for k in range(K):
            asdf = (x_numpy.T - c[k]).T
            V[k] = torch.mv(torch.tensor(asdf), a_bar[k].double())

        return V
