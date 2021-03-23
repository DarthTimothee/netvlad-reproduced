import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NetVladCNN(torch.nn.Module):

    def __init__(self, base_cnn, K, c):
        # TODO: input params
        super(NetVladCNN, self).__init__()

        self.base_cnn = base_cnn
        self.c = c
        self.D = base_cnn.get_output_dim()
        self.netvlad_layer = NetVladLayer(K=K, D=self.D)

    def forward(self, x):
        """
        inputs:
            x <- (batch_size, 3, H_pixels, W_pixels) input image tensor
        output:
                 (K, N, batch_size)
        """

        feature_map = self.base_cnn(x)
        # feature_map is now a (D x N) tensor
        return self.netvlad_layer(feature_map, self.c)

    def set_clusters(self, c):
        self.c = c


class NetVladLayer(nn.Module):

    def __init__(self, K, D):
        super(NetVladLayer, self).__init__()
        self.K = K
        self.D = D
        self.conv = nn.Conv2d(in_channels=D, out_channels=K, kernel_size=(1, 1), bias=True)  # TODO: bias?
        self.vlad_core = VladCore(K, D)

    def forward(self, x, c):
        """
        input: x <- (batch_size x D x H x W) map interpreted as N x D local descriptors x
        output:     (K x D) x batch_size, VLAD vectors
        """
        K, D = self.K, self.D
        H, W = x.shape[2:]
        batch_size = x.shape[0]
        N = W * H
        assert x.shape == (batch_size, D, H, W)

        # Soft assignment
        features = self.conv(x)
        assert features.shape == (batch_size, K, H, W)

        a_bar = F.softmax(features, dim=2)
        assert a_bar.shape == (batch_size, K, H, W)

        x = x.view(batch_size, D, N)  # "interpret as N x D local descriptors"
        a_bar = a_bar.view(batch_size, K, N)

        # VLAD vector
        V = self.vlad_core(x, a_bar, c)
        assert V.shape == (D, K, batch_size)

        # TODO: double check the normalization here, because I don't trust it
        # NOTE: intra_normalization should be column-wise. Assumed K are the columns here
        # Intra normalization (column-wise L2 normalization):
        V = F.normalize(V, p=2, dim=1)

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
        output: V     <- (D x K) x batch_size
        """
        K, D, N, batch_size = self.K, self.D, x.shape[2], x.shape[0]
        assert x.shape == (batch_size, D, N)
        assert a_bar.shape == (batch_size, K, N)

        V = torch.zeros((batch_size, K, D))
        x_numpy = x.detach().numpy()
        # with torch.no_grad():  # TODO: this fixes an error so we don't need to go to numpy, but will this cause issues when optimizing?
        for b in range(batch_size):
            for k in range(K):
                # print(f"x_numpy.T.shape: {x_numpy.T.shape}")
                # print(f"c.shape: {c.shape}")
                A = torch.tensor(x_numpy.T - c[None, k][:, :, None])  # TODO: probably better syntax for this?
                B = a_bar[:, k].double()
                # print(f"A.shape: {A.shape}")
                # print(f"B.shape: {B.shape}")
                Y = torch.tensordot(A[:, :, b], B[b, :], dims=([0], [0]))
                # print(f"Y.shape: {Y.shape}")

                V[b, k] = Y

        return V.T


class AlexBase(nn.Module):

    def __init__(self):
        super(AlexBase, self).__init__()

        # Setup base network
        self.full_cnn = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        self.features = self.full_cnn.features[:11]

    def forward(self, x):
        return self.features(x)

    def get_output_dim(self):
        return self.features[-1].out_channels  # TODO: klopt dit?
