import torch.nn as nn
import torch.nn.functional as F
from VladCore import VladCore


class NetVladLayer(nn.Module):

    def __init__(self, w, b, K):
        super(NetVladLayer, self).__init__()
        self.K = K

        # TODO: kernel size, used w, b now (because it says something like that in figure 2.
        # TODO: padding based on kernel size
        # TODO: out channels, set them equal to K now to get vladCore to work, is that correct?
        self.conv = nn.Conv2d(in_channels=256, out_channels=self.K, kernel_size=(w, b))
        self.vlad_core = VladCore(K)

    def forward(self, x):
        """
        input: x <- (W x H x D) map interpreted as N x D local descriptors x
        output:         (K x D) x 1, VLAD vector TODO: although now the 1 x dimension is not there
        """

        a_bar = F.softmax(self.conv(x))
        # TODO: a_bar calculation. I don't think what we have now is the same as in the paper?

        # x <- (1 x N x H x W)
        x = x.flatten(0, 1).flatten(1, 2)  # interpret as N x D local descriptors
        # x <- (N x D)

        V = self.vlad_core(x, a_bar).T
        print(f"V.shape: {V.shape}, expected (K x D) = ({self.K}, {x.shape[0]})")

        # NOTE: intra_normalization should be column-wise. Assumed K are the columns here
        # Intra normalization (column-wise L2 normalization):
        V = F.normalize(V, p=2, dim=0)

        # L2 normalization
        y = F.normalize(V, p=2)

        # TODO: double check the normalization here, because I don't trust it
        return y

    def backward(self, dx):
        return dx  # TODO: backward pass
