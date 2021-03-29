import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore
from tqdm import tqdm


class NetVladCNN(torch.nn.Module):

    def __init__(self, base_cnn, K):
        super(NetVladCNN, self).__init__()

        self.base_cnn = base_cnn
        self.K = K
        self.D = base_cnn.get_output_dim()
        self.c = torch.zeros((self.K, self.D))
        self.netvlad_layer = NetVladLayer(K=K, D=self.D)

    def forward(self, x):
        """
        inputs:
            x <- (batch_size, 3, H_pixels, W_pixels) input image tensor
        output:
                 (K, N, batch_size)
        """
        feature_map = self.base_cnn(x.cpu())
        # feature_map is now a (D x N) tensor
        return self.netvlad_layer(feature_map, self.c)

    def init_clusters(self, database, N=25, num_samples=1000):  # TODO: don't hardcode N=25
        ids = np.random.randint(low=0, high=database.num_images, size=num_samples)

        features = np.zeros((num_samples * N, self.D)).astype('float32')
        # TODO: batches
        with tqdm(ids, position=0,
                  leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Fore.YELLOW)) as t:
            t.set_description("Calculating cluster centers\t")
            for i, v in enumerate(t):
                features[i * N:(i + 1) * N] = self.base_cnn(database.image_to_tensor(v).unsqueeze(0)).reshape(
                    self.D, N).T

        model = faiss.Kmeans(self.D, self.K, verbose=False)
        model.train(features)

        self.c = torch.from_numpy(model.centroids)

    def freeze(self):
        self.netvlad_layer.freeze()

    def unfreeze(self):
        self.netvlad_layer.unfreeze()


class NetVladLayer(nn.Module):

    def __init__(self, K, D, bias=False):
        super(NetVladLayer, self).__init__()
        self.K = K
        self.D = D
        self.conv = nn.Conv2d(in_channels=D, out_channels=K, kernel_size=(1, 1), bias=bias)
        self.vlad_core = VladCore(K, D)

    def freeze(self):
        for param in self.conv.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.conv.parameters():
            param.requires_grad = True

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

        R = c * torch.sum(a_bar, dim=2).unsqueeze(-1)
        L = torch.bmm(a_bar, x.permute(0, 2, 1))
        return (L + R).T


class AlexBase(nn.Module):

    def __init__(self):
        super(AlexBase, self).__init__()

        # Setup base network
        self.full_cnn = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        self.features = self.full_cnn.features[:11]

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

    def get_output_dim(self):
        return self.features[-1].out_channels  # TODO: klopt dit?
