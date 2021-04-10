import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from colorama import Fore
from sklearn.neighbors import NearestNeighbors

from helpers import pbar, get_device

device = get_device()


class Reshape(nn.Module):
    def forward(self, input):
        return input.view((input.shape[0], input.shape[1], input.shape[2] * input.shape[3])).permute(0, 2, 1).view(
            (input.shape[0], -1))


class L2Norm(nn.Module):
    def __init__(self, dim=1):  # TODO: what dim?
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class FullNetwork(nn.Module):

    def __init__(self, features, pooling):
        super(FullNetwork, self).__init__()

        self.K = pooling.K if isinstance(pooling, NetVLAD) else 1
        self.D = features.get_output_dim()
        self.group = nn.Sequential(features, pooling)

    def forward(self, x):
        return self.group(x.to(device))

    def freeze(self):
        self.eval()
        self.group[0].freeze()
        if isinstance(self.group[1], NetVLAD):
            self.group[1].freeze()

    def unfreeze(self):
        self.train()
        self.group[0].unfreeze()
        if isinstance(self.group[1], NetVLAD):
            self.group[1].unfreeze()

class NetVLAD(nn.Module):

    def __init__(self, K, N, cluster_database, base_cnn, cluster_samples=1000, bias=False):
        super(NetVLAD, self).__init__()
        self.K = K
        self.D = base_cnn.get_output_dim()
        self.conv = nn.Conv2d(in_channels=self.D, out_channels=K, kernel_size=(1, 1), bias=bias)

        # Initialize the clusters
        clusters, features = self.init_clusters(database=cluster_database, base_cnn=base_cnn, N=N, num_samples=cluster_samples)
        self.c = nn.Parameter(torch.from_numpy(clusters))

        # Initializations below are borrowed from Nanne,
        # because we could not figure out what they should be based on only the appendix of the paper.
        knn = NearestNeighbors(n_jobs=-1)  # TODO: faiss?
        knn.fit(features)
        _, distances = knn.kneighbors(clusters, 2)
        distances_squared = distances ** 2

        self.alpha = (-np.log(0.01) / (distances_squared[:, 1] - distances_squared[:, 0]).mean()).item()
        self.conv.weight = nn.Parameter((2.0 * self.alpha * self.c).unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = nn.Parameter(-self.alpha * self.c.norm(dim=1))

    def init_clusters(self, database, base_cnn, N, num_samples=1000):
        ids = np.random.randint(low=0, high=database.num_images, size=num_samples)

        features = torch.zeros(num_samples * N, self.D, dtype=torch.float32, device=device)
        with pbar(ids, color=Fore.YELLOW, desc="Calculating cluster centers") as t:
            for i, v in enumerate(t):
                feature = base_cnn(database.get_image_tensor(v).to(device))
                features[i * N:(i + 1) * N] = feature.reshape(self.D, N).T

        features = features.cpu().detach().numpy()
        model = faiss.Kmeans(self.D, self.K, verbose=False)
        model.train(features)
        return model.centroids, features

    def freeze(self):
        for param in self.conv.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.conv.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        input: x <- (batch_size, D, H, W) map interpreted as N x D local descriptors x
        output:     (batch_size, K * D) VLAD vectors
        """
        K, D = self.K, self.D
        H, W = x.shape[2:]
        batch_size = x.shape[0]
        N = W * H
        # assert x.shape == (batch_size, D, H, W)

        # Soft assignment
        features = self.conv(x)
        # assert features.shape == (batch_size, K, H, W)

        a_bar = F.softmax(features, dim=2)
        # assert a_bar.shape == (batch_size, K, H, W)

        x = x.view(batch_size, D, N)  # "interpret as N x D local descriptors"
        a_bar = a_bar.view(batch_size, K, N)
        # assert x.shape == (batch_size, D, N)
        # assert a_bar.shape == (batch_size, K, N)

        R = self.c * torch.sum(a_bar, dim=2).unsqueeze(-1)
        L = torch.bmm(a_bar, x.permute(0, 2, 1))
        V = L + R
        # assert V.shape == (batch_size, K, D)

        # TODO: double check the normalization here, because I don't trust it
        # NOTE: intra_normalization should be column-wise. Assumed K are the columns here
        # Intra normalization (column-wise L2 normalization):

        # First normalize column-wise, then flatten the entire thing to be able to normalize in its entirety
        V = F.normalize(V, p=2, dim=2)
        V = V.view(x.shape[0], D * K)
        V = F.normalize(V, p=2)

        return V


class AlexBase(nn.Module):

    def __init__(self):
        super(AlexBase, self).__init__()

        # Setup base network
        self.full_cnn = models.alexnet(pretrained=True, progress=True)
        self.features = nn.Sequential(*self.full_cnn.features[:-2])

        for param in self.features.parameters():
            param.requires_grad = False

        for param in self.features[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.features(x.to(device))

    def get_output_dim(self):
        return self.features[-1].out_channels

    def freeze(self):
        self.eval()
        for param in self.features[-1].parameters():
            param.requires_grad = False

    def unfreeze(self):
        self.train()
        for param in self.features[-1].parameters():
            param.requires_grad = True


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # Setup base network
        self.full_cnn = models.vgg16(pretrained=True, progress=True)
        self.features = nn.Sequential(*self.full_cnn.features[:-2])

        for param in self.features.parameters():
            param.requires_grad = False

        # TODO: unfreeze correct layers
        for param in self.features[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.features(x.to(device))

    def get_output_dim(self):
        return self.features[-1].out_channels  # TODO: klopt dit?

    def freeze(self):
        self.eval()
        for param in self.features[-1].parameters():  # TODO: correct layers
            param.requires_grad = False

    def unfreeze(self):
        self.train()
        for param in self.features[-1].parameters():  # TODO: correct layers
            param.requires_grad = True