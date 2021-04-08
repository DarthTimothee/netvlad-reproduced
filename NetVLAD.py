import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from colorama import Fore

from helpers import pbar, get_device


device = get_device()


class Reshape(nn.Module):
    def forward(self, input):
        return input.view((input.shape[0], input.shape[1], input.shape[2] * input.shape[3])).permute(0, 2, 1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):  # TODO: what dim?
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class FullNetwork(nn.Module):

    def __init__(self, K, D, features, pooling):
        super(FullNetwork, self).__init__()

        self.K = K
        self.D = D
        self.group = nn.Sequential(features, pooling)

    def forward(self, x):
        return self.group(x.to(device))


class NetVLAD(nn.Module):

    def __init__(self, K, D, cluster_database, base_cnn, N, cluster_samples=1000, alpha=None, bias=False):
        super(NetVLAD, self).__init__()
        self.K = K
        self.D = D
        self.conv = nn.Conv2d(in_channels=D, out_channels=K, kernel_size=(1, 1), bias=bias)

        if not alpha:
            self.alpha = 1000  # NANNE: (-np.log(0.01) / np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
        else:
            self.alpha = alpha

        self.c = nn.Parameter(self.init_clusters(database=cluster_database, base_cnn=base_cnn, N=N, num_samples=cluster_samples))
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.c).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.c.norm(dim=1)
        )

    def init_clusters(self, database, base_cnn, N, num_samples=1000):
        ids = np.random.randint(low=0, high=database.num_images, size=num_samples)

        features = torch.zeros(num_samples * N, self.D, dtype=torch.float32, device=device)
        with pbar(ids, color=Fore.YELLOW, desc="Calculating cluster centers") as t:
            for i, v in enumerate(t):
                feature = base_cnn(database.image_tensor_from_stash(v).unsqueeze(0).to(device))
                features[i * N:(i + 1) * N] = feature.reshape(self.D, N).T

        model = faiss.Kmeans(self.D, self.K, verbose=False)
        model.train(features.cpu().numpy())
        return torch.from_numpy(model.centroids)

    def freeze(self):
        for param in self.conv.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.conv.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        input: x <- (batch_size x D x H x W) map interpreted as N x D local descriptors x
        output:     (K x D) x batch_size, VLAD vectors
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
        V = F.normalize(V, p=2, dim=1)

        # L2 normalization
        y = F.normalize(V, p=2)

        return y


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
