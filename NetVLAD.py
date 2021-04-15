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


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class FullNetwork(nn.Module):

    def __init__(self, features, pooling):
        super(FullNetwork, self).__init__()

        self.K = pooling.K if isinstance(pooling, NetVLAD) else 1
        self.D = features.get_output_dim()
        self.group = nn.Sequential(*list(filter(None, [
            features,
            pooling,
            L2Norm(dim=2) if isinstance(pooling, NetVLAD) else None,
            nn.Flatten(),
            L2Norm()
        ])))

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

    def __init__(self, K, N, cluster_database, base_cnn, cluster_samples=1000, alpha=0.1, bias=False):
        super(NetVLAD, self).__init__()
        self.K = K
        self.D = base_cnn.get_output_dim()
        self.alpha = alpha
        self.conv = nn.Conv2d(in_channels=self.D, out_channels=K, kernel_size=(1, 1), bias=bias)

        # Initialize the clusters
        clusters, features = self.init_clusters(database=cluster_database, base_cnn=base_cnn, N=N,
                                                num_samples=cluster_samples)
        # TODO: initialize alpha according to appendix
        self.c = nn.Parameter(torch.from_numpy(clusters))
        self.conv.weight = nn.Parameter((2.0 * self.alpha * self.c).unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = nn.Parameter(-self.alpha * self.c.norm(dim=1) ** 2)

    def init_clusters(self, database, base_cnn, N, num_samples=1000):
        ids = np.random.randint(low=0, high=database.num_images, size=num_samples)

        features = torch.zeros(num_samples * N, self.D, dtype=torch.float32, device=device)
        with pbar(ids, color=Fore.YELLOW, desc="Calculating cluster centers") as t:
            for i, image_id in enumerate(t):
                feature = base_cnn(database.get_image_tensor(image_id).to(device))  # B, D, W, H
                features[i * N:(i + 1) * N] = feature.reshape(self.D, N).T

        features = features.cpu().detach().numpy()
        model = faiss.Kmeans(self.D, self.K, niter=100, verbose=False)
        model.train(features)
        return model.centroids, features

    def freeze(self):
        self.eval()
        for param in self.conv.parameters():
            param.requires_grad = False

    def unfreeze(self):
        self.train()
        for param in self.conv.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        input: x <- (batch_size, D, H, W) map interpreted as N x D local descriptors x
        output:     (batch_size, K * D) VLAD vectors
        """
        K, D, (batch_size, _, H, W) = self.K, self.D, x.shape[:]

        # Soft assignment
        a_bar = F.softmax(self.conv(x), dim=1)
        a_bar = a_bar.view(batch_size, K, -1)

        x = x.view(batch_size, D, -1)  # "interpret as N x D local descriptors"

        # Vlad core calculation
        return torch.bmm(a_bar, x.permute(0, 2, 1)) - self.c * torch.sum(a_bar, dim=2).unsqueeze(-1)


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
