import torch

from NetVladLayer import NetVladLayer


class NetVladCNN(torch.nn.Module):

    def __init__(self, base_cnn, K, w=1, b=1):
        # TODO: input params
        super(NetVladCNN, self).__init__()

        self.base_cnn = base_cnn
        self.netvlad_layer = NetVladLayer(w, b, K)

    def forward(self, x):
        # TODO: make it batch proof, right now it can handle only batches of size 1
        feature_map = self.base_cnn(x)
        # feature_map is now a (D x N) tensor
        y = self.netvlad_layer(feature_map)
        return y

    def backward(self, dx):
        return dx
