from torch import nn

from NetVLAD import AlexBase, NetVLAD, Reshape, L2Norm, FullNetwork
from database import Database
from validation import validate

train_database = Database('./datasets/pitts30k_val.mat', dataset_url='./data/', image_resolution=224, preprocess_mode='disk')
# train_set = Vlataset(train_database)
# train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=3)
base_network = AlexBase().cuda()
D = base_network.get_output_dim()

# Get the N by passing a random image from the dataset though the network
sample_out = base_network(train_database.query_tensor_from_stash(0).unsqueeze(0))
_, _, W, H = sample_out.shape
N = W * H

alpha = 1000000

# Specify the type of pooling to use
using_vlad = False
if using_vlad:
    K = 64
    pooling_layer = NetVLAD(K=K, D=D, cluster_database=train_database, base_cnn=base_network, N=N, cluster_samples=1000, alpha=alpha)
else:
    # TODO: over what dimension should we normalize in the L2Norm layer?
    # TODO: not 1x1 but other shape -> but what shape?
    # K = N
    # pooling_layer = nn.Sequential(nn.MaxPool2d((1, 1)), Reshape(), L2Norm())
    K = 1
    pooling_layer = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)), Reshape(), L2Norm())

# Create the full net
net = FullNetwork(K, D, base_network, pooling_layer).cuda()
train_database.cache.update(net, cache_batch_size=42)

print(validate(net, train_database))
