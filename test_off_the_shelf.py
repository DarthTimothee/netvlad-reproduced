from torch import cuda, nn

from NetVLAD import AlexBase, FullNetwork, L2Norm
from database import Database
from helpers import get_device
from validation import validate

device = get_device()

# Hacky data path:
DATA_PATH = '../pittsburgh250k' if cuda.is_available() else 'G:/School/Deep Learning/data/'

train_database = Database(data_path=DATA_PATH, database='pitts30k_val', input_scale=224, preprocess_mode='disk')
# train_set = Vlataset(train_database)
# train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=3)
base_network = AlexBase().to(device)
D = base_network.get_output_dim()

# Get the N by passing a random image from the dataset though the network
sample_out = base_network(train_database.get_query_tensor(0))
_, _, W, H = sample_out.shape
N = W * H

# Specify the type of pooling to use
# pooling_layer = NetVLAD(K=64, N=N, cluster_database=train_database, base_cnn=base_network, cluster_samples=1000)
pooling_layer = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten(), L2Norm())

# Create the full net
net = FullNetwork(features=base_network, pooling=pooling_layer).to(device)

# Update the cache for the train validation set, and validate
train_database.cache.update(net, cache_batch_size=42)
print(validate(net, train_database))
