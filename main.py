import gc
import os
import time

import psutil

import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm as progress

from NetVladCNN import NetVladCNN, AlexBase
from database import Database
from vlataset import Vlataset, VlataTest

use_torch_summary = False
try:
    from torchsummary import summary

    use_torch_summary = True
except ImportError:
    pass

use_tensorboard = False
try:
    from torch.utils.tensorboard import SummaryWriter

    use_tensorboard = True
except ImportError:
    pass

process = psutil.Process(os.getpid())
max_cache_lifetime = 1000
cache_lifetime = 0


def ram_usage():
    return f"{process.memory_info().rss / 10 ** 9: .3} GB"


def current_tensors():
    # return
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def custom_distance(x1, x2):
    return pairwise_distance(x1, x2) ** 2


def train(epoch, train_loader, net, optimizer, criterion):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """
    total_loss = 0
    global cache_lifetime

    net.train()

    # iterate through batches
    with progress(train_loader, position=0, smoothing=0,
                  leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.BLUE)) as t:
        t.set_description(f'{"Training epoch " + str(epoch) : <32}')
        t.set_postfix(ram_usage=ram_usage())
        for training_tuple in t:
            q_tensor, p_tensor, n_tensors = training_tuple

            if cache_lifetime > max_cache_lifetime:
                train_loader.dataset.database.update_cache(net, t_parent=t)
                cache_lifetime = 0

            # zero the parameter gradients
            optimizer.zero_grad()

            q_vlad, p_vlad, n_vlads = net(q_tensor), net(p_tensor), [net(h) for h in n_tensors]
            del q_tensor, p_tensor, n_tensors

            loss = torch.zeros(1)
            for n_vlad in n_vlads:
                loss += criterion(q_vlad, p_vlad, n_vlad)

            loss.backward()
            optimizer.step()
            total_loss += loss.detach().numpy()

            cache_lifetime += batch_size
            t.set_postfix(ram_usage=ram_usage(), total_loss=total_loss)

    return total_loss / len(train_loader)


def test(epoch, test_loader, net, criterion):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """
    total_loss = 0

    net.eval()
    net.freeze()

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        with progress(test_loader, position=0,
                      leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.CYAN)) as t:
            t.set_description(f'{"Testing epoch " + str(epoch) : <32}')
            t.set_postfix(ram_usage=ram_usage())
            for training_tuple in t:

                q_vlad, p_vlad, n_vlads = training_tuple
                loss = torch.zeros(1)

                for n_vlad in n_vlads:
                    loss += criterion(q_vlad, p_vlad, n_vlad)

                total_loss += loss.detach().numpy()

                t.set_postfix(ram_usage=ram_usage(), total_loss=total_loss)

    net.unfreeze()
    return total_loss / len(test_loader)


if __name__ == '__main__':
    # Create a writer to write to Tensorboard
    if use_tensorboard:
        writer = SummaryWriter()

    # Hyper parameters, based on the appendix
    K = 64  # amount of kernels
    m = 0.1  # margin for the loss
    lr = 0.001  # or 0.0001 depending on the experiment, which is halved every 5 epochs
    momentum = 0.9
    wd = 0.001
    batch_size = 4
    epochs = 30
    # torch.set_num_threads(4)

    # Create instance of Network
    base_network = AlexBase()
    D = base_network.get_output_dim()

    # Create loss function and optimizer
    pairwise_distance = nn.PairwiseDistance()
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=custom_distance, margin=m, reduction='sum')
    # loss_function2 = nn.TripletMarginLoss(margin=m ** 0.5, reduction='sum')
    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss

    net = NetVladCNN(base_cnn=base_network, K=K)
    path = None  # TODO
    if path:
        net.load_state_dict(torch.load(path))

    optimizer = optim.SGD(net.parameters(), lr=5e-1)

    if use_torch_summary:
        summary(net, (3, 480, 640))

    train_database = Database('./datasets/pitts30k_train.mat', dataset_url='./data/')
    test_database = Database('./datasets/pitts30k_test.mat', dataset_url='./data/')
    train_set = Vlataset(train_database)
    test_set = VlataTest(test_database)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # if use_tensorboard:
    #     writer.add_graph(net, train_set[0])

    net.init_clusters(train_database, num_samples=1000)
    train_database.update_cache(net)

    for epoch in range(epochs):  # loop over the dataset multiple times

        if epoch > 0 and epoch % 5 == 0:
            max_cache_lifetime *= 2
            lr /= 2.0
            optimizer.learning_rate = lr

        # Train on data
        train_loss = train(epoch, train_loader, net, optimizer, criterion)

        # Test on data
        test_database.update_cache(net)
        test_loss = test(epoch, test_loader, net, criterion)

        # Write metrics to Tensorboard
        if use_tensorboard:
            writer.add_scalars("Loss", {'Train': train_loss, 'Test': test_loss}, epoch)

        torch.save(net.state_dict(), "./nets/net-" + str(epoch))

    print('Finished Training')

    if use_tensorboard:
        writer.flush()
        writer.close()
