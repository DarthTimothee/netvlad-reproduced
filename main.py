import gc
import os
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
from vlataset import Vlataset

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
    correct = 0
    total = 0

    # iterate through batches
    with progress(train_loader, position=0,
                  leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.BLUE)) as t:
        t.set_description("Training epoch " + str(epoch) + "\t\t\t")
        for training_tuple in t:
            # print(f"===== batch {i + 1} / {len(train_loader)} =====")
            t.set_postfix(ram_usage=ram_usage())

            # current_tensors()

            query_id, input_image, best_positive, hard_negatives = training_tuple

            # print(input_image.shape)
            # print(best_positive.shape)
            # print(hard_negatives.shape)

            # big_batch = torch.cat([input_image, best_positive, *hard_negatives])
            # net(big_batch)
            # continue

            if use_tensorboard:
                writer.add_graph(net, input_image)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # image = database.get_image(query)

            outputs = net(input_image)

            best_positive_vlad = net(best_positive)
            del best_positive
            loss = 0
            for j, n in enumerate(hard_negatives):
                # print(f"for negative {j} / {len(hard_negatives)}:")
                # ram_usage()

                n_vlad = net(n)
                loss += criterion(outputs, best_positive_vlad, n_vlad)

                del n_vlad
                del n

            del best_positive_vlad
            del outputs

            gc.collect()

            loss.backward()
            optimizer.step()

            # TODO: keep track of k-of-n 'correct'
            # keep track of loss and accuracy
            total_loss += loss
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    return total_loss / len(train_loader), 0  # 100 * correct / total


def test(epoch, test_loader, net, criterion):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    total_loss = 0
    correct = 0
    total = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        with progress(test_loader, position=0,
                      leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.CYAN)) as t:
            t.set_description("Testing epoch " + str(epoch) + "\t\t\t\t")
            for training_tuple in t:
                # print(f"===== batch {i + 1} / {len(train_loader)} =====")
                t.set_postfix(ram_usage=ram_usage())

                query_id, input_image, best_positive, hard_negatives = training_tuple

                # forward pass
                outputs = net(input_image)

                best_positive_vlad = net(best_positive)
                loss = 0
                for n in hard_negatives:
                    n_vlad = net(n)
                    loss += criterion(outputs, best_positive_vlad, n_vlad)

                # TODO: keep track of loss and accuracy
                total_loss += loss
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

    return total_loss / len(test_loader), 0  # 100 * correct / total


if __name__ == '__main__':
    # Create a writer to write to Tensorboard
    if use_tensorboard:
        writer = SummaryWriter()

    timos_arme_laptopje = False
    if timos_arme_laptopje:
        torch.set_num_threads(4)

    # Hyper parameters, based on the appendix
    K = 64  # amount of kernels
    m = 0.1  # margin for the loss
    lr = 0.001  # or 0.0001 depending on the experiment, which is halved every 5 epochs
    momentum = 0.9
    wd = 0.001
    batch_size = 4  # TODO: batch size is 4 tuples
    epochs = 2  # TODO: 32 but usually convergence occurs much faster

    # Create instance of Network
    base_network = AlexBase()
    D = base_network.get_output_dim()

    # Create loss function and optimizer
    pairwise_distance = nn.PairwiseDistance()
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=custom_distance, margin=m, reduction='sum')
    # loss_function2 = nn.TripletMarginLoss(margin=m ** 0.5, reduction='sum')
    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss

    train_database = Database('./datasets/pitts30k_train.mat')  # , dataset_url='./data/')
    test_database = Database('./datasets/pitts30k_test.mat')  # , dataset_url='./data/')
    train_set = Vlataset(train_database)
    test_set = Vlataset(test_database)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    net = NetVladCNN(base_cnn=base_network, K=K)
    optimizer = optim.SGD(net.parameters(), lr=5e-1)

    if use_torch_summary:
        summary(net, (3, 480, 640))

    net.init_clusters(train_database, num_samples=100)  # TODO: don't forget to change num_samples back!

    for epoch in range(epochs):  # loop over the dataset multiple times
        # Train on data
        train_database.update_cache(net)
        train_loss, train_acc = train(epoch, train_loader, net, optimizer, criterion)

        # Test on data
        test_database.update_cache(net)
        test_loss, test_acc = test(epoch, test_loader, net, criterion)

        # Write metrics to Tensorboard
        if use_tensorboard:
            writer.add_scalars("Loss", {'Train': train_loss, 'Test': test_loss}, epoch)
            writer.add_scalars('Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)

    print('Finished Training')

    if use_tensorboard:
        writer.flush()
        writer.close()
