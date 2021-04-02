import torch
from torch import nn, optim
from colorama import Fore
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.utils.data import DataLoader
from NetVladCNN import NetVladCNN, AlexBase, L2Norm, Reshape
from database import Database
from helpers import *
from validation import validate
from vlataset import Vlataset, VlataTest

max_cache_lifetime = 1000
cache_lifetime = 0


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
    total_count = 0
    global cache_lifetime

    net.unfreeze()

    # iterate through batches
    t = pbar(train_loader, color=Fore.BLUE, desc=f"Training epoch {epoch}", smoothing=0)
    t.set_postfix(ram_usage=ram_usage())
    for q_tensor, p_tensor, n_tensors in t:
        if cache_lifetime > max_cache_lifetime:
            train_loader.dataset.database.update_cache(net, t_parent=t)
            cache_lifetime = 0

        # zero the parameter gradients
        optimizer.zero_grad()

        q_vlad, p_vlad, n_vlads = net(q_tensor), net(p_tensor), [net(h) for h in n_tensors]
        loss = sum([criterion(q_vlad, p_vlad, n_vlad) for n_vlad in n_vlads])

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().numpy()

        cache_lifetime += batch_size
        total_count += batch_size
        avg_loss = total_loss / total_count
        t.set_postfix(ram_usage=ram_usage(), loss=avg_loss)

    return avg_loss


def test(epoch, test_loader, net, criterion):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """
    total_loss = 0
    total_count = 0
    test_database.update_cache(net)

    net.freeze()

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        t = pbar(test_loader, color=Fore.CYAN, desc=f"Testing epoch {epoch}")
        t.set_postfix(ram_usage=ram_usage())
        for q_vlad, p_vlad, n_vlads in t:

            loss = sum([criterion(q_vlad, p_vlad, n_vlad) for n_vlad in n_vlads])

            total_loss += loss.detach().numpy()
            total_count += batch_size
            avg_loss = total_loss / total_count

            t.set_postfix(ram_usage=ram_usage(), loss=avg_loss)

    # Calculate recall@N accuracies
    accs = validate(net, test_database)
    net.unfreeze()

    return avg_loss, accs


if __name__ == '__main__':
    # Create a writer to write to Tensorboard
    writer = SummaryWriter()

    # Hyper parameters, based on the appendix
    m = 0.1  # margin for the loss
    lr = 0.001  # or 0.0001 depending on the experiment, which is halved every 5 epochs
    momentum = 0.9
    wd = 0.001
    batch_size = 4
    epochs = 30
    # torch.set_num_threads(4)

    # Create instance of Network
    # base_network = VGG16()  # AlexBase()
    base_network = AlexBase()
    D = base_network.get_output_dim()

    # Create loss function and optimizer
    pairwise_distance = nn.PairwiseDistance()
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=custom_distance, margin=m, reduction='sum')
    # loss_function2 = nn.TripletMarginLoss(margin=m ** 0.5, reduction='sum')
    # TODO: try with other loss function
    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss

    using_vlad = True
    if using_vlad:
        pooling_layer = None
        K = 64
    else:
        # TODO: over what dimension should we normalize in the L2Norm layer?
        # TODO: not 1x1 but other shape -> but what shape?
        pooling_layer = nn.Sequential(nn.MaxPool2d((1, 1)), Reshape(), L2Norm())
        K = 1

    net = NetVladCNN(base_cnn=base_network, pooling_layer=pooling_layer, K=K)
    path = None  # TODO: import net from file
    if path:
        net.load_state_dict(torch.load(path))

    optimizer = optim.SGD(net.parameters(), lr=lr)

    summary(net, (3, 480, 640))

    try:
        train_database = Database('./datasets/pitts30k_train.mat')
        test_database = Database('./datasets/pitts30k_test.mat')
    except:
        train_database = Database('./datasets/pitts30k_train.mat', dataset_url='./data/')
        test_database = Database('./datasets/pitts30k_test.mat', dataset_url='./data/')

    train_set = Vlataset(train_database)
    test_set = VlataTest(test_database)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2)

    # Get the N by passing a random image from the dataset though the network
    sample_out = base_network(train_database.query_tensor_from_stash(0).unsqueeze(0))
    _, _, W, H = sample_out.shape
    N = W * H

    writer.add_graph(net, train_database.query_tensor_from_stash(0).unsqueeze(0))

    net.init_clusters(train_database, N=N, num_samples=1000)
    train_database.update_cache(net)

    for epoch in range(1, epochs + 1):
        if epoch > 1 and epoch % 5 == 1:  # 1 because the epochs are 1-indexed
            max_cache_lifetime *= 2
            lr /= 2.0
            optimizer.learning_rate = lr

        # Train on data
        train_loss = train(epoch, train_loader, net, optimizer, criterion)

        # Calculate loss and recall@N accuracies with test set  TODO: validation set
        test_loss, accuracies = test(epoch, test_loader, net, criterion)

        # Write metrics to Tensorboard and save the model
        writer.add_scalars("Loss", {'Train': train_loss, 'Test': test_loss}, epoch)
        writer.add_scalars("Recall@N",
                           {'1@N': accuracies[0], '2@N': accuracies[1], '3@N': accuracies[2], '4@N': accuracies[3],
                            '5@N': accuracies[4], '10@N': accuracies[5], '15@N': accuracies[6],
                            '20@N': accuracies[7], '25@N': accuracies[-1]}, epoch)
        writer.flush()
        torch.save(net.state_dict(), "./nets/net-" + str(epoch))

    # Finalize
    print('Finished Training')
    writer.flush()
    writer.close()

    # TODO: test model with lowest lost (or highest accuracy?) on test set
