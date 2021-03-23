import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm as progress
from NetVladCNN import NetVladCNN
from database import Vlataset
import numpy as np


def train(train_loader, net, optimizer, criterion):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # iterate through batches
    for training_tuple in progress(train_loader):

        input_image, query, best_positive, hard_negatives = training_tuple

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # image = database.get_image(query)
        K = 64  # amount of kernels
        D = 256
        c = np.zeros((K, D))  # TODO: get actual c (parameter) used zeros for now

        outputs = net(input_image, c)

        best_positive_vlad = net(best_positive, c)
        loss = 0
        for n in hard_negatives:
            n_vlad = net(n, c)
            loss += criterion(outputs, best_positive_vlad, n_vlad)
        # loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # TODO: keep track of k-of-n 'correct'
        # keep track of loss and accuracy
        # avg_loss += loss
        # _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 0  # 100 * correct / total


def test(test_loader, net, criterion):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        for i, training_tuple in enumerate(test_loader):

            input_image, query, best_positive, hard_negatives = training_tuple

            K = 64  # amount of kernels
            D = 256
            c = np.zeros((K, D))  # TODO: get actual c (parameter) used zeros for now

            # forward pass
            outputs = net(input_image, c)

            best_positive_vlad = net(best_positive, c)
            loss = 0
            for n in hard_negatives:
                n_vlad = net(n, c)
                loss += criterion(outputs, best_positive_vlad, n_vlad)

            # TODO: keep track of loss and accuracy
            # avg_loss += loss
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    return avg_loss / len(test_loader), 0  # 100 * correct / total


def alex_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.features(x)


if __name__ == '__main__':
    # Create a writer to write to Tensorboard
    # writer = SummaryWriter()

    # Hyper parameters, based on the appendix
    K = 64  # amount of kernels
    m = 0.1  # margin for the loss
    lr = 0.001  # or 0.0001 depending on the experiment, which is halved every 5 epochs
    momentum = 0.9
    wd = 0.001
    batch_size = 1  # TODO: batch size is 4 tuples
    epochs = 2  # but usually convergence occurs much faster

    # Setup base network
    alex_base = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    alex_base.features = alex_base.features[:11]  # cut off before last ReLu
    alex_base.avgpool = nn.Identity()  # bypass avg_pool and classifier
    alex_base.classifier = nn.Identity()  # bypass avg_pool and classifier
    alex_base.forward = type(alex_base.forward)(alex_forward,
                                                alex_base)  # TODO: find a better way to override this function. Maybe by extending the AlexNet module?

    # Create instance of Network
    net = NetVladCNN(base_cnn=alex_base, K=K)


    def custom_distance(x1, x2):
        return pairwise_distance(x1, x2) ** 2


    # Create loss function and optimizer
    pairwise_distance = nn.PairwiseDistance()
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=custom_distance, margin=m, reduction='sum')
    # loss_function2 = nn.TripletMarginLoss(margin=m ** 0.5, reduction='sum')
    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss

    optimizer = optim.SGD(net.parameters(), lr=5e-1)

    train_set = Vlataset(database_url='./datasets/pitts250k_train.mat')
    test_set = Vlataset(database_url='./datasets/pitts250k_test.mat')
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    for epoch in progress(range(epochs)):  # loop over the dataset multiple times
        # Train on data
        train_loss, train_acc = train(train_loader, net, optimizer, criterion)

        # Test on data
        test_loss, test_acc = test(test_loader, net, criterion)

        # Write metrics to Tensorboard
        # writer.add_scalars("Loss", {'Train': train_loss, 'Test': test_loss}, epoch)
        # writer.add_scalars('Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)

    print('Finished Training')
    # writer.flush()
    # writer.close()
