from torch import nn, optim, cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from NetVLAD import AlexBase, L2Norm, Reshape, NetVLAD, FullNetwork
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
    avg_loss = 0
    global cache_lifetime

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
        total_loss += loss.cpu().detach().numpy()

        cache_lifetime += batch_size
        total_count += batch_size
        avg_loss = total_loss / total_count
        t.set_postfix(ram_usage=ram_usage(), loss=avg_loss)

    accs = validate(net, train_database)
    return avg_loss, accs


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
    avg_loss = 0
    test_database.update_cache(net)

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
    num_cluster_samples = 1000
    image_resolution = 224
    preprocessing_mode = 'disk'  # should be 'ram' or 'disk'.
    # torch.set_num_threads(4)

    # Create the databases + datasets + dataloaders
    if cuda.is_available():
        device = torch.device('cuda')
        train_database = Database('./datasets/pitts30k_train.mat', dataset_url='./data/',
                                  image_resolution=image_resolution, preprocess_mode=preprocessing_mode)
        test_database = Database('./datasets/pitts30k_test.mat', dataset_url='./data/',
                                 image_resolution=image_resolution, preprocess_mode=preprocessing_mode)
        train_set = Vlataset(train_database)
        test_set = VlataTest(test_database)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=3, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=3, pin_memory=True)
    else:
        device = torch.device('cpu')
        train_database = Database('./datasets/pitts30k_train.mat', image_resolution=image_resolution,
                                  preprocess_mode=preprocessing_mode)
        test_database = Database('./datasets/pitts30k_test.mat', image_resolution=image_resolution,
                                 preprocess_mode=preprocessing_mode)
        train_set = Vlataset(train_database)
        test_set = VlataTest(test_database)
        train_loader = DataLoader(train_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)

    # Create instance of Network
    # base_network = VGG16()  # AlexBase()
    base_network = AlexBase().cuda()
    D = base_network.get_output_dim()

    # Get the N by passing a random image from the dataset though the network
    sample_out = base_network(train_database.query_tensor_from_stash(0).unsqueeze(0))
    _, _, W, H = sample_out.shape
    N = W * H

    # Create loss function and optimizer
    pairwise_distance = nn.PairwiseDistance()
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=custom_distance, margin=m, reduction='sum').cuda()
    # loss_function2 = nn.TripletMarginLoss(margin=m ** 0.5, reduction='sum')
    # TODO: try with other loss function
    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss

    # Specify the type of pooling to use
    using_vlad = True
    if using_vlad:
        K = 64
        pooling_layer = NetVLAD(K=K, D=D, cluster_database=train_database, base_cnn=base_network, N=N,
                                cluster_samples=num_cluster_samples)
    else:
        # TODO: over what dimension should we normalize in the L2Norm layer?
        # TODO: not 1x1 but other shape -> but what shape?
        # K = N
        # pooling_layer = nn.Sequential(nn.MaxPool2d((1, 1)), Reshape(), L2Norm())
        K = 1
        pooling_layer = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)), Reshape(), L2Norm())

    # Create the full net
    net = FullNetwork(K, D, base_network, pooling_layer).cuda()

    path = None  # TODO: import net from file
    if path:
        net.load_state_dict(torch.load(path))
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # Write the architecture of the net
    summary(net, (3, 480, 640), device='cuda' if cuda.is_available() else 'cpu')
    writer.add_graph(net, train_database.query_tensor_from_stash(0).unsqueeze(0))

    # Init the cache
    train_database.update_cache(net)

    for epoch in range(1, epochs + 1):
        if epoch > 1 and epoch % 5 == 1:  # 1 because the epochs are 1-indexed
            max_cache_lifetime *= 2
            lr /= 2.0
            optimizer.learning_rate = lr

        # Train on data
        train_loss, accs = train(epoch, train_loader, net, optimizer, criterion)
        print(f"Train loss: {train_loss}, Accuracy: {accs}")
        write_accs("TrainRecall@N", accs, writer, epoch)
        write_loss('Train', train_loss, writer, epoch)
        writer.flush()

        # Calculate loss and recall@N accuracies with test set  TODO: validation set
        test_loss, accs = test(epoch, test_loader, net, criterion)
        print(f"Train loss: {test_loss}, Accuracy: {accs}")
        write_accs("TestRecall@N", accs, writer, epoch)
        write_loss('Test', test_loss, writer, epoch)
        writer.flush()

        torch.save(net.state_dict(), "./nets/net-" + str(epoch))

    # Finalize
    print('Finished Training')
    writer.flush()
    writer.close()

    # TODO: test model with lowest lost (or highest accuracy?) on test set
