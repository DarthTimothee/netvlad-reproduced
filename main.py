import sys
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from NetVLAD import AlexBase, NetVLAD, FullNetwork
from database import Database
from helpers import *
from validation import validate
from vlataset import Vlataset, VlataTest

max_cache_lifetime = 1000
cache_lifetime = 0


def train(epoch, train_loader, net, optimizer, criterion):
    total_loss = 0
    total_count = 0
    avg_loss = 0
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
        total_loss += loss.cpu().detach().numpy()

        cache_lifetime += batch_size
        total_count += batch_size
        avg_loss = total_loss / total_count
        t.set_postfix(ram_usage=ram_usage(), loss=avg_loss)

    accs = validate(net, train_database)
    return avg_loss, accs


def test(epoch, test_loader, net, criterion):
    total_loss = 0
    total_count = 0
    avg_loss = 0

    net.freeze()
    test_database.update_cache(net)

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

    # Data path can be passed as the first argument to the program, otherwise the default is used
    DATA_PATH = 'G:/School/Deep Learning/data/' if len(sys.argv) < 2 else sys.argv[1]

    # Hyper parameters, based on the appendix
    m = 0.1  # margin for the loss
    lr = 0.001  # or 0.0001 depending on the experiment, which is halved every 5 epochs
    momentum = 0.9
    wd = 0.001
    batch_size = 4
    epochs = 30
    num_cluster_samples = 1000
    input_scale = 224
    preprocessing_mode = 'disk'
    assert preprocessing_mode in ['disk', 'ram', None]
    save_model = True
    load_model_path = None  # TODO: import net from file
    # torch.set_num_threads(4)

    # Create the databases + datasets + dataloaders
    train_database = Database(data_path=DATA_PATH, database='pitts30k_train', input_scale=input_scale,
                              preprocess_mode=preprocessing_mode)
    test_database = Database(data_path=DATA_PATH, database='pitts30k_val', input_scale=input_scale,
                             preprocess_mode=preprocessing_mode)
    train_set = Vlataset(train_database)
    test_set = VlataTest(test_database)

    device = get_device()
    if cuda.is_available():
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=3, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=3, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)

    # Create instance of Network
    # base_network = VGG16()  # AlexBase()
    base_network = AlexBase().to(device)

    # Get the N by passing a random image from the dataset though the network
    sample_out = base_network(train_database.get_query_tensor(0))
    _, _, W, H = sample_out.shape
    N = W * H

    # Create loss function and optimizer
    pairwise_distance = nn.PairwiseDistance()
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=custom_distance, margin=m, reduction='sum').to(device)
    # criterion = nn.TripletMarginLoss(margin=m ** 0.5, reduction='sum').to(device)

    # Specify the type of pooling to use
    pooling_layer = NetVLAD(K=64, N=N, cluster_database=train_database, base_cnn=base_network, cluster_samples=num_cluster_samples)
    # pooling_layer = nn.AdaptiveMaxPool2d((1, 1))

    # Create the full net
    net = FullNetwork(features=base_network, pooling=pooling_layer).to(device)

    if load_model_path:
        net.load_state_dict(torch.load(load_model_path))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                net.parameters()), lr=lr, momentum=momentum, weight_decay=wd)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    # Write the architecture of the net
    summary(net, (3, 480, 640), device='cuda' if cuda.is_available() else 'cpu')
    writer.add_graph(net, train_database.get_query_tensor(0))

    # Init the cache
    train_database.update_cache(net)

    for epoch in range(1, epochs + 1):
        if epoch > 1 and epoch % 5 == 1:  # 1 because the epochs are 1-indexed
            max_cache_lifetime *= 2
            lr /= 2.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # optimizer.learning_rate = lr

        # Train on data
        train_loss, train_accs = train(epoch, train_loader, net, optimizer, criterion)
        print(f"Train loss: {train_loss}, Accuracy: {train_accs}")
        write_accs("TrainRecall@N", train_accs, writer, epoch)
        write_loss('Train', train_loss, writer, epoch)
        writer.flush()

        # Calculate loss and recall@N accuracies with validations
        test_loss, test_accs = test(epoch, test_loader, net, criterion)
        print(f"Test loss: {test_loss}, Accuracy: {test_accs}")
        write_accs("TestRecall@N", test_accs, writer, epoch)
        write_loss('Test', test_loss, writer, epoch)
        writer.flush()

        if save_model:
            torch.save(net.state_dict(), "./nets/net-" + str(epoch))

    # Finalize
    print('Finished Training')
    writer.flush()
    writer.close()

    # TODO: test model with highest accuracy on test set
