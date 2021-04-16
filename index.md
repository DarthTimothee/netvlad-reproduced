# NetVLAD Reproduction Project
_Elwin Duinkerken & Timo Verlaan_

![NetVLAD banner image](/netvlad-banner.png)


## Introduction

In this article we will discuss the results and progress we made during the reproduction project for the `CS4240 Deep Learning` course. We reproduced the paper `NetVLAD: CNN architecture for weakly supervised place recognition`, in which an end-to-end trainable layer for image retrieval is proposed.

We will start by describing the problem the paper attempts to solve and their proposed solution. After that we will describe our implementation of this solution and support some of the design choices we made. Next, we discuss the experiments we ran, the results we got and compare them to the original results of the paper. Finally we draw some conclusions on the reproducibility of this paper and reflect on our development process.

## Problem description

The problem the NetVlad paper tries to solve, is to determine the location of a given image. A query image with unknown location is compared to all images from a dataset where the location is known. By matching the query image to the correct image from the dataset, the location of the query can be estimated.
The difficulty of this problem is that 2 photos taken in the same location might look very different based on factor such as the time it was taken, how many background objects there are, or slight changes in viewpoints.

## Solution

Before we describe the proposed solution in a more in-depth fashion, we summarize it as follows: Features of a query image are first extracted, using a base network like AlexNet, or VGG16. These are then fed through a new kind of layer, the NetVLAD layer, which assigns the features to cluster centers. The proximity to each these cluster centers is then used to build a fixed-sized feature representation vector, the VLAD vector, of the image that can be used to compare it to images. If their VLAD vectors are very similar, the images are likely to be taken a at (geographically) close location. In this section we describe the proposed NetVLAD layer, and how it can be trained and evaluated.


### NetVLAD layer

In the NetVLAD layer, we want to assign each input feature that comes out of the base network to one of the predefined cluster centers. More specifically, we use a convolution layer, followed by a softmax layer layer to calculate the soft-assignment $\bar{a}_k(x_i)$ of each of the features $x_i$ to each of the $k$-th cluster center. We then use this soft-assignment to weigh the difference between each of the features and clusters as follows (equation 1 from the original paper):

![equation 1](/netvlad-eqn1.gif)

where $N$ is the number of features from the base network, which depends on the input image resolution. $a_k$ is the soft assignment, $x_i$ is a specific input feature and $c_k$ is the $k$'th cluster center. The result is a $(K x D)$ vector $V$, the VLAD vector, where $K$ is the chosen number of clusters, and $D$ is the number of output channels of the base network. This VLAD vector is then L2-normalized in a column-wise fashion (which the paper refers to as intra-normalization), flattened into a vector of length $K\dot D$ and then L2-normalized in its entirety. The VLAD layer is shown schematically in the figure below (image credits to the original paper):

![NetVLAD layer image](/netvlad-fig2.png)


### Triplet ranking loss

In order to be able to train the proposed layer in an end-to-end fashion, the paper suggests to use training tuples. Each training tuple consists of a query image $q$, the best matching positive image $p_{i*}^q$ that lies within $10$ meters of the query image (geographically) and a couple of negative images \{q_j\}, which are further than $25$ meters away from the query image. We want to train the NetVLAD layer to assign features to clusters in such a way that the distance between a query and the best positive $d(q, p)$ is always less (by some margin $m$) than the distance between the query and any of the negative images $d(q,n)$. To this end, the paper proposes a weakly supervised ranking loss $L_\theta$, which is defined as follows (equation 7 in the original paper):

$$
L_\theta = \sum_j l(\min_i{d_\theta^2(q, p_i)} + m - d^2_\theta(q, n_j^q))
$$

In this equation, the $l(x)$ denotes the hinge loss $l(x)=\max(x, 0)$. This is the loss for one training tuple, so during a training epoch, we want to minimize the sum of all such losses over the entire train dataset.

### Recall@N accuracy

In order to be able to evaluate the proposed network we need a way to measure its accuracy. To this end the recall@N accuracy is proposed. In this particular application, the recall@N accuracy is the percentage of queries for which at least one out of the top N results from the database (VLAD vectors with the smallest difference) is within $25$ meters of the query image.

## Implementation

### Base network

We use 2 different models for the base network, Alexnet and VGG-16. For both models, we use a pretrained version. The last layers after the last convolution layer are removed since they will be replaced with the NetVLAD pooling layer. We only train the last convolution layer of the base network. The paper didn't exactly specify how many layers they trained to get their result, but it does state that the largest improvements are thanks to training the NetVLAD layer, and almost no improvement by training more layers other than the last one.

### Pooling layers

#### fmax

We implemented 2 different types of pooling, the NetVLAD layer as described in the paper and fmax pooling.
For the fmax pooling, we used the AdaptiveMaxPool2d from pytorch in order to specify the output dimension. Initialy, the output dimension were hardcoded to (1,1), because the resulting accuracy seemed to match the one in the paper. In the experiments, we also look at what happens with different dimensions.

#### NetVLAD

The NetVLAD convolution layer initializes the weights and biases using an alpha parameter. According to the paper, this value should be computed so that the ratio of the largest and the second largest soft assignment weight is on average equal to 100. Since we were unsure on how to implement this, we instead looked at the effect of different values for alpha and picked the best one as the final value to use.

To implement the VLAD vector calculation in pytorch, we restructed equation 1 as follows:

$$
V(j,k) = \sum_{i=1}^N a_k(x_i)(x_i(j) - c_k(j))
$$
$$
= \sum_{i=1}^N ( a_k(x_i) x_i(j) - a_k(x_i) c_k(j) )
$$
$$
= \sum_{i=1}^N a_k(x_i) x_i(j) - c_k(j) \sum_{i=1}^N a_k(x_i)
$$

The last sum over the soft assignments can then be implemented using `torch.sum` turning the second half of the expression into a vector-scalar product. The first half is calculated using `torch.bmm` the built-in tensor operation for batch-wise matrix multiplication. By reshaping the a_bar and x before putting them into the equation, we allow the calculation to be carried out for all the features at once, instead of having to loop over the indices `j` and `k` in `V`. The final (simplified) VLAD core calculation is:

```python
V = torch.bmm(a_bar, x) - c * torch.sum(a_bar)
```

where `a_bar` is the soft-assignment and c are the cluster centers.

### Database

Each image in the database needs to be preprocessed and converted to a tensor before it can be used as input to the network. This processing takes a bit of time, especially when the images are stored on a harddisk instead of a ssd. Therefore, we compute the corresponding tensor for each image once and save the result to a single file on disk. This reduces the total time it takes to run the model significantly.

```python
if path.exists(query_filename):
    print("Using preprocessed data", r)
    self.query_tensors = np.memmap(query_filename, ...)
    self.image_tensors = np.memmap(image_filename, ...)
```

In order to calculate the loss for each query-image, it requires the VLAD-vector of at least 1000 other database images. For the training dataset, we compute these vectors regularly and store them in a cache which is updated after a certain number of queries. For the test set, the cache is only updated once every epoch, because the network doesn't change during the test phase.

```python
net.freeze()
test_database.update_cache(net)

with torch.no_grad():
    # iterate through batches
    for q_vlad, p_vlad, n_vlads in test_loader:
        loss = sum([criterion(q_vlad, p_vlad, n_vlad) for n_vlad in n_vlads])
```

The loss is calculated with the TripletMarginWithDistanceLoss class using a custom distance function. There is also a TripletMarginLoss class which we first considered to use, but the problem is that it doesn't use the squared distances.

```python
def distance(x1, x2):
    return torch.pairwise_distance(x1, x2) ** 2

criterion = nn.TripletMarginWithDistanceLoss(distance_function=distance,
                                             margin=m, reduction='sum')
```

## Experiments

In this section we describe our test setup and show the final results of our implementation. We conducted several experiments with different base networks and pooling strategies, which we compare to the results in the paper.

### Test setup

For the training of the network we used the Pitts30k dataset, which contains 30k street view photos from Pittsburgh. The original paper used the Pitt250k dataset, but the training on this full dataset required a lot more time to complete. Therefore we decided to use the smaller dataset instead.

All our results use the following hyperparameters:

| Hyperparameter  | Value                      |
|-----------------|----------------------------|
| optimizer       | Adam                       |
| learning rate   | Alexnet = 0.0001, VGG16 = 0.00001 |
| weight decay    | 0.01                       |
| margin          | 0.1                        |
| cluster samples | 1000                       |
| image scale     | 224                        |
| alpha           | 0.1                        |
| K               | 64                         |


## Plaatje voor de results

![Figure with our results](/fig5a.png)



![NetVLAD banner image](/netvlad-banner.png)