
## Introduction

In this article we will discuss the results and progress we made during the reproduction project for the CS4240 Deep Learning course. We reproduced the paper NetVLAD: CNN architecture for weakly supervised place recognition, in which an end-to-end trainable layer for image retrieval is proposed.

We will start by describing the problem the paper attempts to solve and their proposed solution. After that we will describe our implementation of this solution and support some of the design choices we made. Next, we discuss the experiments we ran, the results we got and compare them to the original results of the paper. Finally we draw some conclusions on the reproducibility of this paper and reflect on our development process.

======

## Problem description

The problem the NetVlad paper tries to solve, is to determine the location of a given image. A query image with unknown location is compared to all images from a dataset where the location is known. By matching the query image to the correct image from the dataset, the location of the query can be estimated.
The difficulty of this problem is that 2 photos taken in the same location might look very different based on factor such as the time it was taken, how many background objects there are, or slight changes in viewpoints.

======

## Solution

Before we describe the proposed solution in a more in-depth fashion, we summarize it as follows: Features of a query image are first extracted, using a base network like AlexNet, or VGG16. These are then fed through a new kind of layer, the NetVLAD layer, which assigns the features to cluster centers. The proximity to each these cluster centers is then used to build a fixed-sized feature representation vector, the VLAD vector, of the image that can be used to compare it to images. If their VLAD vectors are very similar, the images are likely to be taken a at (geographically) close location. In this section we describe the proposed NetVLAD layer, and it can be trained and evaluated.

### NetVLAD layer

In the NetVLAD layer, we want to assign each input feature that comes out of the base network to one of the predefined cluster centers. More specifically, we use a convolution layer, followed by a softmax layer layer to calculate the soft-assignment $\bar{a}_k(x_i)$ of each of the features $x_i$ to each of the $k$-th cluster center. We then use this soft-assignment to weigh the difference between each of the features and clusters as follows (equation 1 from the original paper):

![equation 1](/netvlad-eqn1.gif)

where $N$ is the number of features from the base network, which depends on the input image resolution. $a_k$ is the soft assignment, $x_i$ is a specific input feature and $c_k$ is the $k$'th cluster center. The result is a $(K x D)$ vector $V$, the VLAD vector, where $K$ is the chosen number of clusters, and $D$ is the number of output channels of the base network. This VLAD vector is then L2-normalized in a column-wise fashion (which the paper refers to as intra-normalization), flattened into a vector of length $K\dot D$ and then L2-normalized in its entirety.

### Triplet ranking loss

In order to be able to train the proposed layer in an end-to-end fashion, the paper suggests to use training tuples. Each training tuple consists of a query image $q$, the best matching positive image $p_{i*}^q$ that lies within $10$ meters of the query image (geographically) and a couple of negative images \{q_j\}, which are further than $25$ meters away from the query image. We want to train the NetVLAD layer to assign features to clusters in such a way that the distance between a query and the best positive $d(q, p)$ is always less (by some margin $m$) than the distance between the query and any of the negative images $d(q,n)$. To this end, the paper proposes a weakly supervised ranking loss $L_\theta$, which is defined as follows (equation 7 in the original paper):

$$
L_\theta = \sum_j l(\min_i{d_\theta^2(q, p_i)} + m - d^2_\theta(q, n_j^q))
$$

In this equation, the $l(x)$ denotes the hinge loss $l(x)=\max(x, 0)$. This is the loss for one training tuple, so during a training epoch, we want to minimize the sum of all such losses over the entire train dataset.

### Recall@N accuracy

In order to be able to evaluate the proposed network we need a way to measure its accuracy. To this end the recall@N accuracy is proposed. In this particular application, the recall@N accuracy is the percentage of queries for which at least one out of the top N results from the database (VLAD vectors with the smallest difference) is within $25$ meters of the query image.

## Implementation

======

### Base network

We use 2 different models for the base network, Alexnet and VGG-16. For both models, we use a pretrained version. The last layers after the last convolution layer are removed since they will be replaced with the NetVLAD pooling layer. We only train the last convolution layer of the base network. The paper didn't exactly specify how many layers they trained to get their result, but they do say the largest improvements are thanks to training the NetVLAD layer, and almost no improvement by training more layers other than the last one.

```python
# Setup base network
self.full_cnn = models.alexnet(pretrained=True, progress=True)
self.features = nn.Sequential(*self.full_cnn.features[:-2])
for param in self.features.parameters():
    param.requires_grad = False
for param in self.features[-1].parameters():
    param.requires_grad = True
```

### Pooling layers

#### fmax

We implemented 2 different types of pooling, the NetVLAD layer as described in the paper and fmax pooling.
For the fmax pooling, we used the AdaptiveMaxPool2d from pytorch in order to specify the output dimension. Initialy, the output dimension were hardcoded to (1,1), because the resulting accuracy seemed to match the one in the paper. In the experiments, we also look at what happen with different dimensions.

#### NetVLAD

The NetVLAD convolution layer initializes the weights and biases using an alpha parameter. According to the paper, this value should be computed so that the ratio of the largest and the second largest soft assignment weight is on average equal to 100. Since we were unsure on how to implement this, we instead looked at the effect of different values for alpha and picked the best one as the final value to use.