# NetVLAD Reproduction Project
_Elwin Duinkerken & Timo Verlaan_

![NetVLAD banner image](/netvlad-banner.png)

{% include style.html %}

## Introduction

In this article we will discuss the results and progress we made during the reproduction project for the `CS4240 Deep Learning` course. We reproduced the paper `NetVLAD: CNN architecture for weakly supervised place recognition`, in which an end-to-end trainable layer for image retrieval is proposed.

We will start by describing the problem the paper attempts to solve and their proposed solution. After that we will describe our implementation of this solution and support some of the design choices we made. Next, we discuss the experiments we ran, the results we got and compare them to the original results of the paper. Finally we draw some conclusions on the reproducibility of this paper and reflect on our development process.

## Problem description

The problem the NetVlad paper tries to solve, is to determine the location of a given image. A query image with unknown location is compared to all images from a dataset where the location is known. By matching the query image to the correct image from the dataset, the location of the query can be estimated.
The difficulty of this problem is that 2 photos taken in the same location might look very different based on factor such as the time it was taken, how many background objects there are, or slight changes in viewpoints.

## Solution

Before we describe the proposed solution in a more in-depth fashion, we summarize it as follows: Features of a query image are first extracted, using a base network like AlexNet, or VGG16. These are then fed through a new kind of layer, the NetVLAD layer, which assigns the features to cluster centers. The proximity to each these cluster centers is then used to build a fixed-sized feature representation vector, the VLAD vector, of the image that can be used to compare it to images. If their VLAD vectors are very similar, the images are likely to be taken a at (geographically) close location. In this section we describe the proposed NetVLAD layer, and how it can be trained and evaluated.


### NetVLAD layer

In the NetVLAD layer, we want to assign each input feature that comes out of the base network to one of the predefined cluster centers. More specifically, we use a convolution layer, followed by a softmax layer layer to calculate the soft-assignment ![a_k(x_i)](/a_bar_k-x_i-.gif) of each of the features ![x_i](/x_i.gif) to each of the ![k](/k.gif)-th cluster center. We then use this soft-assignment to weigh the difference between each of the features and clusters as follows (equation 1 from the original paper):

![equation 1](/netvlad-eqn1.gif)

where ![N](/N.gif) is the number of features from the base network, which depends on the input image resolution. ![a_k](/a_k.gif) is the soft assignment, ![x_i](/x_i.gif) is a specific input feature and ![c_k](/c_k.gif) is the ![k](/k.gif)'th cluster center. The result is a ![(KxD)](/KxD.gif) vector ![V](/V.gif), the VLAD vector, where ![K](/K.gif) is the chosen number of clusters, and ![D](/D.gif) is the number of output channels of the base network. This VLAD vector is then L2-normalized in a column-wise fashion (which the paper refers to as intra-normalization), flattened into a vector of length ![K * D](/KcdotD.gif) and then L2-normalized in its entirety. The VLAD layer is shown schematically in the figure below (image credits to the original paper):

![NetVLAD layer image](/netvlad-fig2.png)


### Triplet ranking loss

In order to be able to train the proposed layer in an end-to-end fashion, the paper suggests to use training tuples. Each training tuple consists of a query image ![q](/q.gif), the best matching positive image ![p_i^q](/p_i^q.gif) that lies within 10 meters of the query image (geographically) and a couple of negative images ![q_j](/q_j.gif), which are further than 25 meters away from the query image. We want to train the NetVLAD layer to assign features to clusters in such a way that the distance between a query and the best positive ![d(q,p)](/d-q,p-.gif) is always less (by some margin ![m](/m.gif)) than the distance between the query and any of the negative images ![d(q,n)](/d-q,n-.gif). To this end, the paper proposes a weakly supervised ranking loss ![L](/Loss.gif), which is defined as follows (equation 7 in the original paper):

![Loss function](/Loss_function.gif)

In this equation, the ![l(x)](/l-x-.gif) denotes the hinge loss ![l(x)=max(x,0)](/hinge_loss.gif). This is the loss for one training tuple, so during a training epoch, we want to minimize the sum of all such losses over the entire train dataset.

### Recall@N accuracy

In order to be able to evaluate a network we need a way to measure its accuracy. To this end the recall@N accuracy is used. In this particular application, the recall@N accuracy is the percentage of queries for which at least one out of the top N results from the database (VLAD vectors with the smallest difference) is within 25 meters of the query image.

## Implementation

### Base network

We use 2 different models for the base network, Alexnet and VGG-16. For both models, we use a pretrained version. The last layers after the last convolution layer are removed since they will be replaced with the NetVLAD pooling layer. We only train the last convolution layer of the base network. The paper didn't exactly specify how many layers they trained to get their result, but it does state that the largest improvements are thanks to training the NetVLAD layer, and almost no improvement by training more layers other than the last one.

### Pooling layers

#### fmax

We implemented 2 different types of pooling, the NetVLAD layer as described in the paper and fmax pooling.
For the fmax pooling, we used the AdaptiveMaxPool2d from pytorch in order to specify the output dimension. Initialy, the output dimension were hardcoded to (1,1), because the resulting accuracy seemed to match the one in the paper. In the experiments, we also look at what happens with different dimensions.

#### NetVLAD

As described in the previous section, the NetVLAD layer uses a convolution followed by a softmax to get the soft-assignments of the feature vectors to each of the clusters. We implemented this in pytorch as follows:

```python
a_bar = F.softmax(self.conv(x), dim=1)
```

Here, the `self.conv` attribute is initialized as `torch.nn.Conv2d` module, with the weights and biases initialized using an ![alpha](/alpha.gif) parameter. According to the paper, this value should be computed so that the ratio of the largest and the second largest soft assignment weight is on average equal to 100. Since we were unsure on how to implement this, we instead looked at the effect of different values for alpha and picked the best one as the final value to use. The figure below shows the off-the-shelf (without any additional training, because we are looking for the best possible initialization) recall@N accuracies for several ![alpha](/alpha.gif) values:

![Graph showing off-the-shelf recall@N accuracies for several alpha](/fig-tune-alpha.png)

Based on this graph, and keeping in mind that we do not want the weights to grow unnecessarily large, we decided on the value ![alpha](/alpha=0.1.gif). Using this value we initialize the weights and biases of the convolutional layer as the paper proposes:

```python
self.conv.weight = nn.Parameter(2.0 * self.alpha *  self.c)
self.conv.bias = nn.Parameter(-self.alpha * self.c.norm(dim=1) ** 2)
```
To implement the VLAD vector calculation in pytorch, we restructed equation 1 as follows:

![vlad1](/vlad1.gif)

![vlad2](/vlad2.gif)

![vlad3](/vlad3.gif)

The last sum over the soft assignments can then be implemented using `torch.sum` turning the second half of the expression into a vector-scalar product. The first half is calculated using `torch.bmm` the built-in tensor operation for batch-wise matrix multiplication. By reshaping the ![a_bar](/a_bar.gif) and ![x](/x.gif) before putting them into the equation, we allow the calculation to be carried out for all the features at once, instead of having to loop over the indices ![j](/j.gif) and ![k](/k.gif) in ![V](/V.gif). The final (simplified) VLAD core calculation is:

```python
V = torch.bmm(a_bar, x) - c * torch.sum(a_bar)
```

where ![a_bar](/a_bar.gif) is the soft-assignment and c are the cluster centers.

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
| ![alpha](/alpha.gif)           | 0.1                        |
| K               | 64                         |

### Results

Below are the resulting accuracies after training the networks. After our result is a copy of the result from the original paper.

#### Our results

![Figure with our results](/fig5a.png)

#### Original paper results

![Original paper results](/netvlad-og-plots.png)

Looking at the graphs, we notice that our off-the-shelf accuracies are similar, but become comparatively lower after training. Especially alexnet has significant lower accuracies in our model compared to the original paper.

We also tested the accuracy when we remove the pooling layer entirely and only use the base network. Surprisingly, the accuracies are the best in all cases when compared to the same architecture which does use a pooling layer. A downside of not using any pooling, is that the output dimension is higher, and thus slower to compare against the rest of the database images. The higher dimensionality might also explain why the accuracy is this high, since all the features of the images are being used to compare the output vectors.

### Discussion

After having already done all of the experiments we also found out that we did the cropping of the images slightly different than how this is usually done in similar applications (but the paper does not explicitly mention how they did this). We immediately scaled down the `(480 x 640)` images to `(224 x 224)`, while usually it is first scaled down such that the smallest dimension is `224` pixels, and then cropped to `224` pixels over the other dimension. In addition to this, we found out that AlexNet is pretrained using `(256 x 256)` images, while we are then using it with images scaled down to `(224 x 224)`. We ran a couple of small tests, and it did not have a significant impact on the off-the-shelf performance, but it could still be the case that this has a negative impact on the results we got.

## Conclusion

As shown in the section above, our off-the-shelf is quite similar to the results presented in the paper's table 1, but slightly less than the results shown in their figures. We don't know exactly why this is the case, because we don't know why the paper has better performance on the test set than on the validation set. Probable causes of this problem could be due to the size of the dataset, small differences between the test and validation set, or maybe scaling of the images, but since the paper does not clearly state any of these, we cannot know for sure which is the cause.

The results after training until convergence are promising, but also not the same as the results presented in the paper. Again, this could be causes by differences in (size of) the training and validation sets used. Additionally, we believe hyperparameter tuning can have a big impact on the performance after training as well, but we did not have enough time left to try a lof of different training parameters.

Another remarkable result, is that during our the experiments, the model that performed best was actually just the base network without any form of pooling, although it does not perform as good as the achieved results of the paper. But, as was already mentioned in the results, this might not be a fair comparison, because the output size of this model is much larger.

Finally, we were not able to implement the input whitening, because we did not have enough time left. This would likely have changed the optimal achieved results a bit, but it would not have changed the differences in the the results obtained by the models without whitening.

Taking all of the above into account, we believe we reproduced the paper up to some extent. Some of our results are not as good as the results aquired in the original paper, but it seems plausible that with some extra hyperparameter tuning and tweaking of the dataset and preprocessing their results can be reproduced. There are however, a couple of factors that make reproducing this paper very difficult: some steps in the proposed approach are not clearly defined, and can very well be interpreted in different ways. An example of this is the calculation of the cluster centers, which is not described clear enough to implement, so we had to guess here and there what was meant. Similarly, the input sizes and processing are not mentioned, which we believe is one of the factors that impacted our results in a negative way.


![NetVLAD banner image](/netvlad-banner.png)


## Appendix

[https://arxiv.org/abs/1511.07247][NetVLAD: CNN architecture for weakly supervised place recognition]
