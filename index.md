
## Introduction

In this article we will discuss the results and progress we made during the reproduction project for the CS4240 Deep Learning course. We reproduced the paper NetVLAD: CNN architecture for weakly supervised place recognition, in which an end-to-end trainable layer for image retrieval is proposed.

We will start by describing the problem the paper attempts to solve and their proposed solution. After that we will describe our implementation of this solution and support some of the design choices we made. Next, we discuss the experiments we ran, the results we got and compare them to the original results of the paper. Finally we draw some conclusions on the reproducibility of this paper and reflect on our development process.

## Problem description

The problem the NetVlad paper tries to solve, is to determine the location of a given image. A query image with unknown location is compared to all images from a dataset where the location is known. By matching the query image to the correct image from the dataset, the location of the query can be estimated.
The difficulty of this problem is that 2 photos taken in the same location might look very different based on factor such as the time it was taken, how many background objects there are, or slight changes in viewpoints.