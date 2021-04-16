# NetVLAD Reproduction Study

_Elwin Duinkerken & Timo Verlaan_


![NetVLAD banner image](/netvlad-banner.png)

This repository contains our implementation of our reproduction study for the `CS4240 Deep Learning` course. We replicated parts of figure 5, 10 and table 1.

Original paper: [NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/abs/1511.07247)


## Results

![Figure 5a reproduction](/plots/fig5a.png)

The blog post about this reproduction study, containing all of our results can be found here: [https://darthtimothee.github.io/netvlad-reproduced/](https://darthtimothee.github.io/netvlad-reproduced/)


## Running the application 

The dependencies of this implementation are:
- numpy
- scipy 
- pytorch
- torchvision
- tqdm
- colorama
- sklearn
- faiss (or if possible faiss-gpu)

The application can then be run using:
```bash
python3 ./main.py DATAFOLDER
```

where `DATAFOLDER` should point to the directory containing all the training and test images and queries, and a sub-directory `datasets` that contains the database files.

The `test_off_the_shelf.py` script can be used to quickly evaluate off-the-shelf model setups without training and does not take any input parameters. The path to the dataset has to be manually configured in the file.