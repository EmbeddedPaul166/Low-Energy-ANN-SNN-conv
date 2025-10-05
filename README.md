# Low-Energy-ANN-SNN-conv

This repository contains experiment code from the following paper:

TBA

## Guide

Experiments were conducted using the environment created from environment.yml file through micromamba (https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

The following directories:
- mnist
- fashion_mnist
- cifar10
- cifar100
- malaria
- oxford_iiit_pet

contain scripts used to compute results for table I and figures 9-11 in the article. To best replicate the results for table I it is advised to use the default arguments for those scripts. Figures 9-11 were computed using quantization scripts from those directories with varying numbers of bits set using command line arguments.

When running training or quantization scripts for the first time you may need to set "download" argument to True in tfds.load() function. This is due to Tensorflow's necessity to download benchmarking datasets when they are used for the first time.

Script mnist/test_snn_fc2.py implements a proof of concept not mentioned in the article, in which test accuracy is calculated using actual binary search spiking neural network converted from the quantized ANN.

NOTE: Despite fixing random seeds in all of the scripts precise replication of results may not be possible due to differences in hardware used for computation, although they should be pretty close.

File contents:
- layers.py: contains fake-quantize layers and the Binary Search Spiking Dense layer implementation (used for proof of concept in mnist/test_snn_fc2.py script)
- calibration.py: contains code used for calibration of models which is described in chapter II.D of the article. Main function, calibrate(), contains two arguments which eventually were not used for computation of results and these are: a_outlier_idxs and w_outlier_idxs.
- models.py: contains model construction functions

Directory called "tests" contains unit and functional tests for functions defined in layers.py and calibration.py files. To run them pytest should be installed additionally using pip. Test files should be run individually otherwise some tests may fail due to mutliple settings of random seeds.


