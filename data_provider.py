import sys

import tensorflow as tf

# Add models/ modules to path
sys.path.append('models')
sys.path.append('models/research')
sys.path.append('models/research/gan')
sys.path.append('models/research/slim')

# TF-Slim data provider.
from datasets import download_and_convert_cifar10

# TFGAN CIFAR examples from `tensorflow/models`.
from cifar import data_provider


def download_train_data(dataset_dir):
    # Download CIFAR-10 data, only if the files weren't downloaded before.
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    download_and_convert_cifar10.run(dataset_dir)


def get_training_data_iterator(batch_size, dataset_dir):
    download_train_data(dataset_dir)
    # Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
    # for forward and backwards propogation.
    with tf.name_scope('inputs'):
        with tf.device('/cpu:0'):
            images, one_hot_labels, _, num_classes = data_provider.provide_data(
                batch_size, dataset_dir)

    return images
