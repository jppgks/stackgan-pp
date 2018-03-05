import os
import sys
from functools import partial

import numpy
import tensorflow as tf
from six.moves import cPickle

sys.path.append('models/research')
# sys.path.append('models/research/gan')
sys.path.append('models/research/slim')

# TF-Slim data provider.
from models.research.slim.datasets import dataset_utils

slim = tf.contrib.slim


def get_images_dataset(split_name,
                       dataset_dir,
                       batch_size,
                       file_pattern=None):
    _FILE_PATTERN = '%s*'

    SPLITS_TO_SIZES = {'train': 5394, 'test': 5794, 'val': 600}

    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    keys_to_features = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/colorspace': tf.FixedLenFeature([], tf.string),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/id': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/extra': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64,
                                                default_value=tf.zeros([],
                                                                       dtype=tf.int64)),
        'image/class/text': tf.FixedLenFeature([], tf.string),
        'image/class/conf': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/text': tf.VarLenFeature(dtype=tf.string),
        'image/object/bbox/conf': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/score': tf.VarLenFeature(dtype=tf.float32),
        'image/object/parts/x': tf.VarLenFeature(dtype=tf.float32),
        'image/object/parts/y': tf.VarLenFeature(dtype=tf.float32),
        'image/object/parts/v': tf.VarLenFeature(dtype=tf.int64),
        'image/object/parts/score': tf.VarLenFeature(dtype=tf.float32),
        'image/object/count': tf.FixedLenFeature([], tf.int64),
        'image/object/area': tf.VarLenFeature(dtype=tf.float32),
        'image/object/id': tf.VarLenFeature(dtype=tf.string)
    }

    # Create tf.data.Dataset
    filenames = tf.gfile.Glob(file_pattern)
    images_dataset = tf.data.TFRecordDataset(filenames)

    # Parse TFRecord.
    def parser(record):
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.image.decode_jpeg(parsed['image/encoded'])
        image = tf.to_float(image)
        return image

    images_dataset = images_dataset.map(parser)
    # Normalize. TODO(joppe): if needed, normalize like StackGAN pytorch source
    images_dataset = images_dataset.map(
        lambda image: (image - 128.0) / 128.0)
    images_dataset = images_dataset.repeat()
    images_dataset = images_dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    return images_dataset


def load_text_embeddings(text_data_dir):
    # Get text embeddings.
    embedding_filename = 'char-CNN-RNN-embeddings.pickle'
    with tf.gfile.Open(text_data_dir + embedding_filename, 'rb') as f:
        if sys.version_info < (3,):
            embeddings = cPickle.load(f)
        else:
            embeddings = cPickle.load(f, encoding='bytes')

    # >> tf.shape(embeddings)
    # [8855, 10, 1024]
    return numpy.array(embeddings)


def provide_data(batch_size,
                 image_dataset_dir='tf_birds_dataset/cub/with_600_val_split/',
                 text_dataset_dir='birds/train/',
                 split_name='train',
                 stack_depth=1):
    """Provides batches of CIFAR data.

    Args:
      batch_size: The number of images in each batch.
      image_dataset_dir: The directory where the CIFAR10 data can be found. If `None`,
        use default.
      dataset_name: Name of the dataset.
      split_name: Should be either 'train' or 'test'.
      one_hot: Output one hot vector instead of int32 label.

    Returns:
      images: A `Tensor` of size [batch_size, 32, 32, 3]. Output pixel values are
        in [-1, 1].
      labels: Either (1) one_hot_labels if `one_hot` is `True`
              A `Tensor` of size [batch_size, num_classes], where each row has a
              single element set to one and the rest set to zeros.
              Or (2) labels if `one_hot` is `False`
              A `Tensor` of size [batch_size], holding the labels as integers.
      num_samples: The number of total samples in the dataset.
      num_classes: The number of classes in the dataset.

    Raises:
      ValueError: if the split_name is not either 'train' or 'test'.
    """
    images_dataset = get_images_dataset(split_name,
                                        image_dataset_dir,
                                        batch_size)

    # Get text embedding.
    def _select_one_caption(embedded_captions):
        import random
        index = random.randint(0, embedded_captions.shape[0] - 1)
        return embedded_captions[index, :]

    embedded_captions = load_text_embeddings(text_dataset_dir)
    embedded_captions_dataset = tf.data.Dataset.from_tensor_slices(
        embedded_captions)
    embedded_captions_dataset = embedded_captions_dataset.map(
        lambda emb: tf.py_func(_select_one_caption, [emb], [emb.dtype]))
    embedded_captions_dataset = embedded_captions_dataset.map(
        lambda tuple: tuple[0])
    embedded_captions_dataset = embedded_captions_dataset.repeat()
    embedded_captions_dataset = embedded_captions_dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    image_caption_dataset = tf.data.Dataset.zip(
        (images_dataset, embedded_captions_dataset))  # type: tf.data.Dataset
    image_caption_iterator = image_caption_dataset.make_one_shot_iterator()
    image, embedded_caption = image_caption_iterator.get_next()

    print(image.shape)
    print(embedded_caption.shape)

    return image, embedded_caption


def get_training_data_iterator(batch_size,
                               image_dataset_dir,
                               text_dataset_dir,
                               stack_depth=1):
    # Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
    # for forward and backwards propogation.
    with tf.name_scope('inputs'):
        with tf.device('/cpu:0'):
            images, caption_embedding = provide_data(
                batch_size,
                image_dataset_dir,
                text_dataset_dir,
                stack_depth=stack_depth)

    return images, caption_embedding
