import os
import sys

import numpy
import tensorflow as tf
from six.moves import cPickle

sys.path.append('models/research')
# sys.path.append('models/research/gan')
sys.path.append('models/research/slim')

# TF-Slim data provider.
from models.research.slim.datasets import dataset_utils

slim = tf.contrib.slim


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading cifar10.
  
    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.
  
    Returns:
      A `Dataset` namedtuple.
  
    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    _FILE_PATTERN = '%s*'

    SPLITS_TO_SIZES = {'train': 5394, 'test': 5794, 'val': 600}

    _ITEMS_TO_DESCRIPTIONS = {
        'image': 'A [64 x 64 x 3] color image.',
        'label': 'A single integer between 0 and 9',
    }

    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if not reader:
        reader = tf.TFRecordReader

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

    items_to_handlers = {
        # TODO(joppe): replace with actual image size
        'image': slim.tfexample_decoder.Image(shape=[64, 64, 3]),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        labels_to_names=labels_to_names)


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
    dataset = get_split(split_name, image_dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=5 * batch_size,
        common_queue_min=batch_size,
        shuffle=(split_name == 'train'))
    [image] = provider.get(['image'])

    # Preprocess the images.
    crop_size = 64 * (2 ** (stack_depth - 1))
    # - resize,
    image = tf.image.resize_images(image, int(crop_size * 76 / 64))
    # - crop,
    image = tf.random_crop(image, crop_size)
    # - flip horizontally, and
    image = tf.image.random_flip_up_down(image)
    # TODO(joppe): if needed, normalize like StackGAN pytorch source
    # - normalize.
    image = (tf.to_float(image) - 128.0) / 128.0

    # Get text embedding.
    def _select_one_caption(embedded_captions):
        # index = tf.random_uniform([1],
        #                           minval=0,
        #                           maxval=(embedded_captions.shape[0] - 1),
        #                           dtype=tf.int8)
        import random
        index = random.randint(0, embedded_captions.shape[0] - 1)
        return embedded_captions[index, :]

    embedded_captions = load_text_embeddings(text_dataset_dir)
    embedded_captions_dataset = tf.data.Dataset.from_tensor_slices(
        embedded_captions)
    embedded_captions_dataset = embedded_captions_dataset.map(
        lambda emb: tf.py_func(_select_one_caption, emb, emb.dtype))
    epochs = 100
    embedded_captions_dataset = embedded_captions_dataset.repeat(epochs)
    embedded_captions_iterator = embedded_captions_dataset.make_one_shot_iterator()
    embedded_caption = embedded_captions_iterator.get_next()

    # Creates a QueueRunner for the pre-fetching operation.
    images, embedded_caption = tf.train.batch(
        [image, embedded_caption],
        batch_size=batch_size,
        num_threads=32,
        capacity=5 * batch_size)

    return images, embedded_caption


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
