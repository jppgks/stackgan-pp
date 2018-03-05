import os
import sys

import tensorflow as tf

# Add models/ modules to path
from data.decode_example import decode_serialized_example

sys.path.append('models/research')
sys.path.append('models/research/gan')
sys.path.append('models/research/slim')

# TF-Slim data provider.
from models.research.slim.datasets import download_and_convert_cifar10

# TFGAN CIFAR examples from `tensorflow/models`.
from models.research.gan.cifar import data_provider


def download_train_data(dataset_dir):
    # Download CIFAR-10 data, only if the files weren't downloaded before.
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    download_and_convert_cifar10.run(dataset_dir)


def provide_data(batch_size,
                 dataset_dir='tf_birds_dataset/cub/with_600_val_split/',
                 split_name='train', one_hot=False, stack_depth=1):
    """Provides batches of CIFAR data.
  
    Args:
      batch_size: The number of images in each batch.
      dataset_dir: The directory where the CIFAR10 data can be found. If `None`,
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
    dataset = get_split(split_name, dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=5 * batch_size,
        common_queue_min=batch_size,
        shuffle=(split_name == 'train'))
    [image, label] = provider.get(['image', 'label'])

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

    # Creates a QueueRunner for the pre-fetching operation.
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=32,
        capacity=5 * batch_size)

    # labels = tf.reshape(labels, [-1])
    #
    # if one_hot:
    #     labels = tf.one_hot(labels, dataset.num_classes)

    return images, text_embedding


def get_training_data_iterator(batch_size, dataset_dir, stack_depth=1):
    download_train_data(dataset_dir)
    # Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
    # for forward and backwards propogation.
    with tf.name_scope('inputs'):
        with tf.device('/cpu:0'):
            images, text_embedding = provide_data(
                batch_size, dataset_dir, stack_depth=stack_depth)

    return images, text_embedding


from models.research.slim.datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s*'

SPLITS_TO_SIZES = {'train': 5394, 'test': 5794, 'val': 600}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [64 x 64 x 3] color image.',
    'label': 'A single integer between 0 and 9',
}


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

# def get_region_data(serialized_example, cfg, fetch_ids=True, fetch_labels=True,
#                     fetch_text_labels=True):
#     """
#     Return the image, an array of bounding boxes, and an array of ids.
#     """
#
#     feature_dict = {}
#
#     if cfg.REGION_TYPE == 'image':
#
#         features_to_extract = [('image/encoded', 'image')]
#
#         if fetch_ids:
#             features_to_extract.append(('image/id', 'id'))
#         if fetch_labels:
#             features_to_extract.append(('image/class/label', 'label'))
#         if fetch_text_labels:
#             features_to_extract.append(('image/class/text', 'text'))
#
#         features = decode_serialized_example(serialized_example,
#                                              features_to_extract)
#
#         image = features['image']
#         feature_dict['image'] = image
#
#         bboxes = tf.constant([[0.0, 0.0, 1.0, 1.0]])
#         feature_dict['bboxes'] = bboxes
#
#         if fetch_ids:
#             ids = [features['id']]
#             feature_dict['ids'] = ids
#         if fetch_labels:
#             labels = [features['label']]
#             feature_dict['labels'] = labels
#         if fetch_text_labels:
#             text = [features['text']]
#             feature_dict['text'] = text
#
#     else:
#         raise ValueError("Unknown REGION_TYPE: %s" % (cfg.REGION_TYPE,))
#
#     return feature_dict
#
#
# def create_training_batch(serialized_example, cfg, add_summaries):
#     features = get_region_data(serialized_example, cfg, fetch_ids=False,
#                                fetch_labels=True, fetch_text_labels=False)
#
#     original_image = features['image']
#     bboxes = features['bboxes']
#     labels = features['labels']
#
#     distorted_inputs = get_distorted_inputs(original_image, bboxes, cfg,
#                                             add_summaries)
#
#     distorted_inputs = tf.subtract(distorted_inputs, 0.5)
#     distorted_inputs = tf.multiply(distorted_inputs, 2.0)
#
#     names = ('inputs', 'labels')
#     tensors = [distorted_inputs, labels]
#     return [names, tensors]
#
#
# def input_nodes(tfrecords, cfg, num_epochs=None, batch_size=32, num_threads=2,
#                 shuffle_batch=True, random_seed=1, capacity=1000,
#                 min_after_dequeue=96,
#                 add_summaries=True, input_type='train',
#                 fetch_text_labels=False):
#     """
#     Args:
#         tfrecords:
#         cfg:
#         num_epochs: number of times to read the tfrecords
#         batch_size:
#         num_threads:
#         shuffle_batch:
#         capacity:
#         min_after_dequeue:
#         add_summaries: Add tensorboard summaries of the images
#         input_type: 'train', 'visualize', 'test', 'classification'
#     """
#     with tf.name_scope('inputs'):
#
#         # A producer to generate tfrecord file paths
#         filename_queue = tf.train.string_input_producer(
#             tfrecords,
#             num_epochs=num_epochs
#         )
#
#         # Construct a Reader to read examples from the tfrecords file
#         reader = tf.TFRecordReader()
#         _, serialized_example = reader.read(filename_queue)
#
#         if input_type == 'train' or input_type == 'test':
#             batch_keys, data_to_batch = create_training_batch(
#                 serialized_example, cfg, add_summaries)
#         else:
#             raise ValueError(
#                 "Unknown input type: %s. Options are `train`, `test`, " \
#                 "`visualize`, and `classification`." % (input_type,))
#
#         if shuffle_batch:
#             batch = tf.train.shuffle_batch(
#                 data_to_batch,
#                 batch_size=batch_size,
#                 num_threads=num_threads,
#                 capacity=capacity,
#                 min_after_dequeue=min_after_dequeue,
#                 seed=random_seed,
#                 enqueue_many=True
#             )
#
#         else:
#             batch = tf.train.batch(
#                 data_to_batch,
#                 batch_size=batch_size,
#                 num_threads=num_threads,
#                 capacity=capacity,
#                 enqueue_many=True
#             )
#
#         batch_dict = {k: v for k, v in zip(batch_keys, batch)}
#
#         return batch_dict
