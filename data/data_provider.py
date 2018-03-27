import os
import sys
from functools import partial

import numpy
import tensorflow as tf
from six.moves import cPickle

sys.path.append('models/research')
# sys.path.append('models/research/gan')
sys.path.append('models/research/slim')

slim = tf.contrib.slim


def get_images_dataset(split_name,
                       dataset_dir,
                       batch_size,
                       stack_depth,
                       file_pattern=None):
    _FILE_PATTERN = '%s*'

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    keys_to_features = {
        'image/height': tf.FixedLenFeature([], tf.float32),
        'image/width': tf.FixedLenFeature([], tf.float32),
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
    def parser(serialized):
        # Decode image.
        parsed = tf.parse_single_example(serialized, keys_to_features)
        image = tf.image.decode_jpeg(parsed['image/encoded'])

        # Resize image.
        image_height, image_width = parsed['image/height'], parsed[
            'image/width']
        image_height = tf.cast(image_height, tf.float32)
        image_width = tf.cast(image_width, tf.float32)
        shortest_side = tf.cond(tf.less_equal(image_height, image_width),
                                true_fn=lambda: image_height,
                                false_fn=lambda: image_width)
        shortest_side = tf.cast(shortest_side, tf.int32)

        def get_closest_res():
            """Use this in order to scale down as little as necessary."""
            closest_res = 2 ** 6
            for i in range(stack_depth, 1, -1):
                curr_res = 2 ** (6 + i)
                closest_res = tf.cond(
                    tf.logical_and(
                        tf.greater_equal(shortest_side, curr_res),
                        closest_res != 2 ** 6  # closest_res hasn't changed (if already changed, we found a solution) TODO: fix this dirt
                    ),
                    true_fn=lambda: curr_res,
                    false_fn=lambda: closest_res)
            return closest_res

        closest_res = get_closest_res()
        scale_ratio = closest_res / shortest_side
        new_height = scale_ratio * image_height
        new_height = tf.cast(new_height, tf.int32)
        new_width = scale_ratio * image_width
        new_width = tf.cast(new_width, tf.int32)
        image = tf.image.resize_images(image, size=[new_height, new_width])

        # Crop image to square.
        image = tf.image.resize_image_with_crop_or_pad(image, closest_res,
                                                       closest_res)
        image = tf.to_float(image)
        image.set_shape((closest_res, closest_res, 3,))
        return image

    images_dataset = images_dataset.map(parser)
    # Normalize. TODO(joppe): if needed, normalize like StackGAN pytorch source
    images_dataset = images_dataset.map(
        lambda image: (image - 128.0) / 128.0)
    images_dataset = images_dataset.padded_batch(batch_size, (64, 64, 3,))

    def _predicate(*xs):
        """Return `True` if this element is a full batch."""
        # Extract the dynamic batch size from the first component of the flattened
        # batched element.
        first_component = xs[0]
        tensor_batch_size = tf.shape(
            first_component, out_type=tf.int32)[0]

        return tf.equal(batch_size, tensor_batch_size)

    images_dataset = images_dataset.filter(_predicate)
    images_dataset = images_dataset.repeat()

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
                                        batch_size,
                                        stack_depth)

    # Get text embedding.
    def parser(embedded_captions):
        import random
        # index = tf.random_uniform(
        #     [1],
        #     minval=0,
        #     maxval=embedded_captions.get_shape().as_list()[0] - 1)
        index = random.randint(0,
                               embedded_captions.get_shape().as_list()[0] - 1)
        selected_caption = embedded_captions[index, :]
        selected_caption.set_shape((1024,))
        return selected_caption

    embedded_captions = load_text_embeddings(
        text_dataset_dir)  # (8855, 10, 1024)
    embedded_captions_dataset = tf.data.Dataset.from_tensor_slices(
        embedded_captions)
    embedded_captions_dataset = embedded_captions_dataset.map(parser)
    embedded_captions_dataset = embedded_captions_dataset.repeat()
    embedded_captions_dataset = embedded_captions_dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    image_caption_dataset = tf.data.Dataset.zip(
        (images_dataset, embedded_captions_dataset))  # type: tf.data.Dataset
    image_caption_iterator = image_caption_dataset.make_one_shot_iterator()
    image, embedded_caption = image_caption_iterator.get_next()

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
