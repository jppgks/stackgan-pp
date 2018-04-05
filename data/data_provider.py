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
    # Create tf.data.Dataset
    _FILE_PATTERN = '%s*'

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    files = tf.data.Dataset.list_files(file_pattern)
    images_dataset = tf.data.TFRecordDataset(files, num_parallel_reads=24)

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

        def get_largest_res():
            """Use this in order to scale down as little as necessary."""
            largest_res = 2 ** (6 + stack_depth - 1)
            return largest_res

        largest_res = get_largest_res()
        scale_ratio = largest_res / shortest_side
        scale_ratio = tf.cast(scale_ratio, tf.float32)
        new_height = scale_ratio * image_height
        new_height = tf.cast(new_height, tf.int32)
        new_width = scale_ratio * image_width
        new_width = tf.cast(new_width, tf.int32)
        image = tf.image.resize_images(image, size=[new_height, new_width])

        # Crop image to square.
        image = tf.image.resize_image_with_crop_or_pad(image, largest_res,
                                                       largest_res)
        image = tf.to_float(image)
        image.set_shape((largest_res, largest_res, 3,))

        #  TODO(joppe): resize to all image resolutions needed in training

        return image

    images_dataset = images_dataset.map(parser, num_parallel_calls=24)
    # Normalize. TODO(joppe): if needed, normalize like StackGAN pytorch source
    largest_res = 2 ** (6 + stack_depth - 1)
    images_dataset = images_dataset.padded_batch(batch_size,
                                                 (largest_res, largest_res, 3,))
    images_dataset = images_dataset.map(
        lambda image_batch: (image_batch - 128.0) / 128.0)

    def _predicate(*xs):
        """Return `True` if this element is a full batch."""
        # Extract the dynamic batch size from the first component of the flattened
        # batched element.
        first_component = xs[0]
        tensor_batch_size = tf.shape(
            first_component, out_type=tf.int32)[0]

        return tf.equal(batch_size, tensor_batch_size)

    images_dataset = images_dataset.filter(_predicate)

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


def get_text_captions_dataset(text_data_dir):
    def load_filenames(data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with tf.gfile.Open(filepath, 'rb') as f:
            if sys.version_info < (3,):
                filenames = cPickle.load(f)
            else:
                filenames = cPickle.load(f, encoding='bytes')
        return filenames

    def get_caption_paths():
        # def load_captions(caption_path):
        #     with open(caption_path, "r") as f:
        #         captions = f.read().decode('utf8').split('\n')
        #     captions = [cap.replace("\ufffd\ufffd", " ")
        #                 for cap in captions if len(cap) > 0]
        #     return captions

        filenames = load_filenames(text_data_dir)
        caption_data_dir = os.path.join(text_data_dir, os.pardir, 'text_c10')
        paths_to_txt_files = []
        for key in filenames:
            caption_path = os.path.join(caption_data_dir, key + '.txt')
            paths_to_txt_files.append(caption_path)

        return paths_to_txt_files

    # Create tf.data.Dataset
    file_paths = get_caption_paths()
    captions_dataset = tf.data.TextLineDataset(file_paths)

    # Collect captions for same img (10 captions / img).
    captions_dataset = captions_dataset.batch(10)

    return captions_dataset


def get_captions_text_with_embedding_dataset(batch_size, text_dataset_dir):
    embedded_captions = load_text_embeddings(
        text_dataset_dir)  # (8855, 10, 1024)
    embedded_captions_dataset = tf.data.Dataset.from_tensor_slices(
        embedded_captions)

    text_captions_dataset = get_text_captions_dataset(text_dataset_dir)

    # Get indices for selecting one caption from 10 given captions per img.
    num_imgs = 8855
    num_captions_per_img = 10
    import random
    indices_chosen_caption = []
    for i in range(num_imgs):
        index = random.randint(0, num_captions_per_img - 1)
        indices_chosen_caption.append(index)

    # Take one caption text.
    indices_chosen_caption = iter(indices_chosen_caption)
    def take_single_text(text_captions):
        return text_captions[next(indices_chosen_caption)]
    text_captions_dataset = text_captions_dataset.map(take_single_text,
                                                      num_parallel_calls=24)
    text_captions_dataset = text_captions_dataset.cache()
    text_captions_dataset = text_captions_dataset.repeat()
    # Batch.
    text_captions_dataset = text_captions_dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    # Take one caption embedding.
    indices_chosen_caption = iter(indices_chosen_caption)
    def take_single_emb(embedded_captions):
        # Select single caption.
        selected_caption_embedding = embedded_captions[next(
            indices_chosen_caption), :]
        selected_caption_embedding.set_shape((1024,))
        return selected_caption_embedding
    embedded_captions_dataset = embedded_captions_dataset.map(take_single_emb,
                                                      num_parallel_calls=24)
    # Batch.
    embedded_captions_dataset = embedded_captions_dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    return text_captions_dataset, embedded_captions_dataset


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

    # TODO(joppe): move as much operations as possible from individual datasets to the shared dataset
    images_dataset = get_images_dataset(split_name,
                                        image_dataset_dir,
                                        batch_size,
                                        stack_depth)

    captions_text_dataset, captions_embedding_dataset = get_captions_text_with_embedding_dataset(
        batch_size,
        text_dataset_dir)

    image_caption_dataset = tf.data.Dataset.zip(
        (images_dataset, captions_embedding_dataset))  # type: tf.data.Dataset
    image_caption_dataset = image_caption_dataset.cache()
    image_caption_dataset = image_caption_dataset.repeat()
    image_caption_dataset = image_caption_dataset.apply(
        tf.contrib.data.prefetch_to_device("/gpu:0"))

    image_embedding_iterator = image_caption_dataset.make_one_shot_iterator()
    images, captions_embedding = image_embedding_iterator.get_next()

    text_iterator = captions_text_dataset.make_one_shot_iterator()
    captions_text = text_iterator.get_next()

    return images, captions_text, captions_embedding


def get_training_data_iterator(batch_size,
                               image_dataset_dir,
                               text_dataset_dir,
                               stack_depth=1):
    # Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
    # for forward and backwards propogation.
    with tf.name_scope('inputs'):
        with tf.device('/cpu:0'):
            images, captions_text, captions_embedding = provide_data(
                batch_size,
                image_dataset_dir,
                text_dataset_dir,
                stack_depth=stack_depth)

    return images, captions_text, captions_embedding
