"""Convenience functions for training and evaluating a TFGAN CIFAR example."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.gan import eval as tfgan_eval
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import control_flow_ops

__all__ = [
    'get_inception_scores',
    'get_frechet_inception_distance',
]


def get_inception_scores(images, batch_size, num_inception_images=10):
    """Get Inception score for some images.
    Args:
      images: Image minibatch. Shape [batch size, width, height, channels]. Values
        are in [-1, 1].
      batch_size: Python integer. Batch dimension.
      num_inception_images: Number of images to run through Inception at once.
    Returns:
      Inception scores. Tensor shape is [batch size].
    Raises:
      ValueError: If `batch_size` is incompatible with the first dimension of
        `images`.
      ValueError: If `batch_size` isn't divisible by `num_inception_images`.
    """
    # Validate inputs.
    images.shape[0:1].assert_is_compatible_with([batch_size])
    if batch_size % num_inception_images != 0:
        raise ValueError(
            '`batch_size` must be divisible by `num_inception_images`.')

    # Resize images.
    size = 299
    resized_images = image_ops.resize_bilinear(images, [size, size])

    # Run images through Inception.
    num_batches = batch_size // num_inception_images
    inc_score = tfgan_eval.inception_score(
        resized_images, num_batches=num_batches)

    return inc_score, control_flow_ops.no_op()


def get_frechet_inception_distance(real_images, generated_images, batch_size,
                                   num_inception_images):
    """Get Frechet Inception Distance between real and generated images.
    Args:
      real_images: Real images minibatch. Shape [batch size, width, height,
        channels. Values are in [-1, 1].
      generated_images: Generated images minibatch. Shape [batch size, width,
        height, channels]. Values are in [-1, 1].
      batch_size: Python integer. Batch dimension.
      num_inception_images: Number of images to run through Inception at once.
    Returns:
      Frechet Inception distance. A floating-point scalar.
    Raises:
      ValueError: If the minibatch size is known at graph construction time, and
        doesn't batch `batch_size`.
    """
    # Validate input dimensions.
    real_images.shape[0:1].assert_is_compatible_with([batch_size])
    generated_images.shape[0:1].assert_is_compatible_with([batch_size])

    # Resize input images.
    size = 299
    resized_real_images = image_ops.resize_bilinear(real_images, [size, size])
    resized_generated_images = image_ops.resize_bilinear(
        generated_images, [size, size])

    # Compute Frechet Inception Distance.
    num_batches = batch_size // num_inception_images
    fid = tfgan_eval.frechet_inception_distance(
        resized_real_images, resized_generated_images, num_batches=num_batches)

    return fid, control_flow_ops.no_op()
