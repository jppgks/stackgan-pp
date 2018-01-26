import tensorflow as tf

# def _convert_tensor_or_l_or_d(tensor_or_l_or_d):
#   """Convert input, list of inputs, or dictionary of inputs to Tensors."""
#   if isinstance(tensor_or_l_or_d, (list, tuple)):
#     return [tf.convert_to_tensor(x) for x in tensor_or_l_or_d]
#   elif isinstance(tensor_or_l_or_d, dict):
#     return {k: tf.convert_to_tensor(v) for k, v in tensor_or_l_or_d.items()}
#   else:
#     return tf.convert_to_tensor(tensor_or_l_or_d)
from tensorflow.contrib.gan.python import namedtuples


def _compute_mean_covariance(img):
  """#### Color loss helpers"""
  img_shape = img.get_shape().as_list()
  batch_size = img_shape[0]
  height = img_shape[1]
  width = img_shape[2]
  channel_num = img_shape[3]
  num_pixels = height * width

  # batch_size * 1 * 1 * channel_num
  mu = tf.reduce_mean(
    tf.reduce_mean(img, axis=1, keep_dims=True),
    axis=2,
    keep_dims=True)

  # batch_size * channel_num * num_pixels
  img_hat = img - mu  # mu.expand_as(img)
  img_hat = tf.reshape(img_hat, (batch_size, channel_num, num_pixels))
  # batch_size * num_pixels * channel_num
  img_hat_transpose = tf.transpose(img_hat, [0, 2, 1])
  # batch_size * channel_num * channel_num
  covariance = tf.matmul(img_hat, img_hat_transpose)
  covariance = covariance / num_pixels

  return mu, covariance


def _tensor_pool_adjusted_model(model, tensor_pool_fn):
  """Adjusts model using `tensor_pool_fn`.
  Args:
    model: A GANModel tuple.
    tensor_pool_fn: A function that takes (generated_data, generator_inputs),
      stores them in an internal pool and returns a previously stored
      (generated_data, generator_inputs) with some probability. For example
      tfgan.features.tensor_pool.
  Returns:
    A new GANModel tuple where discriminator outputs are adjusted by taking
    pooled generator outputs as inputs. Returns the original model if
    `tensor_pool_fn` is None.
  Raises:
    ValueError: If tensor pool does not support the `model`.
  """
  if tensor_pool_fn is None:
    return model

  pooled_generated_data, pooled_generator_inputs = tensor_pool_fn(
    (model.generated_data, model.generator_inputs))

  if isinstance(model, namedtuples.GANModel):
    with tf.variable_scope(model.discriminator_scope, reuse=True):
      dis_gen_outputs = model.discriminator_fn(pooled_generated_data,
                                               pooled_generator_inputs)
    return model._replace(discriminator_gen_outputs=dis_gen_outputs)
  elif isinstance(model, namedtuples.ACGANModel):
    with tf.variable_scope(model.discriminator_scope, reuse=True):
      (dis_pooled_gen_outputs,
       dis_pooled_gen_classification_logits) = model.discriminator_fn(
        pooled_generated_data, pooled_generator_inputs)
    return model._replace(
      discriminator_gen_outputs=dis_pooled_gen_outputs,
      discriminator_gen_classification_logits=
      dis_pooled_gen_classification_logits)
  elif isinstance(model, namedtuples.InfoGANModel):
    with tf.variable_scope(model.discriminator_scope, reuse=True):
      (dis_pooled_gen_outputs,
       pooled_predicted_distributions) = model.discriminator_and_aux_fn(
        pooled_generated_data, pooled_generator_inputs)
    return model._replace(
      discriminator_gen_outputs=dis_pooled_gen_outputs,
      predicted_distributions=pooled_predicted_distributions)
  else:
    raise ValueError('Tensor pool does not support `model`: %s.' % type(model))
