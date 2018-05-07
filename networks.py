import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

slim = tf.contrib.slim


def log(x):
    x = tf.cast(x, dtype=tf.float32)
    numerator = tf.log(x)
    denominator = tf.log(2.)
    with tf.Session():
        result = numerator / denominator
        result = result.eval()
    return result


# https://github.com/nnUyi/SNGAN/blob/master/ops.py
# spectral_norm
def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x / (tf.reduce_sum(input_x ** 2) ** 0.5 + epsilon)
    return input_x_norm


def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None,
                          reuse=False, name='weights_SN'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            u = tf.get_variable('u', shape=[1, w_shape[-1]],
                                initializer=tf.truncated_normal_initializer(),
                                trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite + 1

        u_hat, v_hat, _ = power_iteration(u, iteration)

        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))

        w_mat = w_mat / sigma

        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not (update_collection == 'NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))

            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm


@add_arg_scope
def conv2d(input_x, kernel_size, stride, scope_name='conv2d',
           padding='SAME', spectral_norm=True, update_collection=None,
           normalizer_fn=None, normalizer_params=None,
           activation_fn=tf.nn.leaky_relu):
    output_len = kernel_size[3]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', kernel_size, tf.float32,
                                  initializer=tf.random_normal_initializer(
                                      stddev=0.02))
        bias = tf.get_variable('bias', output_len, tf.float32,
                               initializer=tf.constant_initializer(0))
        if spectral_norm:
            weights = weights_spectral_norm(weights,
                                            update_collection=update_collection)

        conv = tf.nn.bias_add(
            tf.nn.conv2d(input_x, weights, strides=stride, padding=padding),
            bias)
        if normalizer_fn:
            conv = normalizer_fn(conv)
        if activation_fn is not None:
            conv = activation_fn(conv)
        return conv


# TODO(joelshor): Use fused batch norm by default. Investigate why some GAN
# setups need the gradient of gradient FusedBatchNormGrad.
def dcgan_generator(inputs,
                    depth=64,
                    final_size=64,
                    num_outputs=3,
                    is_training=True,
                    reuse=None,
                    scope='DCGANGenerator',
                    fused_batch_norm=False):
    """Generator network for DCGAN.
    Construct generator network from inputs to the final endpoint.
    Args:
      inputs: A tensor with any size N. [batch_size, N]
      depth: Number of channels in last deconvolution layer.
      final_size: The shape of the final output.
      num_outputs: Number of output features. For images, this is the number of
        channels.
      is_training: whether is training or not.
      reuse: Whether or not the network has its variables should be reused. scope
        must be given to be reused.
      scope: Optional variable_scope.
      fused_batch_norm: If `True`, use a faster, fused implementation of
        batch norm.
    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, 32, 32, channels]
      end_points: a dictionary from components of the network to their activation.
    Raises:
      ValueError: If `inputs` is not 2-dimensional.
      ValueError: If `final_size` isn't a power of 2 or is less than 8.
    """
    normalizer_fn = tf.contrib.layers.batch_norm
    normalizer_fn_args = {
        'is_training': is_training,
        'zero_debias_moving_mean': True,
        'fused': fused_batch_norm,
    }
    #  inputs.get_shape().assert_has_rank(2)
    if log(final_size) != int(log(final_size)):
        raise ValueError('`final_size` (%i) must be a power of 2.' % final_size)
    if final_size < 8:
        raise ValueError(
            '`final_size` (%i) must be greater than 8.' % final_size)

    end_points = {}
    with tf.variable_scope(scope, values=[inputs], reuse=reuse) as scope:
        with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
            with slim.arg_scope([slim.conv2d_transpose],
                                normalizer_fn=normalizer_fn,
                                stride=2,
                                kernel_size=4):
                if len(inputs.get_shape()) == 2:
                    # Init stage.
                    num_layers = int(log(final_size)) - 1
                    net = tf.expand_dims(tf.expand_dims(inputs, 1), 1)
                else:
                    # Next stage.
                    num_layers = int(log(final_size)) - (
                        int(log(inputs.shape[1])) - 1) - 1
                    net = inputs

                if num_layers > 1:
                    # First upscaling is different because it takes the input vector.
                    current_depth = depth * 2 ** (num_layers - 1)
                    scope = 'deconv1'
                    # ! Default activation for slim.conv2d_transpose is relu.
                    net = slim.conv2d_transpose(
                        net, current_depth, stride=1, padding='VALID',
                        scope=scope)
                    end_points[scope] = net

                for i in range(2, num_layers):
                    scope = 'deconv%i' % (i)
                    current_depth = depth * 2 ** (num_layers - i)
                    net = slim.conv2d_transpose(net, current_depth, scope=scope)
                    end_points[scope] = net

                # Last layer has different normalizer and activation.
                scope = 'deconv%i' % (num_layers)
                net = slim.conv2d_transpose(
                    net, depth, normalizer_fn=None, activation_fn=None,
                    scope=scope)
                end_points[scope] = net

                # Convert to proper channels.
                scope = 'logits'
                logits = slim.conv2d(
                    net,
                    num_outputs,
                    normalizer_fn=None,
                    activation_fn=None,
                    kernel_size=1,
                    stride=1,
                    padding='VALID',
                    scope=scope)
                end_points[scope] = logits

                logits.get_shape().assert_has_rank(4)
                logits.get_shape().assert_is_compatible_with(
                    [None, final_size, final_size, num_outputs])

                return logits, end_points


def augment(conditioning, new_dim=128):
    """Increases the number of conditioning variables available for training.

    :param conditioning: Text embedding
    :param new_dim: 
        (default of 128 as in StackGAN paper)
    :return: `conditioning`, augmented with additional latent variables, 
        sampled from a Gaussian distribution, with mean and standard deviation 
        as a function of `conditioning`.
    """
    # Encode
    net = slim.fully_connected(conditioning, new_dim * 4, activation_fn=None)
    split = net.get_shape().as_list()[1]
    split = int(split / 2)
    glu = net[:, :split] * tf.sigmoid(net[:, split:])
    mu = glu[:, :new_dim]
    logvar = glu[:, new_dim:]

    # Reparametrize
    std = tf.exp(logvar * 0.5)
    gauss_sample = tf.random_normal(tf.shape(std))
    augmented_conditioning = mu + (gauss_sample * std)

    return augmented_conditioning, mu, logvar


def generator(inputs, final_size=64, apply_batch_norm=False):
    # TODO: docstring
    """Generator to produce CIFAR images.
    Args: 
        :param inputs: 3D tuple of (is_init_stage, noise, conditioning) with
            a) noise as the sample from the noise distribution if is_init_stage,
            otherwise noise is the hidden code of the previous stage
            b) conditioning always as the text embedding.
    Returns:
      A single Tensor with a batch of generated CIFAR images.
    """
    is_init_stage, noise, conditioning = inputs

    if is_init_stage:
        noise = tf.concat([conditioning, noise], 1)  # noise, conditioning -1
        num_layers = int(log(final_size)) - 1
    else:
        h_code_final_size = noise.get_shape()[2]
        conditioning = tf.reshape(conditioning,
                                  [-1, tf.size(conditioning), 1, 1])
        conditioning = tf.tile(conditioning,
                               [1, 1, h_code_final_size, h_code_final_size])
        conditioning = tf.reshape(conditioning,
                                  [noise.get_shape()[0],  # batch size
                                   h_code_final_size,
                                   h_code_final_size,
                                   -1])

        noise = tf.concat([conditioning, noise], -1)
        num_layers = int(log(final_size)) - (int(log(noise.shape[1])) - 1) - 1

    images, end_points = dcgan_generator(
        noise, final_size=final_size,
        is_training=apply_batch_norm)  # scope=stage_scope, reuse=reuse

    hidden_code = end_points['deconv%i' % (num_layers)]

    # Make sure output lies between [-1, 1].
    return tf.tanh(images), hidden_code


def _validate_image_inputs(inputs):
    inputs.get_shape().assert_has_rank(4)
    inputs.get_shape()[1:3].assert_is_fully_defined()
    if inputs.get_shape()[1] != inputs.get_shape()[2]:
        raise ValueError('Input tensor does not have equal width and height: ',
                         inputs.get_shape()[1:3])
    width = inputs.get_shape().as_list()[1]
    if log(width) != int(log(width)):
        raise ValueError('Input tensor `width` is not a power of 2: ', width)


# TODO(joelshor): Use fused batch norm by default. Investigate why some GAN
# setups need the gradient of gradient FusedBatchNormGrad.
def dcgan_discriminator(inputs,
                        depth=64,
                        is_training=True,
                        reuse=None,
                        scope='DCGANDiscriminator',
                        fused_batch_norm=False):
    """Discriminator network for DCGAN.
    Construct discriminator network from inputs to the final endpoint.
    Args:
      inputs: A tensor of size [batch_size, height, width, channels]. Must be
        floating point.
      depth: Number of channels in first convolution layer.
      is_training: Whether the network is for training or not.
      reuse: Whether or not the network variables should be reused. `scope`
        must be given to be reused.
      scope: Optional variable_scope.
      fused_batch_norm: If `True`, use a faster, fused implementation of
        batch norm.
    Returns:
      logits: The pre-softmax activations, a tensor of size [batch_size, 1]
      end_points: a dictionary from components of the network to their activation.
    Raises:
      ValueError: If the input image shape is not 4-dimensional, if the spatial
        dimensions aren't defined at graph construction time, if the spatial
        dimensions aren't square, or if the spatial dimensions aren't a power of
        two.
    """
    normalizer_fn = tf.contrib.layers.batch_norm
    normalizer_fn_args = {
        'is_training': is_training,
        'decay': 0.9,
        'epsilon': 1e-5,
        'scale': True,
        'zero_debias_moving_mean': True,
        'fused': fused_batch_norm,
    }

    _validate_image_inputs(inputs)
    inp_shape = inputs.get_shape().as_list()[1]

    end_points = {}
    with tf.variable_scope(scope, values=[inputs], reuse=reuse) as scope:
        with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
            with slim.arg_scope([conv2d],
                                padding='VALID',
                                activation_fn=tf.nn.leaky_relu):
                net = inputs
                num_layers = int(log(inp_shape)) - int(log(4))
                for i in range(num_layers):
                    scope = 'conv%i' % (i + 1)
                    current_depth = depth * 2 ** i
                    current_depth = current_depth if current_depth <= 512 else 512
                    kernel_size = [4, 4,
                                   net.get_shape().as_list()[3],
                                   current_depth]
                    # No normalizing input layer
                    normalizer_fn_ = None if i == 0 else normalizer_fn
                    net = conv2d(net, kernel_size,
                                 stride=[1, 2, 2, 1],
                                 spectral_norm=True,
                                 normalizer_fn=None,
                                 scope_name=scope)
                    end_points[scope] = net

                # logits = slim.conv2d(net, 1, kernel_size=1, stride=1,
                #                      padding='VALID',
                #                      normalizer_fn=None, activation_fn=None)
                # logits = tf.reshape(logits, [-1, 1])
                # end_points['logits'] = logits

                return None, end_points


def _last_conv_layer(end_points):
    """"Returns the last convolutional layer from an endpoints dictionary."""
    conv_list = [k if k[:4] == 'conv' else '' for k in end_points.keys()]
    conv_list.sort()
    return end_points[conv_list[-1]]


def discriminator(img, conditioning, apply_batch_norm=False):
    """Discriminator for CIFAR images.
    Args:
      img: A Tensor of shape [batch size, width, height, channels], that can be
        either real or generated. It is the discriminator's goal to distinguish
        between the two.
      conditioning: `mu` output of networks.augment for the text embeddings of 
        the images in the given `img` batch.
    Returns:
      A 1D Tensor of shape [batch size] representing the confidence that the
      images are real. The output can lie in [-inf, inf], with positive values
      indicating high confidence that the images are real.
    """
    depth = 64
    _, end_points = dcgan_discriminator(img, depth=depth,
                                        is_training=apply_batch_norm)

    # TODO(joppe): have dcgan_discriminator return the right logits
    net = _last_conv_layer(end_points)
    embedding_dim = 128
    conditioning = tf.reshape(conditioning, [-1, embedding_dim, 1, 1])
    conditioning = tf.tile(conditioning, [1, 1, 4, 4])
    batch_size = img.get_shape().as_list()[0]
    conditioned = tf.concat(
        [conditioning, tf.reshape(net, [batch_size, embedding_dim, 4, 4])],
        1)  # -1

    # Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
    activation_fn_ = tf.nn.leaky_relu
    kernel_size = [3, 3,
                   conditioned.get_shape().as_list()[3],
                   depth]
    conditioned = conv2d(conditioned, kernel_size, [1, 2, 2, 1],
                         padding='VALID',
                         activation_fn=activation_fn_)

    # last layer dcgan_discriminator
    conditioned_logits = slim.conv2d(conditioned, 1, kernel_size=1, stride=1,
                                     padding='VALID',
                                     normalizer_fn=None, activation_fn=None)
    conditioned_logits = tf.reshape(conditioned_logits, [-1, 1])

    unconditoned_logits = slim.conv2d(net, 1, kernel_size=1, stride=1,
                                      padding='VALID',
                                      normalizer_fn=None, activation_fn=None)
    unconditoned_logits = tf.reshape(unconditoned_logits, [-1, 1])

    return conditioned_logits, unconditoned_logits
