import tensorflow as tf

__all__ = [
    'color_loss',
    'wasserstein_generator_loss',
    'wasserstein_discriminator_loss',
    'minimax_generator_loss',
    'minimax_discriminator_loss',
    'kl_loss',
]


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


# Color loss (requires gan_models to be in scope)
def color_loss(weight, models):
    means = []
    covariances = []
    total_color_loss = 0

    for gan_model in models:
        mu, cov = _compute_mean_covariance(gan_model.generated_data)
        means.append(mu)
        covariances.append(cov)
    stack_depth = len(models)
    assert len(means) == len(covariances) == stack_depth

    for i in range(stack_depth - 1):
        # Elements at position i and (i + 1) aren't
        # the last two elements in the list.
        like_mu = weight * tf.losses.mean_squared_error(
            means[i], means[i + 1])
        like_cov = weight * 5 * tf.losses.mean_squared_error(
            covariances[i], covariances[i + 1])
        total_color_loss += like_mu + like_cov

    return weight * total_color_loss

    #  sum_mu = tf.summary.scalar('G_like_mu2', like_mu2.data[0])
    #  self.summary_writer.add_summary(sum_mu, count)
    #  sum_cov = summary.scalar('G_like_cov2', like_cov2.data[0])
    #  self.summary_writer.add_summary(sum_cov, count)
    #  if self.num_Ds > 2:
    #      sum_mu = summary.scalar('G_like_mu1', like_mu1.data[0])
    #      self.summary_writer.add_summary(sum_mu, count)
    #      sum_cov = summary.scalar('G_like_cov1', like_cov1.data[0])
    #      self.summary_writer.add_summary(sum_cov, count)


# Wasserstein losses from `Wasserstein GAN` (https://arxiv.org/abs/1701.07875).
def wasserstein_generator_loss(
        discriminator_gen_outputs_cond,
        discriminator_gen_outputs_uncond,
        uncond_loss_coeff,
        weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    """Wasserstein generator loss for GANs.
    See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.
    Args:
      discriminator_gen_outputs: Discriminator output on generated data. Expected
        to be in the range of (-inf, inf).
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `discriminator_gen_outputs`, and must be broadcastable to
        `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
        the same as the corresponding dimension).
      scope: The scope for the operations performed in computing the loss.
      loss_collection: collection to which this loss will be added.
      reduction: A `tf.losses.Reduction` to apply to loss.
      add_summaries: Whether or not to add detailed summaries for the loss.
    Returns:
      A loss Tensor. The shape depends on `reduction`.
    """
    with tf.name_scope(scope, 'generator_wasserstein_loss', (
            discriminator_gen_outputs_cond, weights)) as scope:
        discriminator_gen_outputs_cond = tf.to_float(
            discriminator_gen_outputs_cond)

        cond_loss = - discriminator_gen_outputs_cond
        cond_loss = tf.losses.compute_weighted_loss(cond_loss, weights, scope,
                                                    loss_collection, reduction)

        discriminator_gen_outputs_uncond = tf.to_float(
            discriminator_gen_outputs_uncond)

        uncond_loss = - discriminator_gen_outputs_uncond
        uncond_loss = tf.losses.compute_weighted_loss(uncond_loss, weights,
                                                      scope, loss_collection,
                                                      reduction)
        uncond_loss = uncond_loss_coeff * uncond_loss

        loss = cond_loss + uncond_loss

        if add_summaries:
            tf.summary.scalar('generator_wass_loss', loss)

    return loss


def wasserstein_discriminator_loss(
        discriminator_real_outputs_cond,
        discriminator_real_outputs_uncond,
        discriminator_gen_outputs_cond,
        discriminator_gen_outputs_uncond,
        uncond_loss_coeff,
        real_weights=1.0,
        generated_weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    """Wasserstein discriminator loss for GANs.
    See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.
    Args:
      discriminator_real_outputs: Discriminator output on real data.
      discriminator_gen_outputs: Discriminator output on generated data. Expected
        to be in the range of (-inf, inf).
      real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `discriminator_real_outputs`, and must be broadcastable to
        `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
        the same as the corresponding dimension).
      generated_weights: Same as `real_weights`, but for
        `discriminator_gen_outputs`.
      scope: The scope for the operations performed in computing the loss.
      loss_collection: collection to which this loss will be added.
      reduction: A `tf.losses.Reduction` to apply to loss.
      add_summaries: Whether or not to add summaries for the loss.
    Returns:
      A loss Tensor. The shape depends on `reduction`.
    """
    with tf.name_scope(scope, 'discriminator_wasserstein_loss', (
            discriminator_real_outputs_cond, discriminator_real_outputs_uncond,
            discriminator_gen_outputs_cond, discriminator_gen_outputs_uncond,
            real_weights,
            generated_weights)) as scope:
        # Conditional loss
        discriminator_real_outputs_cond = tf.to_float(
            discriminator_real_outputs_cond)
        discriminator_gen_outputs_cond = tf.to_float(
            discriminator_gen_outputs_cond)
        discriminator_real_outputs_cond.shape.assert_is_compatible_with(
            discriminator_gen_outputs_cond.shape)

        loss_on_generated_cond = tf.losses.compute_weighted_loss(
            discriminator_gen_outputs_cond, generated_weights, scope,
            loss_collection=None, reduction=reduction)
        loss_on_real_cond = tf.losses.compute_weighted_loss(
            discriminator_real_outputs_cond, real_weights, scope,
            loss_collection=None,
            reduction=reduction)
        cond_loss = loss_on_generated_cond - loss_on_real_cond

        # Unconditional loss
        discriminator_real_outputs_uncond = tf.to_float(
            discriminator_real_outputs_uncond)
        discriminator_gen_outputs_uncond = tf.to_float(
            discriminator_gen_outputs_uncond)
        discriminator_real_outputs_uncond.shape.assert_is_compatible_with(
            discriminator_gen_outputs_uncond.shape)

        loss_on_generated_uncond = tf.losses.compute_weighted_loss(
            discriminator_gen_outputs_uncond, generated_weights, scope,
            loss_collection=None, reduction=reduction)
        loss_on_real_uncond = tf.losses.compute_weighted_loss(
            discriminator_real_outputs_uncond, real_weights, scope,
            loss_collection=None,
            reduction=reduction)
        uncond_loss = loss_on_generated_uncond - loss_on_real_uncond

        # Total loss
        loss = cond_loss - (uncond_loss_coeff * uncond_loss)
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.summary.scalar('discriminator_gen_wass_loss',
                              loss_on_generated_cond)
            tf.summary.scalar('discriminator_real_wass_loss', loss_on_real_cond)
            tf.summary.scalar('discriminator_wass_loss', loss)

    return loss


def minimax_generator_loss(
        discriminator_gen_outputs_cond,
        discriminator_gen_outputs_uncond,
        uncond_loss_coeff,
        label_smoothing=0.0,
        weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    with tf.name_scope(scope, 'generator_modified_loss',
                        [discriminator_gen_outputs_cond,
                         discriminator_gen_outputs_uncond]) as scope:
        loss_cond = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_gen_outputs_cond),
            discriminator_gen_outputs_cond, weights, label_smoothing, scope,
            loss_collection, reduction)

        loss_uncond = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_gen_outputs_uncond),
            discriminator_gen_outputs_uncond, weights, label_smoothing, scope,
            loss_collection, reduction)

        loss = loss_cond + uncond_loss_coeff * loss_uncond

        if add_summaries:
            tf.summary.scalar('generator_modified_loss', loss)

    return loss


def minimax_discriminator_loss(
        discriminator_real_outputs_cond,
        discriminator_real_outputs_uncond,
        discriminator_gen_outputs_cond,
        discriminator_gen_outputs_uncond,
        uncond_loss_coeff,
        label_smoothing=0.25,
        real_weights=1.0,
        generated_weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    with tf.name_scope(scope, 'discriminator_minimax_loss', (
            discriminator_real_outputs_cond, discriminator_real_outputs_uncond,
            discriminator_gen_outputs_cond, discriminator_gen_outputs_uncond,
            real_weights,
            generated_weights, label_smoothing)) as scope:
        # Conditional loss

        # -log((1 - label_smoothing) - sigmoid(D(x)))
        loss_on_real_cond = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_real_outputs_cond),
            discriminator_real_outputs_cond, real_weights, label_smoothing,
            scope,
            loss_collection=None, reduction=reduction)
        # -log(- sigmoid(D(G(x))))
        loss_on_generated_cond = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(discriminator_gen_outputs_cond),
            discriminator_gen_outputs_cond, generated_weights, scope=scope,
            loss_collection=None, reduction=reduction)

        cond_loss = loss_on_real_cond + loss_on_generated_cond

        # Unconditional loss

        # -log((1 - label_smoothing) - sigmoid(D(x)))
        loss_on_real_uncond = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_real_outputs_uncond),
            discriminator_real_outputs_uncond, real_weights, label_smoothing,
            scope,
            loss_collection=None, reduction=reduction)
        # -log(- sigmoid(D(G(x))))
        loss_on_generated_uncond = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(discriminator_gen_outputs_uncond),
            discriminator_gen_outputs_uncond, generated_weights, scope=scope,
            loss_collection=None, reduction=reduction)

        uncond_loss = loss_on_generated_uncond + loss_on_real_uncond

        # Total loss
        loss = cond_loss + (uncond_loss_coeff * uncond_loss)
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            # tf.summary.scalar('discriminator_gen_minimax_loss', loss_on_generated)
            # tf.summary.scalar('discriminator_real_minimax_loss', loss_on_real)
            tf.summary.scalar('discriminator_minimax_loss', loss)

    return loss


def kl_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = 1 + logvar - (mu ** 2) - tf.exp(logvar)
    kld = tf.reduce_mean(kld) * -0.5
    return kld
