import tensorflow as tf

__all__ = [
    'color_loss',
    'wasserstein_generator_loss',
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
        uncond_loss_coeff = 1.0
        uncond_loss = uncond_loss_coeff * uncond_loss

        loss = cond_loss + uncond_loss

        if add_summaries:
            tf.summary.scalar('generator_wass_loss', loss)

    return loss
