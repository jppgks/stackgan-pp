from collections import namedtuple

import tensorflow as tf

import networks
import tfstackgan

tfstackgan_losses = tfstackgan.losses
from data import data_provider

# Useful aliases.
tfgan = tf.contrib.gan
tfgan_losses = tfgan.losses

flags = tf.flags
flags.DEFINE_integer('stack_depth', 3,
                     'Defines the size of the GAN stack: ' +
                     'the number of (discriminator, generator) stages.')

flags.DEFINE_integer('batch_size', 8,
                     'The number of images in each batch.')  # 24

flags.DEFINE_integer('noise_dim', 64,  # 100
                     'Dimension of the noise that\'s input for each generator.')

flags.DEFINE_string('loss_fn', 'minimax',
                    'Type of loss function: \'minimax\' or \'wasserstein\'.')

flags.DEFINE_float('color_loss', 50, 'Weight of color loss (see paper).')

flags.DEFINE_float('uncond_loss_coeff', 1.0, 'Weight of unconditional loss.')

flags.DEFINE_float('generator_lr', 0.0001, 'Generator learning rate.')

flags.DEFINE_boolean('do_lr_decay', True,
                     'Decay the generator\'s learning rate.')

flags.DEFINE_integer('decay_steps', 100000,
                     'After how many steps to decay the learning rate.')

flags.DEFINE_float('decay_rate', 0.9,
                   'How much of the learning rate to retain when decaying.')

flags.DEFINE_float('discriminator_lr', 0.0001,
                   'Discriminator learning rate.')

flags.DEFINE_float('gradient_penalty', None, 'Gradient penalty weight.')

flags.DEFINE_boolean('apply_batch_norm', False,
                     'Apply batch normalization.')

flags.DEFINE_string('train_log_dir', '/tmp/cifar-stackgan-3stage',
                    'Directory to write event logs and checkpoints to. Will '
                    'continue training from a checkpoint in this directory if '
                    'one exists.')

flags.DEFINE_string('image_dataset_dir', '',
                    '')  # TODO: docstring

flags.DEFINE_string('text_dataset_dir', '',
                    '')  # TODO: docstring

flags.DEFINE_integer('max_number_of_steps', 1000000,
                     # num_samples / batch_size * 5 * 120 = 180000
                     'The maximum number of gradient steps.')
FLAGS = flags.FLAGS


def main(_):
    # Create log dir.
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)

    # Get training data.
    images, captions_text, captions_embedding = data_provider.get_training_data_iterator(
        FLAGS.batch_size,
        FLAGS.image_dataset_dir,
        FLAGS.text_dataset_dir,
        FLAGS.stack_depth)
    summary_img_grid_size = 2
    tf.summary.text('Captions', captions_text[:summary_img_grid_size ** 2])

    # Define noise node, instantiate GANModel tuples and keep pointer
    # to a named tuple of GAN models.
    noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dim])
    augmented_conditioning, mu, logvar = networks.augment(captions_embedding)
    gan_models = []
    for stage in range(FLAGS.stack_depth):
        kwargs = {
            'generator_input_fn': _get_generator_input_for_stage(gan_models,
                                                                 stage,
                                                                 noise,
                                                                 augmented_conditioning),
            'real_data': _get_real_data_for_stage(images, stage),
            'disc_conditioning': mu,
            'generator_super_scope': gan_models[
                -1].generator_scope if stage > 0 else None,
            'stage': stage,
            'apply_batch_norm': FLAGS.apply_batch_norm}
        current_model = tfstackgan.gan_model(
            networks.generator,
            networks.discriminator,
            **kwargs)
        gan_models.append(current_model)
        tfgan.eval.add_gan_model_image_summaries(current_model, grid_size=summary_img_grid_size)
    model_names = ['stage_' + str(i) for i in range(FLAGS.stack_depth)]
    GANModels = namedtuple('GANModels', model_names)
    gan_models = GANModels(*gan_models)

    # Instantiate losses.
    # Need to optimize discriminator at each stage independently,
    # so we add a loss for each discriminator to this list.
    # Separate optimizers need to optimize
    # each loss in this list (DiscriminatorTrainOps).
    # Only a need for one overall generator loss, as generator is optimized once
    # per training step in which all discriminator stages are optimized.
    if FLAGS.loss_fn == 'minimax':
        discriminator_loss_fn = tfstackgan_losses.minimax_discriminator_loss
        generator_loss_fn = tfstackgan_losses.minimax_generator_loss
    else:
        discriminator_loss_fn = tfstackgan_losses.wasserstein_discriminator_loss
        generator_loss_fn = tfstackgan_losses.wasserstein_generator_loss

    dis_losses = []
    for stage in range(FLAGS.stack_depth):
        with tf.variable_scope(gan_models[stage].discriminator_scope):
            with tf.name_scope(
                    gan_models[stage].discriminator_scope.original_name_scope):
                print(tf.get_variable_scope().name)
                with tf.variable_scope('losses'):
                    current_stage_dis_loss = tfstackgan.dis_loss(
                        gan_models[stage],
                        discriminator_loss_fn=discriminator_loss_fn,
                        gradient_penalty_weight=FLAGS.gradient_penalty)
                    dis_losses.append(current_stage_dis_loss)
    with tf.variable_scope(gan_models[-1].generator_scope):
        with tf.name_scope(gan_models[-1].generator_scope.original_name_scope):
            with tf.variable_scope('loss'):
                gen_loss_tuple = tfstackgan.gen_loss(
                    gan_models,
                    generator_loss_fn=generator_loss_fn,
                    color_loss_weight=FLAGS.color_loss,
                    uncond_loss_coeff=FLAGS.uncond_loss_coeff,
                    mu=mu,
                    logvar=logvar, )

    # Instantiate train ops.
    # Generator's learning rate decays, while discriminator's stays constant.
    # Both learning rates are logged to TensorBoard.
    # 1 discriminator and 1 generator optimizer are created and used to
    # construct train ops. Then all train ops, including global step increment
    # op, are stored in a GANTrainOps tuple.
    with tf.name_scope(gan_models[-1].generator_scope.original_name_scope):
        if FLAGS.do_lr_decay:
            generator_lr = tf.train.exponential_decay(
                learning_rate=FLAGS.generator_lr,
                global_step=tf.train.get_or_create_global_step(),
                decay_steps=FLAGS.decay_steps,
                decay_rate=FLAGS.decay_rate,
                staircase=True)
        else:
            generator_lr = FLAGS.generator_lr
    tf.summary.scalar('generator_lr', generator_lr)
    tf.summary.scalar('discriminator_lr', FLAGS.discriminator_lr)

    gen_opt, dis_opt = _optimizer(generator_lr, FLAGS.discriminator_lr)

    gen_train_op = tfstackgan.generator_train_op(
        gan_models[-1],
        gen_loss_tuple,
        gen_opt,
        summarize_gradients=True,
        colocate_gradients_with_ops=True,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    disc_train_ops = tfstackgan.discriminator_train_ops(
        gan_models,
        dis_losses,
        dis_opt,
        summarize_gradients=True,
        colocate_gradients_with_ops=True,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        # transform_grads_fn=tf.contrib.training.clip_gradient_norms_fn(1e3)
    )
    global_step = tf.train.get_or_create_global_step()
    global_step_inc_op = global_step.assign_add(1)
    train_ops = tfgan.GANTrainOps(gen_train_op, disc_train_ops,
                                  global_step_inc_op)

    # Actual GAN training. Run the alternating training loop.
    status_message = tf.string_join(
        ['Starting train step: ',
         tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')
    if FLAGS.max_number_of_steps == 0: return
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tfgan.gan_train(
        train_ops,
        FLAGS.train_log_dir,
        get_hooks_fn=tfstackgan.get_sequential_train_hooks(),
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
               tf.train.LoggingTensorHook([status_message], every_n_iter=1000)],
        save_summaries_steps=100,
        config=config
        #    master=FLAGS.master,
        #    is_chief=FLAGS.task == 0
    )


def _get_generator_input_for_stage(models, stage, noise_sample, conditioning):
    assert isinstance(stage, int)

    def get_input():
        is_init_stage = not bool(stage)
        # Input into first stage is z ~ p_noise + conditioning.
        # Input for stage i generator is the hidden code outputted by
        # stage (i-1) + conditioning.
        noise = noise_sample if is_init_stage else models[
            stage - 1].generator_hidden_code
        return is_init_stage, noise, conditioning

    return get_input


def _get_real_data_for_stage(images, i):
    resolution = 2 ** (6 + i)
    current_res_images = tf.image.resize_images(images,
                                                size=[resolution, resolution])
    current_res_images.set_shape([FLAGS.batch_size, resolution, resolution, 3])
    return current_res_images


def _optimizer(gen_lr, dis_lr):
    if FLAGS.loss_fn == 'minimax':
        kwargs = {'beta1': 0.5, 'beta2': 0.999}
        generator_opt = tf.train.AdamOptimizer(gen_lr, **kwargs)
        discriminator_opt = tf.train.AdamOptimizer(dis_lr, **kwargs)
    else:
        generator_opt = tf.train.RMSPropOptimizer(gen_lr, decay=.9,
                                                  momentum=0.1)
        discriminator_opt = tf.train.RMSPropOptimizer(dis_lr, decay=.95,
                                                      momentum=0.1)
    return generator_opt, discriminator_opt


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
