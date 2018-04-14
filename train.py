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

    # 1) Define losses,
    if FLAGS.loss_fn == 'minimax':
        discriminator_loss_fn = tfstackgan_losses.minimax_discriminator_loss
        generator_loss_fn = tfstackgan_losses.minimax_generator_loss
    else:
        discriminator_loss_fn = tfstackgan_losses.wasserstein_discriminator_loss
        generator_loss_fn = tfstackgan_losses.wasserstein_generator_loss

    # 2) Define optimizers,
    gen_opt_fn = _get_gen_opt_fn(FLAGS.generator_lr, FLAGS.do_lr_decay,
                                 FLAGS.decay_steps, FLAGS.decay_rate)
    dis_opt_fn = _get_dis_opt_fn(FLAGS.discriminator_lr)

    # 3) Do some configuration,
    # - Session config
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    # - Distribution config
    distribution = tf.contrib.distribute.OneDeviceStrategy(
        device='/device:GPU:0'
    )

    # - Estimator config
    run_config = tf.estimator.RunConfig(
        train_distribute=distribution,
        session_config=sess_config,
        save_summary_steps=100,
        log_step_count_steps=100)

    # ???
    # Initialize GANEstimator with options and hyperparameters.
    stackgan_estimator = tfstackgan.estimator.StackGANEstimator(
        model_dir=FLAGS.train_log_dir,
        # Model params
        stack_depth=FLAGS.stack_depth,
        batch_size=FLAGS.batch_size,
        noise_dim=FLAGS.noise_dim,
        # Networks
        generator_fn=networks.generator,
        discriminator_fn=networks.discriminator,
        apply_batch_norm=FLAGS.apply_batch_norm,
        # Losses
        generator_loss_fn=generator_loss_fn,
        discriminator_loss_fn=discriminator_loss_fn,
        uncond_loss_coeff=FLAGS.uncond_loss_coeff,
        color_loss_weight=FLAGS.color_loss,
        gradient_penalty_weight=FLAGS.gradient_penalty,
        # Optimizers
        generator_optimizer=gen_opt_fn,
        discriminator_optimizer=dis_opt_fn,
        # Config
        add_summaries=tfstackgan.estimator.SummaryType.IMAGES,
        config=run_config)

    # PROFIT!
    # Actual GAN training. Run the alternating training loop.
    train_input_fn = _get_train_input_fn()

    hooks = [tf.train.ProfilerHook(save_steps=100,
                                   show_dataflow=True,
                                   show_memory=False,
                                   output_dir=FLAGS.train_log_dir), ]

    stackgan_estimator.train(train_input_fn,
                             max_steps=FLAGS.max_number_of_steps,
                             hooks=hooks)


def _get_train_input_fn():
    def train_input_fn():
        input_dataset, captions_text = data_provider.get_training_datasets(
            FLAGS.batch_size,
            FLAGS.noise_dim,
            FLAGS.image_dataset_dir,
            FLAGS.text_dataset_dir,
            FLAGS.stack_depth)
        # Log captions to TensorBoard
        summary_img_grid_size = 2
        tf.summary.text('Captions',
                        captions_text[:summary_img_grid_size ** 2])

        return input_dataset

    return train_input_fn


def _get_predict_input_fn(batch_size, noise_dims):
    return _get_train_input_fn()


def _get_dis_opt_fn(dis_lr):
    tf.summary.scalar('discriminator_lr', dis_lr)

    def dis_opt_fn():
        if FLAGS.loss_fn == 'minimax':
            kwargs = {'beta1': 0.5, 'beta2': 0.999}
            discriminator_opt = tf.contrib.optimizer_v2.AdamOptimizer(dis_lr,
                                                                      **kwargs)
        else:
            discriminator_opt = tf.contrib.optimizer_v2.RMSPropOptimizer(dis_lr,
                                                                         decay=.95,
                                                                         momentum=0.1)
        return discriminator_opt

    return dis_opt_fn


def _get_gen_opt_fn(gen_lr, do_lr_decay, decay_steps, decay_rate):
    def gen_opt_fn():
        with tf.name_scope(
                'Generator'):  # gan_models[-1].generator_scope.original_name_scope
            if do_lr_decay:
                generator_lr = tf.train.exponential_decay(
                    learning_rate=gen_lr,
                    global_step=tf.train.get_or_create_global_step(),
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True)
            else:
                generator_lr = gen_lr
            with tf.device('/cpu:0'):
                tf.summary.scalar('generator_lr', generator_lr)

        if FLAGS.loss_fn == 'minimax':
            kwargs = {'beta1': 0.5, 'beta2': 0.999}
            generator_opt = tf.contrib.optimizer_v2.AdamOptimizer(generator_lr,
                                                                  **kwargs)
        else:
            generator_opt = tf.contrib.optimizer_v2.RMSPropOptimizer(
                generator_lr, decay=.9,
                momentum=0.1)
        return generator_opt

    return gen_opt_fn


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
