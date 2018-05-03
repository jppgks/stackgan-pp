# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A TFGAN-backed GAN Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.contrib.gan.python import namedtuples as tfgan_tuples
from tensorflow.contrib.gan.python import train as tfgan_train
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.canned import head
from tensorflow.python.framework import ops, constant_op
from tensorflow.python.training import training_util
from tensorflow.python.ops import variable_scope, control_flow_ops

from tfstackgan import train as tfstackgan_train
from tfstackgan import namedtuples as tfstackgan_tuples
from tfstackgan.python.metrics.python import util_impl as metrics_lib

__all__ = [
    'StackGANHead',
    'stackgan_head',
]


def _summary_key(head_name, val):
    return '%s/%s' % (val, head_name) if head_name else val


def stackgan_head(generator_loss_fn, discriminator_loss_fn, generator_optimizer,
                  discriminator_optimizer, uncond_loss_coeff,
                  color_loss_weight, gradient_penalty_weight,
                  use_loss_summaries=True, batch_size=0, num_inception_images=0,
                  get_hooks_fn=tfstackgan_train.get_sequential_train_hooks(),
                  name=None):
    """Creates a `GANHead`.
  
    Args:
      generator_loss_fn: A TFGAN loss function for the generator. Takes a
        `GANModel` and returns a scalar.
      discriminator_loss_fn: Same as `generator_loss_fn`, but for the
        discriminator.
      generator_optimizer: The optimizer for generator updates.
      discriminator_optimizer: Same as `generator_optimizer`, but for the
        discriminator updates.
      use_loss_summaries: If `True`, add loss summaries. If `False`, does not.
          If `None`, uses defaults.
      get_hooks_fn: A function that takes a GANTrainOps tuple and returns a list
          of hooks.
      name: name of the head. If provided, summary and metrics keys will be
        suffixed by `"/" + name`.
  
    Returns:
      An instance of `GANHead`.
    """
    return StackGANHead(generator_loss_fn=generator_loss_fn,
                        discriminator_loss_fn=discriminator_loss_fn,
                        generator_optimizer=generator_optimizer,
                        discriminator_optimizer=discriminator_optimizer,
                        uncond_loss_coeff=uncond_loss_coeff,
                        color_loss_weight=color_loss_weight,
                        gradient_penalty_weight=gradient_penalty_weight,
                        use_loss_summaries=use_loss_summaries,
                        batch_size=batch_size,
                        num_inception_images=num_inception_images,
                        get_hooks_fn=get_hooks_fn,
                        name=name)


class StackGANHead(head._Head):  # pylint: disable=protected-access
    """`Head` for a GAN."""

    def __init__(self, generator_loss_fn, discriminator_loss_fn,
                 generator_optimizer, discriminator_optimizer,
                 uncond_loss_coeff, color_loss_weight, gradient_penalty_weight,
                 use_loss_summaries=True,
                 batch_size=0,
                 num_inception_images=0,
                 get_hooks_fn=None,
                 name=None):
        """`Head` for GAN training.
    
        Args:
          generator_loss_fn: A TFGAN loss function for the generator.
          discriminator_loss_fn: Same as `generator_loss_fn`, but for the
          discriminator.
          generator_optimizer: The optimizer for generator updates.
          discriminator_optimizer: Same as `generator_optimizer`, but for the
            discriminator updates.
          use_loss_summaries: If `True`, add loss summaries. If `False`, does not.
            If `None`, uses defaults.
          get_hooks_fn: A function that takes a GANTrainOps tuple and returns a list
            of hooks. Defaults to `train.get_sequential_train_hooks()`
          name: name of the head. If provided, summary and metrics keys will be
            suffixed by `"/" + name`.
        """
        if get_hooks_fn is None:
            get_hooks_fn = tfstackgan_train.get_sequential_train_hooks()
        # TODO(joelshor): Validate inputs.

        if use_loss_summaries in [True, False]:
            generator_loss_fn = functools.partial(
                generator_loss_fn, add_summaries=use_loss_summaries)
            discriminator_loss_fn = functools.partial(
                discriminator_loss_fn, add_summaries=use_loss_summaries)
        self._generator_loss_fn = generator_loss_fn
        self._discriminator_loss_fn = discriminator_loss_fn
        self._uncond_loss_coeff = uncond_loss_coeff
        self._color_loss_weight = color_loss_weight
        self._gradient_penalty_weight = gradient_penalty_weight
        self._generator_optimizer = generator_optimizer
        self._discriminator_optimizer = discriminator_optimizer
        self._get_hooks_fn = get_hooks_fn
        self._batch_size = batch_size
        self._num_inception_images = num_inception_images
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def logits_dimension(self):
        return None

    def create_loss(self, features, mode, logits, labels):
        """Returns a GANLoss tuple from the provided GANModel.
    
        See `Head` for more details.
    
        Args:
          features: Input `dict` of `Tensor` objects. Unused.
          mode: Estimator's `ModeKeys`.
          logits: A GANModel tuple.
          labels: Must be `None`.
    
        Returns:
          A GANLoss tuple.
    
        """
        _validate_logits_and_labels(logits, labels)
        del mode, labels, features  # unused for this head.
        gan_models = logits  # rename variable for clarity

        # Instantiate losses.
        # Need to optimize discriminator at each stage independently,
        # so we add a loss for each discriminator to this list.
        # Separate optimizers need to optimize
        # each loss in this list (DiscriminatorTrainOps).
        # Only a need for one overall generator loss, as generator is optimized once
        # per training step in which all discriminator stages are optimized.
        dis_losses = []
        for stage in range(len(gan_models)):
            current_model = gan_models[stage]
            with variable_scope.variable_scope(
                    current_model.discriminator_scope):
                with ops.name_scope(
                        current_model.discriminator_scope.original_name_scope):
                    print(variable_scope.get_variable_scope().name)
                    with variable_scope.variable_scope('losses'):
                        current_stage_dis_loss = tfstackgan_train.dis_loss(
                            current_model,
                            discriminator_loss_fn=self._discriminator_loss_fn,
                            gradient_penalty_weight=self._gradient_penalty_weight)
                        dis_losses.append(
                            current_stage_dis_loss.discriminator_loss)
        with variable_scope.variable_scope(gan_models[-1].generator_scope):
            with ops.name_scope(
                    gan_models[-1].generator_scope.original_name_scope):
                with variable_scope.variable_scope('loss'):
                    gen_loss_tuple = tfstackgan_train.gen_loss(
                        gan_models,
                        generator_loss_fn=self._generator_loss_fn,
                        color_loss_weight=self._color_loss_weight,
                        uncond_loss_coeff=self._uncond_loss_coeff,
                        mu=gan_models[-1].mu,
                        logvar=gan_models[-1].logvar)

        generator_loss = gen_loss_tuple.generator_loss

        return tfgan_tuples.GANLoss(
            generator_loss=generator_loss,
            discriminator_loss=dis_losses)

    def create_estimator_spec(self, features, mode, logits, labels=None,
                              train_op_fn=tfstackgan_train.gan_train_ops,
                              **kwargs):
        """Returns `EstimatorSpec` that a model_fn can return.
    
        See `Head` for more details.
    
        Args:
          features: Must be `None`.
          mode: Estimator's `ModeKeys`.
          logits: A GANModel tuple.
          labels: Must be `None`.
          train_op_fn: Function that takes a GANModel, GANLoss, generator optimizer,
            and discriminator optimizer, and returns a `GANTrainOps` tuple. For
            example, this function can come from TFGAN's `train.py` library, or can
            be custom.
    
        Returns:
          `EstimatorSpec`.
    
        Raises:
          ValueError: If `features` isn't `None`.
          ValueError: If `train_op_fn` isn't provided in train mode.
          :param **kwargs: 
        """
        _validate_logits_and_labels(logits, labels)
        if features is not None:
            raise ValueError('`features` should be `None`. Instead, found: %s' %
                             features)
        gan_models = logits  # rename variable for clarity
        with ops.name_scope('StackGANHead'):
            if mode == model_fn_lib.ModeKeys.PREDICT:
                gan_model = gan_models  # rename variable for clarity
                return model_fn_lib.EstimatorSpec(
                    mode=model_fn_lib.ModeKeys.PREDICT,
                    predictions=gan_model.generated_data)
            elif mode == model_fn_lib.ModeKeys.EVAL:
                real_data = gan_models[-1].real_data
                generated_data = gan_models[-1].generated_data
                gan_loss = self.create_loss(
                    features=None, mode=mode, logits=gan_models, labels=None)
                scalar_loss = gan_loss.generator_loss + sum(
                    gan_loss.discriminator_loss)
                return model_fn_lib.EstimatorSpec(
                    mode=model_fn_lib.ModeKeys.EVAL,
                    predictions=generated_data,
                    loss=scalar_loss,
                    eval_metric_ops=self._eval_metric_ops(
                        real_data, generated_data, self._batch_size,
                        num_inception_images=self._num_inception_images))
            elif mode == model_fn_lib.ModeKeys.TRAIN:
                if train_op_fn is None:
                    raise ValueError('train_op_fn can not be None.')
                gan_loss = self.create_loss(None, mode, gan_models, None)
                scalar_loss = gan_loss.generator_loss + sum(
                    gan_loss.discriminator_loss)
                train_ops = train_op_fn(gan_models, gan_loss,
                                        self._generator_optimizer,
                                        self._discriminator_optimizer)
                train_ops = control_flow_ops.group(train_ops.generator_train_op,
                                                   train_ops.discriminator_train_op)

                return model_fn_lib.EstimatorSpec(
                    loss=scalar_loss,
                    mode=model_fn_lib.ModeKeys.TRAIN,
                    train_op=train_ops,  # train_ops.global_step_inc_op,
                    training_hooks=None)  # training_hooks
            else:
                raise ValueError('Mode not recognized: %s' % mode)

    def _eval_metric_ops(self, real_data, generated_data,
                         batch_size, num_inception_images):
        with ops.name_scope(None, 'metrics'):
            metric_ops = {
                # Estimator already adds a metric for loss.
                _summary_key(self._name, 'inception_score'):
                    metrics_lib.get_inception_scores(
                        generated_data,
                        batch_size,
                        num_inception_images),
                _summary_key(self._name, 'frechet_inception_distance'):
                    metrics_lib.get_frechet_inception_distance(
                        real_data,
                        generated_data,
                        batch_size,
                        num_inception_images),
                _summary_key(self._name, 'sliced_wasserstein_distance'):
                    metrics_lib.get_sliced_wasserstein_distance(
                        real_data, generated_data),
            }
            return metric_ops


def _validate_logits_and_labels(logits, labels):
    if labels is not None:
        raise ValueError(
            '`GANHead`\'s `create_estimator_spec` input `labels` must '
            'be `None`. Instead, found: %s' % labels)

    if not all(isinstance(elem, tfstackgan_tuples.StackGANModel)
               for elem in logits):
        raise ValueError(
            '`GANHead`\'s `create_estimator_spec` input `logits` must '
            'be an instance of a `StackGANModel`. Instead, found: %s' %
            logits)
