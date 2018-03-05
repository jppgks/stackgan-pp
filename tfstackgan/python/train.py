from contextlib import ExitStack

import tensorflow as tf
from tensorflow.contrib.gan.python.train import _convert_tensor_or_l_or_d, \
    _use_aux_loss, _validate_aux_loss_weight, RunTrainOpsHook
from tensorflow.contrib.training.python.training import training

import tfstackgan

# Useful aliases.
tfgan = tf.contrib.gan
tfgan_losses = tfgan.losses
tfstackgan_losses = tfstackgan.losses

__all__ = [
    'gan_model',
    'dis_loss',
    'gen_loss',
    'generator_train_op',
    'discriminator_train_ops',
    'get_sequential_train_hooks',
]


def _get_or_create_gen_super_scope(super_scope):
    if not super_scope:
        with ExitStack() as stack:
            super_scope = stack.enter_context(
                tf.variable_scope('Generator', reuse=tf.AUTO_REUSE))
    return super_scope


def gan_model(  # Lambdas defining models.
        generator_fn,
        discriminator_fn,
        # Real data and discriminator conditioning.
        real_data,
        disc_conditioning,
        generator_input_fn,
        # Stage (depth in stack).
        stage,
        generator_super_scope=None,
        # Options.
        check_shapes=True,
        apply_batch_norm=False):
    current_stage_generator_scope = 'Generator_stage_' + str(stage)
    current_stage_discriminator_scope = 'Discriminator_stage_' + str(stage)

    # Wrap generator in super scope.
    generator_super_scope = _get_or_create_gen_super_scope(
        generator_super_scope)
    with tf.variable_scope(generator_super_scope):
        with tf.name_scope(generator_super_scope.original_name_scope):
            with tf.variable_scope(
                    current_stage_generator_scope,
                    reuse=tf.AUTO_REUSE) as current_gen_scope:
                print(tf.get_variable_scope().name)
                # Nested scope, specific to this generator stage.
                is_init_stage, noise, conditioning = generator_input_fn()
                generator_inputs = _convert_tensor_or_l_or_d(
                    (noise, conditioning))
                generator_inputs = [is_init_stage] + generator_inputs
                kwargs = {'final_size': 2 ** (6 + stage),
                          'apply_batch_norm': apply_batch_norm}
                generated_data, generator_hidden_code = generator_fn(
                    generator_inputs, **kwargs)

    # Discriminate generated and real data.
    with tf.variable_scope(current_stage_discriminator_scope,
                           reuse=tf.AUTO_REUSE) as dis_scope:
        discriminator_gen_outputs, disc_gen_outputs_uncond = discriminator_fn(
            generated_data, disc_conditioning,
            apply_batch_norm=apply_batch_norm)
    with tf.variable_scope(dis_scope):
        with tf.name_scope(dis_scope.original_name_scope):
            real_data = tf.convert_to_tensor(real_data)
            discriminator_real_outputs, disc_real_outputs_uncond = discriminator_fn(
                real_data, disc_conditioning, apply_batch_norm=apply_batch_norm)

    if check_shapes:
        if not generated_data.shape.is_compatible_with(real_data.shape):
            raise ValueError(
                'Generator output shape (%s) must be the same shape as real '
                'data (%s).' % (generated_data.shape, real_data.shape))

    # Get model-specific variables.
    generator_variables = tf.trainable_variables(generator_super_scope.name)
    discriminator_variables = tf.trainable_variables(dis_scope.name)

    return tfstackgan.StackGANModel(
        generator_inputs,
        generated_data,
        generator_variables,
        generator_super_scope,
        generator_fn,
        real_data,
        discriminator_real_outputs,
        discriminator_gen_outputs,
        discriminator_variables,
        dis_scope,
        discriminator_fn,
        generator_hidden_code,
        stage,
        disc_real_outputs_uncond,
        disc_gen_outputs_uncond, )


def _tensor_pool_adjusted_model(model,
                                tensor_pool_fn):
    """Adjusts model using `tensor_pool_fn`.
    Args:
      :param model: A GANModel tuple.
      :type model: tfstackgan.StackGANModel
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
        return model  # type: tfstackgan.StackGANModel

    pooled_generated_data, pooled_generator_inputs = tensor_pool_fn(
        (model.generated_data, model.generator_inputs))

    if isinstance(model, tfgan.GANModel):
        with tf.variable_scope(model.discriminator_scope, reuse=True):
            dis_gen_outputs = model.discriminator_fn(pooled_generated_data,
                                                     pooled_generator_inputs)
        return model._replace(
            discriminator_gen_outputs=dis_gen_outputs)  # type: tfstackgan.StackGANModel
    elif isinstance(model, tfgan.ACGANModel):
        with tf.variable_scope(model.discriminator_scope, reuse=True):
            (dis_pooled_gen_outputs,
             dis_pooled_gen_classification_logits) = model.discriminator_fn(
                pooled_generated_data, pooled_generator_inputs)
        return model._replace(
            discriminator_gen_outputs=dis_pooled_gen_outputs,
            discriminator_gen_classification_logits=
            dis_pooled_gen_classification_logits)  # type: tfstackgan.StackGANModel
    elif isinstance(model, tfgan.InfoGANModel):
        with tf.variable_scope(model.discriminator_scope, reuse=True):
            (dis_pooled_gen_outputs,
             pooled_predicted_distributions) = model.discriminator_and_aux_fn(
                pooled_generated_data, pooled_generator_inputs)
        return model._replace(
            discriminator_gen_outputs=dis_pooled_gen_outputs,
            predicted_distributions=pooled_predicted_distributions)  # type: tfstackgan.StackGANModel
    else:
        raise ValueError(
            'Tensor pool does not support `model`: %s.' % type(model))


def dis_loss(
        model,
        discriminator_loss_fn=tfstackgan_losses.wasserstein_discriminator_loss,
        # Auxiliary losses.
        gradient_penalty_weight=None,
        gradient_penalty_epsilon=1e-10,
        mutual_information_penalty_weight=None,
        aux_cond_generator_weight=None,
        aux_cond_discriminator_weight=None,
        tensor_pool_fn=None,
        # Options.
        add_summaries=True,
        uncond_loss_coeff=1.0):
    """Returns losses necessary to train discriminator.
    Args:
      :param model: A GANModels tuple containing models for each stage.
      :type model: tfstackgan.StackGANModel
      discriminator_loss_fn: The loss function on the discriminator. Takes a
        GANModel tuple.
      gradient_penalty_weight: If not `None`, must be a non-negative Python number
        or Tensor indicating how much to weight the gradient penalty. See
        https://arxiv.org/pdf/1704.00028.pdf for more details.
      gradient_penalty_epsilon: If `gradient_penalty_weight` is not None, the
        small positive value used by the gradient penalty function for numerical
        stability. Note some applications will need to increase this value to
        avoid NaNs.
      mutual_information_penalty_weight: If not `None`, must be a non-negative
        Python number or Tensor indicating how much to weight the mutual
        information penalty. See https://arxiv.org/abs/1606.03657 for more
        details.
      aux_cond_generator_weight: If not None: add a classification loss as in
        https://arxiv.org/abs/1610.09585
      aux_cond_discriminator_weight: If not None: add a classification loss as in
        https://arxiv.org/abs/1610.09585
      tensor_pool_fn: A function that takes (generated_data, generator_inputs),
        stores them in an internal pool and returns previous stored
        (generated_data, generator_inputs). For example
        `tf.gan.features.tensor_pool`. Defaults to None (not using tensor pool).
      add_summaries: Whether or not to add summaries for the losses.
    Returns:
      A DiscriminatorLoss 1-tuple of (discriminator_loss). Includes
      regularization losses.
    Raises:
      ValueError: If any of the auxiliary loss weights is provided and negative.
      ValueError: If `mutual_information_penalty_weight` is provided, but the
        `model` isn't an `InfoGANModel`.
    """
    # Validate arguments.
    gradient_penalty_weight = _validate_aux_loss_weight(gradient_penalty_weight,
                                                        'gradient_penalty_weight')
    mutual_information_penalty_weight = _validate_aux_loss_weight(
        mutual_information_penalty_weight, 'infogan_weight')
    aux_cond_generator_weight = _validate_aux_loss_weight(
        aux_cond_generator_weight, 'aux_cond_generator_weight')
    aux_cond_discriminator_weight = _validate_aux_loss_weight(
        aux_cond_discriminator_weight, 'aux_cond_discriminator_weight')

    # Verify configuration for mutual information penalty
    if (_use_aux_loss(mutual_information_penalty_weight) and
            not isinstance(model, tfgan.InfoGANModel)):
        raise ValueError(
            'When `mutual_information_penalty_weight` is provided, `model` must be '
            'an `InfoGANModel`. Instead, was %s.' % type(model))

    # Verify configuration for mutual auxiliary condition loss (ACGAN).
    if ((_use_aux_loss(aux_cond_generator_weight) or
             _use_aux_loss(aux_cond_discriminator_weight)) and
            not isinstance(model, tfgan.ACGANModel)):
        raise ValueError(
            'When `aux_cond_generator_weight` or `aux_cond_discriminator_weight` '
            'is provided, `model` must be an `ACGANModel`. Instead, was %s.' %
            type(model))

    # Create standard losses.
    pooled_model = _tensor_pool_adjusted_model(model, tensor_pool_fn)
    dis_loss = discriminator_loss_fn(
        pooled_model.discriminator_real_outputs,
        pooled_model.disc_real_outputs_uncond,
        pooled_model.discriminator_gen_outputs,
        pooled_model.disc_gen_outputs_uncond,
        uncond_loss_coeff,
        add_summaries=add_summaries)

    # Add optional extra losses.
    if _use_aux_loss(gradient_penalty_weight):
        gp_loss = tfgan_losses.wasserstein_gradient_penalty(
            model, epsilon=gradient_penalty_epsilon,
            add_summaries=add_summaries)
        dis_loss += gradient_penalty_weight * gp_loss
    if _use_aux_loss(mutual_information_penalty_weight):
        info_loss = tfgan_losses.mutual_information_penalty(
            model, add_summaries=add_summaries)
        dis_loss += mutual_information_penalty_weight * info_loss
    if _use_aux_loss(aux_cond_generator_weight):
        ac_gen_loss = tfgan_losses.acgan_generator_loss(
            model, add_summaries=add_summaries)
    if _use_aux_loss(aux_cond_discriminator_weight):
        ac_disc_loss = tfgan_losses.acgan_discriminator_loss(
            model, add_summaries=add_summaries)
        dis_loss += aux_cond_discriminator_weight * ac_disc_loss
    # Gathers auxiliary losses.
    if model.discriminator_scope:
        dis_reg_loss = tf.losses.get_regularization_loss(
            model.discriminator_scope.name)
    else:
        dis_reg_loss = 0

    return tfstackgan.DiscriminatorLoss(dis_loss + dis_reg_loss)


def gen_loss(
        models,
        generator_loss_fn=tfstackgan_losses.wasserstein_generator_loss,
        # Auxiliary losses.
        mutual_information_penalty_weight=None,
        aux_cond_generator_weight=None,
        # Options.
        add_summaries=True,
        color_loss_weight=0,
        uncond_loss_coeff=1.0,
        kl_loss_coeff=2.0,
        mu=None,
        logvar=None):
    """Returns losses necessary to train generator.
    Args:
      model: A GANModel tuple.
      generator_loss_fn: The loss function on the generator. Takes a
        GANModel tuple.
      mutual_information_penalty_weight: If not `None`, must be a non-negative
        Python number or Tensor indicating how much to weight the mutual
        information penalty. See https://arxiv.org/abs/1606.03657 for more
        details.
      aux_cond_generator_weight: If not None: add a classification loss as in
        https://arxiv.org/abs/1610.09585
      add_summaries: Whether or not to add summaries for the losses.
    Returns:
      A GeneratorLoss 1-tuple of (generator_loss). Includes
      regularization losses.
    Raises:
      ValueError: If any of the auxiliary loss weights is provided and negative.
      ValueError: If `mutual_information_penalty_weight` is provided, but the
        `model` isn't an `InfoGANModel`.
    """
    model = models[-1]
    # Validate arguments.
    mutual_information_penalty_weight = _validate_aux_loss_weight(
        mutual_information_penalty_weight, 'infogan_weight')
    aux_cond_generator_weight = _validate_aux_loss_weight(
        aux_cond_generator_weight, 'aux_cond_generator_weight')

    # Verify configuration for mutual information penalty
    if (_use_aux_loss(mutual_information_penalty_weight) and
            not isinstance(model, tfgan.InfoGANModel)):
        raise ValueError(
            'When `mutual_information_penalty_weight` is provided, `model` must be '
            'an `InfoGANModel`. Instead, was %s.' % type(model))

    # Verify configuration for mutual auxiliary condition loss (ACGAN).
    if _use_aux_loss(aux_cond_generator_weight and
                             not isinstance(model, tfgan.ACGANModel)):
        raise ValueError(
            'When `aux_cond_generator_weight` or `aux_cond_discriminator_weight` '
            'is provided, `model` must be an `ACGANModel`. Instead, was %s.' %
            type(model))

    ### TODO(joppe): Verify for StackGAN

    # Create standard losses.
    gen_loss = 0

    ### TODO(joppe): use _use_aux_loss helper
    if color_loss_weight > 0:
        # Compute color preserve losses
        color_loss_value = tfstackgan_losses.color_loss(color_loss_weight,
                                                        models)
    else:
        color_loss_value = 0
    gen_loss += color_loss_value

    for i in range(len(models)):
        with tf.name_scope('loss_stage_' + str(i)):
            gen_loss += generator_loss_fn(
                models[i].discriminator_gen_outputs_cond,
                models[i].discriminator_gen_outputs_uncond,
                uncond_loss_coeff,
                add_summaries=add_summaries)

            # Add optional extra losses.
            if _use_aux_loss(mutual_information_penalty_weight):
                info_loss = tfgan_losses.mutual_information_penalty(
                    models[i], add_summaries=add_summaries)
                gen_loss += mutual_information_penalty_weight * info_loss
            if _use_aux_loss(aux_cond_generator_weight):
                ac_gen_loss = tfgan_losses.acgan_generator_loss(
                    models[i], add_summaries=add_summaries)
                gen_loss += aux_cond_generator_weight * ac_gen_loss

    if mu and logvar:
        gen_loss += kl_loss_coeff * tfstackgan_losses.kl_loss(mu, logvar)

    # Gathers auxiliary losses.
    if model.generator_scope:
        gen_reg_loss = tf.losses.get_regularization_loss(model.generator_scope)
    else:
        gen_reg_loss = 0

    return tfstackgan.GeneratorLoss(gen_loss + gen_reg_loss)


def _get_dis_update_ops(kwargs, dis_scope, check_for_unused_ops=True):
    """Gets discriminator update ops.
    Args:
      kwargs: A dictionary of kwargs to be passed to `create_train_op`.
        `update_ops` is removed, if present.
      dis_scope: A scope for the discriminator.
      check_for_unused_ops: A Python bool. If `True`, throw Exception if there are
        unused update ops.
    Returns:
      discriminator update ops.
    Raises:
      ValueError: If there are update ops outside of the generator or
        discriminator scopes.
    """
    if 'update_ops' in kwargs:
        update_ops = set(kwargs['update_ops'])
        del kwargs['update_ops']
    else:
        update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    all_dis_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS, dis_scope))

    # if check_for_unused_ops:
    #  unused_ops = update_ops - all_gen_ops - all_dis_ops
    #  if unused_ops:
    #    raise ValueError('There are unused update ops: %s' % unused_ops)

    dis_update_ops = list(all_dis_ops & update_ops)

    return dis_update_ops


def _get_gen_update_ops(kwargs, gen_scope, check_for_unused_ops=True):
    """Gets generator update ops.
    Args:
      kwargs: A dictionary of kwargs to be passed to `create_train_op`.
        `update_ops` is removed, if present.
      gen_scope: A scope for the generator.
      check_for_unused_ops: A Python bool. If `True`, throw Exception if there are
        unused update ops.
    Returns:
      generator update ops
    Raises:
      ValueError: If there are update ops outside of the generator or
        discriminator scopes.
    """
    if 'update_ops' in kwargs:
        update_ops = set(kwargs['update_ops'])
        del kwargs['update_ops']
    else:
        update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    all_gen_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS, gen_scope))

    # if check_for_unused_ops:
    #  unused_ops = update_ops - all_gen_ops - all_dis_ops
    #  if unused_ops:
    #    raise ValueError('There are unused update ops: %s' % unused_ops)

    gen_update_ops = list(all_gen_ops & update_ops)

    return gen_update_ops


def generator_train_op(
        model,
        loss,
        optimizer,
        check_for_unused_update_ops=True,
        # Optional args to pass directly to the `create_train_op`.
        **kwargs):
    # return GeneratorTrainOps tuple with one gen train op in
    # generator_train_op field
    gen_update_ops = _get_gen_update_ops(
        kwargs, model.generator_scope.name,
        check_for_unused_update_ops)

    generator_global_step = None
    # if isinstance(generator_optimizer,
    #              sync_replicas_optimizer.SyncReplicasOptimizer):
    # TODO(joelshor): Figure out a way to get this work without including the
    # dummy global step in the checkpoint.
    # WARNING: Making this variable a local variable causes sync replicas to
    # hang forever.
    #  generator_global_step = variable_scope.get_variable(
    #      'dummy_global_step_generator',
    #      shape=[],
    #      dtype=global_step.dtype.base_dtype,
    #      initializer=init_ops.zeros_initializer(),
    #      trainable=False,
    #      collections=[ops.GraphKeys.GLOBAL_VARIABLES])
    #  gen_update_ops += [generator_global_step.assign(global_step)]
    with tf.name_scope('generator_train'):
        gen_train_op = training.create_train_op(
            total_loss=loss.generator_loss,
            optimizer=optimizer,
            variables_to_train=model.generator_variables,
            global_step=generator_global_step,
            update_ops=gen_update_ops,
            **kwargs)

    return tfstackgan.GeneratorTrainOp(gen_train_op)


def discriminator_train_ops(
        models,
        losses,
        optimizer,
        check_for_unused_update_ops=True,
        # Optional args to pass directly to the `create_train_op`.
        **kwargs):
    # return DiscriminatorTrainOps tuple with one train op per discriminator
    # in the discriminator_train_op field
    stack_depth = len(models)
    dis_update_ops = []
    for i in range(stack_depth):
        current_dis_update_ops = _get_dis_update_ops(
            kwargs, models[i].discriminator_scope.name,
            check_for_unused_update_ops)
        dis_update_ops.append(current_dis_update_ops)

    discriminator_global_step = None
    # if isinstance(discriminator_optimizer,
    #              sync_replicas_optimizer.SyncReplicasOptimizer):
    # See comment above `generator_global_step`.
    #  discriminator_global_step = variable_scope.get_variable(
    #      'dummy_global_step_discriminator',
    #      shape=[],
    #      dtype=global_step.dtype.base_dtype,
    #      initializer=init_ops.zeros_initializer(),
    #      trainable=False,
    #      collections=[ops.GraphKeys.GLOBAL_VARIABLES])
    #  dis_update_ops += [discriminator_global_step.assign(global_step)]
    disc_train_ops = []
    for i in range(stack_depth):
        with tf.name_scope('discriminator_train'):
            current_disc_train_op = training.create_train_op(
                total_loss=losses[i].discriminator_loss,
                optimizer=optimizer,
                variables_to_train=models[i].discriminator_variables,
                global_step=discriminator_global_step,
                update_ops=dis_update_ops[i],
                **kwargs)
        disc_train_ops.append(current_disc_train_op)

    return tfstackgan.DiscriminatorTrainOps(disc_train_ops)


def get_sequential_train_hooks(train_steps=tfgan.GANTrainSteps(1, 1)):
    """Returns a hooks function for sequential GAN training.
    Args:
      train_steps: A `GANTrainSteps` tuple that determines how many generator
        and discriminator training steps to take.
    Returns:
      A function that takes a GANTrainOps tuple and returns a list of hooks.
    """

    def get_hooks(train_ops):
        # train_ops: GANTrainOps TUPLE WITH ONE GEN TRAIN OP + LIST OF DIS TRAIN OPS
        hooks = []
        for train_op in train_ops.discriminator_train_op:
            current_discriminator_hook = RunTrainOpsHook(
                train_op,
                train_steps.discriminator_train_steps)
            hooks.append(current_discriminator_hook)

        generator_hook = RunTrainOpsHook(train_ops.generator_train_op,
                                         train_steps.generator_train_steps)
        hooks.append(generator_hook)

        return hooks

    return get_hooks
