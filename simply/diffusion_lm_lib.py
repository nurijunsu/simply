# Copyright 2024 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion LM helpers (masking, loss, and training loop)."""

from collections.abc import Callable, Mapping, MutableMapping
import functools
import time
from typing import Any, ClassVar
import warnings

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from simply import data_lib
from simply import model_lib
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import experiment_helper as exp_helper
from simply.utils import optimizers as opt_lib
from simply.utils import registry
from simply.utils import sharding as sharding_lib

Batch = MutableMapping[str, np.ndarray | jnp.ndarray]
CollectExtraLossFn = Callable[[Any], tuple[jnp.ndarray, dict[str, Any]]]


class AlphaSchedulerRegistry(registry.RootRegistry):
  """Registry for alpha schedulers used in diffusion language models."""
  namespace: ClassVar[str] = 'Scheduler'


@AlphaSchedulerRegistry.register
class BaseAlphaScheduler:
  """Base class for alpha schedulers in diffusion language models."""

  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
    return self.alpha(t)

  def alpha(self, t: jnp.ndarray) -> jnp.ndarray:
    """Compute alpha(t) for a batch of timesteps t in [0, 1]."""
    raise NotImplementedError

  def alpha_derivative(self, t: jnp.ndarray) -> jnp.ndarray:
    """Compute d(alpha)/dt for a batch of timesteps t in [0, 1]."""
    raise NotImplementedError

  def reverse_mask_probability(
      self, s: jnp.ndarray, t: jnp.ndarray
  ) -> jnp.ndarray:
    """Compute reverse mask probability from timestep t to s."""
    return (1 - self(s)) / (1 - self(t))


@AlphaSchedulerRegistry.register
class LinearAlphaScheduler(BaseAlphaScheduler):
  """Linear alpha scheduler: alpha(t) = 1 - t."""

  def alpha(self, t: jnp.ndarray) -> jnp.ndarray:
    return 1.0 - t

  def alpha_derivative(self, t: jnp.ndarray) -> jnp.ndarray:
    return -jnp.ones_like(t)


@AlphaSchedulerRegistry.register
class CosineAlphaScheduler(BaseAlphaScheduler):
  """Cosine alpha scheduler: alpha(t) = 1 - cos(pi(1-t)/2)."""

  def alpha(self, t: jnp.ndarray) -> jnp.ndarray:
    return 1 - jnp.cos((jnp.pi / 2) * (1 - t))

  def alpha_derivative(self, t: jnp.ndarray) -> jnp.ndarray:
    return -(jnp.pi / 2) * jnp.sin((jnp.pi / 2) * (1 - t))


def apply_diffusion_mask_to_batch(
    batch: Batch,
    rng_key: jax.Array,
    *,
    mask_token_id: int,
    time_epsilon: float = 1e-5,
    scheduler_name: str = 'LinearAlphaScheduler',
) -> Batch:
  """Masks decoder_input_tokens and updates loss weights for diffusion LM."""
  inputs = batch['decoder_input_tokens']
  key_t, key_mask = jax.random.split(rng_key)

  scheduler = AlphaSchedulerRegistry.get_instance(scheduler_name)
  t = time_epsilon + (1 - time_epsilon) * jax.random.uniform(
      key_t, (inputs.shape[0],), dtype=jnp.float32
  )
  mask_ratio = (1.0 - scheduler(t))[:, None]

  loss_weights = batch.get('decoder_loss_weights', None)
  if loss_weights is None:
    loss_weights = jnp.ones_like(inputs, dtype=jnp.float32)
  else:
    loss_weights = loss_weights.astype(jnp.float32)

  mask_candidates = loss_weights > 0
  mask = (jax.random.uniform(key_mask, inputs.shape) < mask_ratio) & (
      mask_candidates
  )
  masked_inputs = jnp.where(
      mask, jnp.array(mask_token_id, dtype=inputs.dtype), inputs
  )
  masked_loss_weights = loss_weights * mask.astype(loss_weights.dtype)

  new_batch = dict(batch)
  new_batch['decoder_input_tokens'] = masked_inputs
  new_batch['decoder_loss_weights'] = masked_loss_weights
  # Keep original tokens for non-right-shifted diffusion targets.
  new_batch['decoder_diffusion_targets'] = inputs
  return new_batch


def compute_diffusion_loss(
    model,
    params,
    batch: Batch,
    *,
    right_shift: bool = False,
    add_extra_loss: bool = True,
) -> tuple[jnp.ndarray, dict[str, Any]]:
  inputs = batch['decoder_input_tokens']
  if right_shift:
    targets = batch['decoder_target_tokens']
  else:
    targets = batch.get('decoder_diffusion_targets', None)
    if targets is None:
      raise ValueError(
          'decoder_diffusion_targets is required when right_shift is False.'
      )
  loss_weights = batch.get('decoder_loss_weights', None)
  if loss_weights is None:
    loss_weights = jnp.ones_like(targets, dtype=jnp.float32)
  else:
    loss_weights = loss_weights.astype(jnp.float32)

  segment_ids = batch.get('decoder_segment_ids', None)
  segment_positions = batch.get('decoder_positions', None)
  logits, model_extra_output = model.apply(
      params,
      inputs,
      segment_ids=segment_ids,
      segment_positions=segment_positions,
      extra_inputs=batch.get('extra_inputs', None),
  )
  logits = logits.astype(jnp.float32)
  targets_one_hot = jax.nn.one_hot(targets, logits.shape[-1], axis=-1)
  token_loss = -jnp.einsum(
      'blv,blv->bl', targets_one_hot, jax.nn.log_softmax(logits)
  )
  total_loss = jnp.sum(token_loss * loss_weights)
  total_loss_weight = sharding_lib.with_sharding_constraint(
      jnp.sum(loss_weights), None
  )
  loss = jnp.where(
      total_loss_weight > 0,
      total_loss / total_loss_weight,
      jnp.array(0.0, dtype=total_loss.dtype),
  )
  loss = sharding_lib.with_sharding_constraint(loss, None)
  # Compute accuracy on masked positions.
  pred = jnp.argmax(logits, axis=-1)
  correct = (pred == targets).astype(jnp.float32) * loss_weights
  accuracy = jnp.where(
      total_loss_weight > 0,
      jnp.sum(correct) / total_loss_weight,
      jnp.array(0.0, dtype=jnp.float32),
  )
  accuracy = sharding_lib.with_sharding_constraint(accuracy, None)

  extra_output = {'accuracy': accuracy, 'loss_weight': total_loss_weight}
  extra_loss, extra_metric_dict = model_lib.collect_loss_and_metric(model_extra_output)
  extra_output.update(extra_metric_dict)
  if add_extra_loss:
    loss += extra_loss
  return loss, extra_output


def compute_diffusion_train_loss(
    model,
    params,
    batch: Batch,
    *,
    right_shift: bool = False,
) -> tuple[jnp.ndarray, dict[str, Any]]:
  return compute_diffusion_loss(
      model,
      params,
      batch,
      right_shift=right_shift,
      add_extra_loss=True,
  )


def compute_diffusion_eval_loss(
    model,
    params,
    batch: Batch,
    *,
    right_shift: bool = False,
) -> tuple[jnp.ndarray, dict[str, Any]]:
  return compute_diffusion_loss(
      model,
      params,
      batch,
      right_shift=right_shift,
      add_extra_loss=False,
  )


@functools.partial(model_lib.TrainLoopRegistry.register, name='diffusion_lm')
def run_experiment(
    config,
    # Leave `experiment_dir` as empty string to skip saving experiment data.
    # Useful if no need to save any data and can reduce some overhead.
    experiment_dir='',
    # All the args below are deprecated.
    mesh_shape=None,
    dcn_mesh_shape=None,
    decoding_mesh_shape=None,
    sharding_config=None,
    create_dataset=None,
):
  if create_dataset is not None:
    warnings.warn('create_dataset is deprecated.')
  if mesh_shape is not None:
    warnings.warn('mesh_shape is deprecated.')
  if dcn_mesh_shape is not None:
    warnings.warn('dcn_mesh_shape is deprecated.')
  if decoding_mesh_shape is not None:
    warnings.warn('decoding_mesh_shape is deprecated.')
  if sharding_config is not None:
    warnings.warn('sharding_config is deprecated.')
  del (
      create_dataset, decoding_mesh_shape, dcn_mesh_shape,
      sharding_config, mesh_shape)
  logging.info('jax.process_index(): %s', jax.process_index())
  # Setup model, optimizer, initial state, and mesh.
  sharding_lib.set_default_mesh_shape(
      mesh_shape=config.mesh_shape, dcn_mesh_shape=config.dcn_mesh_shape,
      axis_names=config.sharding_config.mesh_axis_names,
  )
  helper = exp_helper.ExperimentHelper(
      experiment_dir,
      ckpt_interval=config.ckpt_interval,
      ckpt_max_to_keep=config.ckpt_max_to_keep,
      ckpt_keep_period=config.ckpt_keep_period,
      num_train_steps=config.num_train_steps,
      metric_log_interval=config.tb_log_interval,
      log_additional_info=config.log_additional_info,
      should_save_ckpt=config.should_save_ckpt,
  )
  model, _ = model_lib.create_model(config, config.sharding_config)
  helper.save_config_info(config, config.sharding_config, model)
  opt = config.optimizer
  state = model_lib.get_init_state(
      config, config.sharding_config, helper.ckpt_mngr, helper.ckpt_dir)
  helper.save_state_info(state)

  diffusion_mask_key = jax.random.key(config.diffusion_seed)
  diffusion_process_index = jax.process_index()
  diffusion_loss_fn = functools.partial(
      compute_diffusion_train_loss,
      right_shift=config.diffusion_right_shift
  )
  scheduler_name=config.diffusion_alpha_scheduler or 'LinearAlphaScheduler'

  # Compile loss, train and learning rate functions.
  @functools.partial(
      jax.jit, donate_argnames=['state'], static_argnames=['add_log_info']
  )
  def train_one_step_fn(state, batch, lr, add_log_info=False):
    step = jax.lax.convert_element_type(state['steps'], jnp.uint32)
    mask_key = jax.random.fold_in(diffusion_mask_key, step)
    mask_key = jax.random.fold_in(mask_key, diffusion_process_index)
    batch = apply_diffusion_mask_to_batch(
        batch,
        mask_key,
        mask_token_id=config.diffusion_mask_token_id,
        time_epsilon=config.diffusion_time_epsilon,
        scheduler_name=scheduler_name,
    )
    return model_lib.train_one_step(
        state=state,
        batch=batch,
        lr=lr,
        model=model,
        opt=opt,
        grad_accum_steps=config.grad_accum_steps,
        clip_grad_norm=config.clip_grad_norm,
        clip_update_norm=config.clip_update_norm,
        clip_local_update_rms=config.clip_local_update_rms,
        weight_decay=config.weight_decay,
        custom_loss_fn=diffusion_loss_fn,
        add_log_info=add_log_info,
    )

  lr_fn = common.named_jit(opt_lib.create_lr_schedule(config), 'lr_fn')

  # Prepare datasets.
  logging.info('Initializing dataset.')
  train_set = data_lib.create_iter_dataset(config, training=True)
  logging.info('sharding_config.data_partition: %s',
               config.sharding_config.data_partition)

  train_iter = iter(train_set)

  train_iter_state = None
  if helper.ckpt_mngr and helper.ckpt_mngr.latest_step() is not None:
    data_state = ckpt_lib.load_data_state_from_dir(
        helper.ckpt_dir, helper.ckpt_mngr.latest_step()
    )
    assert isinstance(data_state, Mapping)
    train_iter_state = data_state.get('train_iter_state', None)
  if train_iter_state is not None:
    train_iter.set_state(train_iter_state)

  # Start training.
  prev_step_timestamp = time.time()
  final_result = {}
  steps = int(state['steps'].addressable_data(0))

  # Create eval_fn for validation set.
  if config.use_validation_set:
    loss_fn = common.named_jit(
        functools.partial(
            compute_diffusion_eval_loss,
            right_shift=config.diffusion_right_shift,
        ),
        'validation_loss_fn',
        model=model,
    )

    def _eval_batch_transform(batch: Batch, step: int) -> Batch:
      step = jnp.asarray(step, dtype=jnp.uint32)
      mask_key = jax.random.fold_in(diffusion_mask_key, step)
      mask_key = jax.random.fold_in(mask_key, diffusion_process_index)
      return apply_diffusion_mask_to_batch(
          batch,
          mask_key,
          mask_token_id=config.diffusion_mask_token_id,
          time_epsilon=config.diffusion_time_epsilon,
          scheduler_name=scheduler_name,
      )

    validation_set = data_lib.create_iter_dataset(
        config, training=False
    )
    eval_fn = functools.partial(
        model_lib.run_eval,
        eval_set=validation_set,
        num_eval_steps=config.validation_num_eval_steps,
        loss_fn=loss_fn,
        batch_transform_fn=_eval_batch_transform,
    )
  else:
    eval_fn = None

  agg_metrics = {}
  eval_result = {}
  should_early_stop = False
  while steps <= config.num_train_steps and not should_early_stop:
    with jax.profiler.StepTraceAnnotation('train', step_num=steps):
      logging.info('steps: %s', steps)
      helper.save_ckpt(state, steps, data=train_iter.get_state())
      # Run eval every validation_eval_interval steps and at the very end.
      if config.use_validation_set and (
          steps % config.validation_eval_interval == 0
          or steps == config.num_train_steps
      ):
        eval_result = eval_fn(state=state)
        helper.write_scalars(steps, eval_result)
        helper.flush()

      t1 = time.time()
      batch = next(train_iter)
      logging.info('batch=%s', batch)

      batch = model_lib.build_global_array_from_replicated(
          batch, data_partition=(('replica', 'data'),)
      )
      data_generation_step_time = time.time() - t1

      t1 = time.time()
      lr = lr_fn(state['steps'])
      loss, state, log_dict = train_one_step_fn(
          state=state,
          batch=batch,
          lr=lr,
          add_log_info=helper.should_log_additional_info(steps),
      )
      train_loss = float(loss.addressable_data(0))
      train_step_time = time.time() - t1
      logging.info('train_loss: %s', train_loss)

      if helper.should_log_additional_info(steps):
        # Log batch stats info for debugging purpose.
        batch_stats_info = model_lib.compute_batch_stats_info(batch)
        logging.info('========== batch_stats_info ==========')
        for k, v in batch_stats_info.items():
          logging.info('%s: %s', k, v)
        log_dict.update(batch_stats_info)

      step_time = time.time() - prev_step_timestamp
      prev_step_timestamp = time.time()

      # Track and log all the metrics.
      if helper.should_log_additional_info(steps):
        helper.add_metric('total_step_time_with_additional_info', step_time)
        helper.add_metric(
            'train_step_time_with_additional_info', train_step_time)
      else:
        helper.add_metric('total_step_time', step_time)
        helper.add_metric('train_step_time', train_step_time)
      helper.add_metric('avg_total_step_time', step_time)
      logging.info('%s secs per step, log_additional_info: %s',
                   step_time, helper.should_log_additional_info(steps))
      helper.add_metric('loss', train_loss)
      helper.add_metric(
          'accuracy', float(log_dict['accuracy'].addressable_data(0)))
      helper.add_metric(
          'data_generation_step_time', data_generation_step_time)

      agg_metrics = helper.get_aggregated_metrics()
      should_early_stop = should_early_stop or (
          config.early_stop and
          config.early_stop.should_stop(
              steps, agg_metrics))
      if helper.should_log_metrics(steps):
        t1 = time.time()
        metrics_dict = dict(
            lr=lr,
            secs_per_step=agg_metrics['avg_total_step_time'],
            steps_per_sec=1 / agg_metrics['avg_total_step_time'],
        )
        metrics_dict.update(agg_metrics)
        metrics_dict.update(model_lib.flatten_dict(log_dict))
        helper.write_scalars(steps, metrics_dict)
        helper.flush()
        event_write_time = time.time() - t1
        logging.info('%s secs per writing metrics.', event_write_time)
      steps += 1
  final_result['steps'] = steps - 1
  final_result['train_loss'] = float(agg_metrics['loss'])
  final_result['train_accuracy'] = float(agg_metrics['accuracy'])
  if eval_result:
    final_result['validation_loss'] = float(eval_result['eval_loss'])
    final_result['validation_accuracy'] = float(
        eval_result['eval_accuracy'])
  final_result['early_stop'] = should_early_stop
  if should_early_stop:  # pylint: disable=multiple-statements
    logging.info('Training is early stopped!')
  helper.close(final_result)
  return final_result
