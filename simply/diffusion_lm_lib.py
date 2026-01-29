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

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import dataclasses
import functools
import math
import time
from typing import Any, ClassVar
import warnings

from absl import logging
import einops
import jax
import jax.numpy as jnp
import numpy as np
import seqio
from seqio import vocabularies as seqio_vocabularies
from sentencepiece import sentencepiece_model_pb2
import sentencepiece as sentencepiece_processor

from simply import data_lib
from simply import model_lib
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import experiment_helper as exp_helper
from simply.utils import module
from simply.utils import optimizers as opt_lib
from simply.utils import registry
from simply.utils import sampling_lib
from simply.utils import sharding as sharding_lib
from simply.utils import tokenization

Batch = MutableMapping[str, np.ndarray | jnp.ndarray]
PRNGKey = jax.typing.ArrayLike
PyTree = common.PyTree
Array = common.Array
SamplingParams = sampling_lib.SamplingParams
SamplingState = model_lib.SamplingState
SamplingOutput = model_lib.SamplingOutput
compute_log_likelihood = model_lib.compute_log_likelihood
pad_along_axis = model_lib.pad_along_axis
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


def _ensure_sentencepiece_mask_token(
    vocab: seqio_vocabularies.SentencePieceVocabulary,
    mask_token: str,
) -> int:
  model_context = vocab._model_context()
  sp_model = model_context.sp_model
  model = sentencepiece_model_pb2.ModelProto.FromString(sp_model)
  pieces = [piece.piece for piece in model.pieces]
  word_boundary = '\u2581'
  for candidate in (mask_token, f'{word_boundary}{mask_token}'):
    if candidate in pieces:
      return pieces.index(candidate)
  model.pieces.add(
      piece=f'{word_boundary}{mask_token}',
      score=0.0,
      type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
  )
  sp_model = model.SerializeToString()
  tokenizer = sentencepiece_processor.SentencePieceProcessor()
  tokenizer.LoadFromSerializedProto(sp_model)
  vocab._model = type(model_context)(tokenizer=tokenizer, sp_model=sp_model)
  return len(model.pieces) - 1


def _resolve_vocab_from_dataset_name(dataset_name: str):
  if not dataset_name:
    return None
  if dataset_name.startswith('simply_json:'):
    return None
  task_name = dataset_name
  if task_name.startswith('simply_det:'):
    task_name = task_name.removeprefix('simply_det:')
  for candidate in (task_name, dataset_name):
    try:
      task_or_mixture = seqio.get_mixture_or_task(candidate)
      break
    except ValueError:
      task_or_mixture = None
  if task_or_mixture is None:
    return None
  output_features = task_or_mixture.output_features
  if not output_features:
    return None
  if 'targets' in output_features:
    return output_features['targets'].vocabulary
  return next(iter(output_features.values())).vocabulary


def _resolve_mask_token_id(config) -> int:
  if config.diffusion_mask_token_id >= 0:
    return config.diffusion_mask_token_id
  vocab = None
  if config.vocab_name:
    vocab = tokenization.TokenizerRegistry.get_instance(config.vocab_name)
  else:
    vocab = _resolve_vocab_from_dataset_name(
        getattr(config, 'dataset_name', '')
    )
  if vocab is None:
    raise ValueError(
        'vocab_name must be set (or dataset_name must be a SeqIO task) to '
        'resolve <|MASK|>.'
    )
  mask_token = '<|MASK|>'
  if isinstance(vocab, tokenization.HuggingFaceVocab):
    tokenizer = vocab.tokenizer
    mask_id = tokenizer.token_to_id(mask_token)
    if mask_id is None:
      tokenizer.add_special_tokens([mask_token])
      mask_id = tokenizer.token_to_id(mask_token)
    if mask_id is None:
      raise ValueError(f'Unable to add {mask_token} to tokenizer.')
    if mask_id >= config.vocab_size:
      raise ValueError(
          f'{mask_token} id {mask_id} exceeds vocab_size={config.vocab_size}.'
      )
    return int(mask_id)
  if isinstance(vocab, seqio_vocabularies.SentencePieceVocabulary):
    mask_id = _ensure_sentencepiece_mask_token(vocab, mask_token)
    if mask_id >= config.vocab_size:
      raise ValueError(
          f'{mask_token} id {mask_id} exceeds vocab_size={config.vocab_size}.'
      )
    return int(mask_id)
  if hasattr(vocab, 'encode') and hasattr(vocab, 'decode'):
    token_ids = vocab.encode(mask_token)
    if not token_ids or len(token_ids) != 1:
      raise ValueError(
          f'{mask_token} is not a single token in vocab {config.vocab_name}.'
      )
    if vocab.decode(token_ids) != mask_token:
      raise ValueError(
          f'{mask_token} does not round-trip in vocab {config.vocab_name}.'
      )
    return int(token_ids[0])
  raise ValueError(
      f'Unsupported vocab type for {config.vocab_name} when resolving {mask_token}.'
  )


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

  mask_token_id = _resolve_mask_token_id(config)
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
        mask_token_id=mask_token_id,
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
          mask_token_id=mask_token_id,
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
  

@dataclasses.dataclass(frozen=True)
class DiffusionSamplingParams(sampling_lib.SamplingParams):
  """Sampling parameters for diffusion decoding."""

  chunk_size: int = 128
  steps: int = 128
  remasking: str = 'low_confidence'
  stochastic_transfer: bool = False
  right_shift_logits: bool = False
  mask_token_id: int | None = None
  scheduler_name: str = 'LinearAlphaScheduler'

  def get_decoding_schedule(
      self, min_input_length: int, max_input_length: int
      ) -> sampling_lib.DecodingSchedule:
    min_input_length = int(min_input_length)
    prefill_size = (
        (min_input_length + self.chunk_size - 1) // self.chunk_size
    ) * self.chunk_size
    begin_position = min_input_length - 1

    end_position_exclusive = min(
        self.max_seq_len - 1,
        # Ensure int to avoid overflow.
        int(max_input_length) + self.max_decode_steps - 1,
    )
    prefill_size = min(prefill_size, end_position_exclusive + 1)

    return sampling_lib.DecodingSchedule(
        prefill_size=prefill_size,
        begin_position=begin_position,
        end_position=end_position_exclusive,
        chunk_size=self.chunk_size,
    )


def _reverse_transfer_probabilities(
    steps: int, scheduler_name: str
) -> jnp.ndarray:
  step_ids = jnp.arange(steps, dtype=jnp.float32)
  s = (steps - 1 - step_ids) / steps
  t = (steps - step_ids) / steps
  if scheduler_name == 'LinearAlphaScheduler':
    alpha_s = 1.0 - s
    alpha_t = 1.0 - t
  elif scheduler_name == 'CosineAlphaScheduler':
    alpha_s = 1 - jnp.cos((jnp.pi / 2) * (1 - s))
    alpha_t = 1 - jnp.cos((jnp.pi / 2) * (1 - t))
  else:
    raise ValueError(f'Unknown scheduler: {scheduler_name}')
  reverse_mask_prob = (1 - alpha_s) / (1 - alpha_t)
  return 1 - reverse_mask_prob

# TODO(kimjunsu) continue decode doesnt decode all masked tokens. Identify why. 
def continue_decode(
    apply_fn: Callable[..., Array],
    params: PyTree,
    init_sampling_state: model_lib.SamplingState,
    extra_inputs: Mapping[str, PyTree] | None = None,
    temperature: float = 1.0,
    *,
    mask_token_id: int,
    pad_id: int,
    chunk_size: int,
    steps: int,
    remasking: str = 'low_confidence',
    stochastic_transfer: bool = False,
    right_shift_logits: bool = False,
    scheduler_name: str = 'LinearAlphaScheduler',
) -> model_lib.SamplingState:
  tokens = init_sampling_state.tokens
  seq_len = tokens.shape[1]
  topk_size = min(chunk_size, seq_len)
  prng_key = init_sampling_state.prng_key

  mask_index = tokens == mask_token_id
  reverse_probs = _reverse_transfer_probabilities(steps, scheduler_name)

  def compute_num_transfer_tokens(
      rng_key: jax.Array, mask_index: Array
  ) -> tuple[jax.Array, Array]:
    mask_num = jnp.sum(mask_index, axis=1).astype(jnp.int32)
    if stochastic_transfer:
      keys = jax.random.split(rng_key, steps + 1)
      rng_key = keys[0]
      step_keys = keys[1:]

      def scan_fn(mask_num, inputs):
        rev_prob, step_key = inputs
        num = jax.random.binomial(step_key, mask_num, rev_prob)
        num = num.astype(jnp.int32)
        num = jnp.minimum(num, mask_num)
        mask_num = mask_num - num
        return mask_num, num

      _, nums = jax.lax.scan(
          scan_fn, mask_num, (reverse_probs, step_keys)
      )
    else:
      def scan_fn(mask_num, rev_prob):
        num = jnp.round(mask_num * rev_prob).astype(jnp.int32)
        num = jnp.minimum(num, mask_num)
        mask_num = mask_num - num
        return mask_num, num

      _, nums = jax.lax.scan(scan_fn, mask_num, reverse_probs)
    num_transfer_tokens = jnp.transpose(nums, (1, 0))
    return rng_key, num_transfer_tokens

  prng_key, num_transfer_tokens = compute_num_transfer_tokens(
      prng_key, mask_index
  )

  pos_idx = jnp.arange(seq_len)[None, :]
  segment_positions = jnp.broadcast_to(pos_idx, tokens.shape)

  def scatter_indices(idx: Array, take: Array) -> Array:
    mask = jnp.zeros((seq_len,), dtype=jnp.bool_)
    return mask.at[idx].set(take)

  def step_fn(carry, num_transfer):
    prng_key, tokens = carry
    prng_key, gumbel_key, rand_key = jax.random.split(prng_key, 3)
    logits, _ = apply_fn(
        params,
        tokens,
        segment_positions=segment_positions,
        extra_inputs=extra_inputs,
    )
    if right_shift_logits:
      logits = jnp.concatenate([logits[:, :1], logits[:, :-1]], axis=1)

    logits_with_noise = sampling_lib.add_gumbel_noise(
        logits, temperature=temperature, rng=gumbel_key
    )
    x0 = jnp.argmax(logits_with_noise, axis=-1)

    if remasking == 'low_confidence':
      probs = jax.nn.softmax(logits, axis=-1)
      x0_p = jnp.take_along_axis(probs, x0[..., None], axis=-1).squeeze(-1)
    elif remasking == 'random':
      x0_p = jax.random.uniform(rand_key, x0.shape)
    else:
      raise ValueError(f'Unknown remasking: {remasking}')

    mask_index = tokens == mask_token_id
    x0 = jnp.where(mask_index, x0, tokens)
    confidence = jnp.where(mask_index, x0_p, jnp.full_like(x0_p, -jnp.inf))

    _, topk_idx = jax.lax.top_k(confidence, k=topk_size)
    k = num_transfer
    take = jnp.arange(topk_size)[None, :] < k[:, None]
    transfer_index = jax.vmap(scatter_indices)(topk_idx, take)
    tokens = jnp.where(transfer_index, x0, tokens)
    return (prng_key, tokens), None

  num_transfer_tokens = num_transfer_tokens[:, :steps]
  num_transfer_tokens = jnp.transpose(num_transfer_tokens, (1, 0))
  (prng_key, tokens), _ = jax.lax.scan(
      step_fn, (prng_key, tokens), num_transfer_tokens
  )
  pad_mask = tokens == pad_id
  rev_pad = jnp.flip(pad_mask, axis=1)
  has_nonpad = jnp.any(~rev_pad, axis=1)
  first_nonpad = jnp.argmax(~rev_pad, axis=1)
  trailing_pad = jnp.where(has_nonpad, first_nonpad, seq_len)
  last_nonpad = jnp.where(has_nonpad, seq_len - trailing_pad - 1, 0)
  return dataclasses.replace(
      init_sampling_state,
      prng_key=prng_key,
      tokens=tokens,
      position=jnp.min(last_nonpad),
      token_logprobs=init_sampling_state.token_logprobs,
      token_scores=init_sampling_state.token_scores,
  )

@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class DiffusionSamplingState(model_lib.SamplingState):
  def mask_and_pad_to(
      self,
      length: int,
      mask_block_size: int,
      *,
      mask_id: int,
      pad_id: int = 0,
      mask_all: bool = False,
  ) -> 'DiffusionSamplingState':
    """Pads to `length` and masks a trailing pad block for diffusion decoding.

    If `mask_all` is False, pads tokens/logprobs/scores to `length`, then masks
    up to `mask_block_size` trailing pad positions (if at least that many pads
    exist). If `mask_all` is True, does not pad and instead masks all existing
    pad positions.

    Args:
      length: Target sequence length to pad to when `mask_all` is False.
      mask_block_size: Maximum number of trailing pad tokens to mask.
      mask_id: Token id to use for masked positions.
      pad_id: Token id used for padding in the sequence.
      mask_all: Whether to mask all pad positions without padding.

    Returns:
      A new `DiffusionSamplingState` with updated tokens/logprobs/scores.
    """
    tokens = self.tokens
    token_logprobs = self.token_logprobs
    token_scores = self.token_scores 

    if not mask_all:
      tokens = model_lib.pad_to_along_axis(tokens, length, axis=1)
      token_logprobs = model_lib.pad_to_along_axis(
          token_logprobs, length, axis=1
      )
      token_scores = model_lib.pad_to_along_axis(
          token_scores, length, axis=1
      )

    seq_len = tokens.shape[1]
    pad_mask = tokens == pad_id
    if mask_all:
      mask_indices = pad_mask
    else:
      rev_pad = jnp.flip(pad_mask, axis=1)
      has_nonpad = jnp.any(~rev_pad, axis=1)
      first_nonpad = jnp.argmax(~rev_pad, axis=1)
      trailing_pad = jnp.where(has_nonpad, first_nonpad, seq_len)
      mask_len = jnp.where(
          trailing_pad > mask_block_size, mask_block_size, 0
      )
      mask_start = seq_len - trailing_pad
      pos_idx = jnp.arange(seq_len)[None, :]
      mask_indices = (
          (pos_idx >= mask_start[:, None])
          & (pos_idx < (mask_start + mask_len)[:, None])
          & pad_mask
      )

    tokens = jnp.where(mask_indices, mask_id, tokens)
    return dataclasses.replace(
        self,
        tokens=tokens,
        token_logprobs=token_logprobs,
        token_scores=token_scores,
    )


class DLMInterface:

  def __init__(
      self,
      model: module.SimplyModule,
      params: PyTree,
      vocab: tokenization.SimplyVocab[str] | None = None,
      input_processor: sampling_lib.InputProcessorInterface | None = None,
      default_sampling_params: DiffusionSamplingParams | None = None,
      bos_id: int | None = None,
      pad_id: int | None = None,
      mask_token_id: int | None = None,
      extra_eos_ids: Sequence[int] | None = None,
      extra_eos_tokens: Sequence[str] | None = None,
  ) -> None:
    """An interface to interact with a language model.

    Args:
      model: The model to use, for example, a TransformerLM instance.
      params: The `params` to use in `model.apply`.
      vocab: The vocabulary instance to use. Either `vocab` or `input_processor`
        should be specified. If `vocab` is specified, it will be used to a
        instantiate a default input processor for basic text inputs.
      input_processor: The input processor to use, for specialized input
        processing. For basic text inputs, it is enough to specify `vocab`.
      default_sampling_params: Default sampling params for `generate`.
      bos_id: The bos id to use, if not given then it will use the `bos_id`
        field of the `vocab`.
      pad_id: The pad id to use, if not given then it will use the `pad_id`
        field of the `vocab`.
      mask_token_id: The mask token id to use for diffusion sampling.
      extra_eos_ids: Extra eos ids to include.
      extra_eos_tokens: Extra eos tokens to include.
    """
    self.model = model
    if input_processor:
      self.input_processor = input_processor
    else:
      assert vocab is not None, 'Must provide one of vocab or input_processor!'
      self.input_processor = sampling_lib.BasicTextInputProcessor(
          vocab,
          bos_id_override=bos_id,
          pad_id_override=pad_id,
          extra_eos_ids=extra_eos_ids,
          extra_eos_tokens=extra_eos_tokens,
      )
    self.mask_token_id = mask_token_id
    self.default_sampling_params = default_sampling_params or DiffusionSamplingParams()

    self.decode_fn = jax.jit(
        common.named_partial_fn(
            continue_decode,
            'decode_fn',
            apply_fn=model.apply,
        ),
        donate_argnames='init_sampling_state',
        static_argnames=(
            'chunk_size',
            'steps',
            'remasking',
            'stochastic_transfer',
            'right_shift_logits',
            'scheduler_name',
        ),
    )
    self.mask_and_pad_to = jax.jit(
        DiffusionSamplingState.mask_and_pad_to,
        donate_argnames='self',
        static_argnames=['length', 'mask_all'],
    )
    self.model_params = params

  @property
  def eos_ids(self) -> list[int]:
    return self.input_processor.eos_ids

  def generate(
      self,
      input_text: (
          sampling_lib.SamplingInput | Sequence[sampling_lib.SamplingInput]
      ),
      prng_key: int | PRNGKey | None = None,
      params: PyTree = None,
      sampling_params: DiffusionSamplingParams | None = None,
      include_eos_in_output_text: bool = False,
      batch_size: int | None = None,
  ) -> list[model_lib.SamplingOutput] | list[list[model_lib.SamplingOutput]]:
    """Generate samples from a given input text.

    Args:
      input_text: Single input or sequence of inputs to generate samples for.
        Input can be either string or sequence of Chunks.
      prng_key: A PRNGKey or seed for controlling the randomness. The key would
        be released inside, and cannot be reused.
      params: parameters of the model, if None, use the default parameters.
      sampling_params: Sampling params to use for the generation.
      include_eos_in_output_text: Whether to include the eos token when
        generating the `output_text` field of the sampling outputs. Note that
        even if this is set to `True`, the `vocab.decode` can still skip the eos
        token.
      batch_size: The batch size to use for the generation. If not specified,
        the batch size will be inferred from the length of the input text.
        
    Returns:
      If the `input_text` is a single text string or a single raw sequence,
      returns a list of `SamplingOutput`, else if the `input_text` is a
      list of text strings or a list of raw sequences, returns a list of list of
      `SamplingOutput`.

      The result `SamplingOutput` instances for each `input_text` are
      ranked by the `sort_by` field of the `sampling_params`.

      Note that the eos token and bos token are included in the
      `output_token_ids` and `input_token_ids` field of the `SamplingOutput`,
      but the `input_token_scores` will not include the bos token so its length
      is one less than `input_token_ids`.
    """
    if params is None:
      params = self.model_params

    if prng_key is None:
      seed = int(time.time() * 1000)
      # This is to guarantee all hosts have the same seed.
      seed = jax.experimental.multihost_utils.broadcast_one_to_all(seed)
      prng_key = jax.random.key(seed=seed)
    elif isinstance(prng_key, int):
      prng_key = jax.random.key(seed=prng_key)
    if sampling_params is None:
      sampling_params = self.default_sampling_params

    is_singleton_input = isinstance(input_text, str)
    if input_text and isinstance(input_text[0], sampling_lib.Chunk):
      is_singleton_input = True

    if is_singleton_input:
      raw_inputs = [sampling_lib.input_as_chunks(input_text)]
    else:
      raw_inputs = [sampling_lib.input_as_chunks(x) for x in input_text]

    unpadded_inputs = [
        self.input_processor.encode(
            x, max_input_len=sampling_params.max_input_len)
        for x in raw_inputs
    ]
    processed_input = sampling_lib.ProcessedInputBatch.from_unpadded_inputs(
        unpadded_inputs, pad_id=self.input_processor.pad_id
    )

    # Compute before padding the batch which may create length zero inputs.
    decoding_schedule = sampling_params.get_decoding_schedule(
        min_input_length=processed_input.min_length,
        max_input_length=processed_input.max_length,
    )

    if batch_size is not None:
      if processed_input.batch_size > batch_size:
        raise ValueError(
            f'Batch size {processed_input.batch_size=} is larger than the'
            f' specified batch size {batch_size=}.'
        )
      if processed_input.batch_size < batch_size:
        processed_input = processed_input.pad_batch_to(batch_size)
        logging.info('processed_input=%s after batch padding', processed_input)

    mask_token_id = sampling_params.mask_token_id
    if mask_token_id is None:
      mask_token_id = self.mask_token_id
    if mask_token_id is None:
      raise ValueError(
          'mask_token_id must be set in DiffusionSamplingParams or DLMInterface.'
      )
    processed_input = processed_input.pad_to(
        max(
            decoding_schedule.get_next_length(processed_input.max_length - 1),
            decoding_schedule.prefill_size,
        ),
        pad_id=self.input_processor.pad_id,
    )
    
    if sampling_params.num_samples > 1:
      processed_input = processed_input.repeat(sampling_params.num_samples)

    position = decoding_schedule.begin_position

    token_scores = jnp.zeros(
        (processed_input.batch_size, decoding_schedule.prefill_size + 1),
        dtype=jnp.float32,
    )
    token_logprobs = jnp.zeros_like(token_scores)

    sampling_state = DiffusionSamplingState(
        prng_key=jnp.copy(prng_key),
        position=jnp.array(position),
        decode_state=None,
        tokens=processed_input.tokens,
        token_logprobs=token_logprobs,
        token_scores=token_scores,
        input_lens=jnp.reshape(processed_input.lengths, [-1, 1]),
        max_decode_steps=einops.repeat(
            jnp.array(sampling_params.max_decode_steps),
            '-> b 1',
            b=processed_input.batch_size,
        ),
        eos_ids=jnp.array(self.input_processor.eos_ids, dtype=jnp.int32),
    )

    # NOTE that `position + 1` is the output position.
    logging.info('position: %d', position)
    logging.info('decoding chunk size: %d', decoding_schedule.chunk_size)
    logging.info('max_input_len: %d', processed_input.max_length)
    logging.info(
        'sampling_params.max_decode_steps: %d',
        sampling_params.max_decode_steps,
    )
    logging.info(
        'sampling_params.max_seq_len: %d', sampling_params.max_seq_len
    )
    mask_all = False
    while position < decoding_schedule.end_position:
      chunk_size = sampling_params.chunk_size
      length = decoding_schedule.get_next_length(position + chunk_size) + 1
      sampling_state = self.mask_and_pad_to(
          sampling_state,
          length=length,
          mask_block_size=decoding_schedule.chunk_size,
          mask_id=mask_token_id,
          pad_id=self.input_processor.pad_id,
          mask_all=mask_all,
      )
      sampling_state = self.decode_fn(
          params=params,
          init_sampling_state=sampling_state,
          extra_inputs=processed_input.extra_inputs,
          temperature=sampling_params.temperature,
          mask_token_id=mask_token_id,
          pad_id=self.input_processor.pad_id,
          chunk_size=chunk_size,
          steps=sampling_params.steps,
          remasking=sampling_params.remasking,
          stochastic_transfer=sampling_params.stochastic_transfer,
          right_shift_logits=sampling_params.right_shift_logits,
          scheduler_name=sampling_params.scheduler_name,
      )
      position = jax.device_get(sampling_state.position)
      if jax.device_get(sampling_state.all_has_ended):
        break
      mask_all = bool(length >= decoding_schedule.end_position + 1)
    # Post process the outputs.
    all_raw_token_ids = jax.experimental.multihost_utils.process_allgather(
        sampling_state.tokens, tiled=True
    ).tolist()

    sample_outputs = []
    num_outputs = len(raw_inputs) * sampling_params.num_samples
    for i in range(num_outputs):
      raw_token_ids = all_raw_token_ids[i]
      assert isinstance(raw_token_ids, list)
      assert isinstance(raw_token_ids[0], int)
      input_token_ids = []
      input_token_scores = []
      output_token_ids = []
      output_token_scores = []
      output_token_logprobs = []
      for t, token_id in enumerate(raw_token_ids):
        if t >= min(
            # Ensure python int to prevent overflow.
            int(processed_input.lengths[i])
            + sampling_params.max_decode_steps,
            sampling_params.max_seq_len,
        ):
          break
        if t < processed_input.lengths[i]:
          input_token_ids.append(token_id)
          if t > 0:
            input_token_scores.append(0.0)
        else:
          output_token_ids.append(token_id)
          output_token_scores.append(0.0)
          output_token_logprobs.append(0.0)
          if token_id in self.input_processor.eos_ids:
            # Generated eos token can only appear in output_tokens.
            break

      ends_in_eos = (
          output_token_ids
          and output_token_ids[-1] in self.input_processor.eos_ids
      )
      if ends_in_eos and not include_eos_in_output_text:
        output_chunks = self.input_processor.decode(output_token_ids[:-1])
      else:
        output_chunks = self.input_processor.decode(output_token_ids)

      input_index = i // sampling_params.num_samples
      sample_outputs.append(
          SamplingOutput(
              input_chunks=raw_inputs[input_index],
              output_chunks=output_chunks,
              input_token_ids=input_token_ids,
              output_token_ids=output_token_ids,
              output_token_logprobs=output_token_logprobs,
              input_token_scores=input_token_scores,
              output_token_scores=output_token_scores,
              is_truncated=(not ends_in_eos),
              processed_input=unpadded_inputs[input_index],
          )
      )

    if not is_singleton_input:
      sample_outputs = [
          sample_outputs[i : i + sampling_params.num_samples]
          for i in range(0, len(sample_outputs), sampling_params.num_samples)
      ]

    if sampling_params.sort_by is not None:
      if is_singleton_input:
        sample_outputs.sort(key=lambda x: getattr(x, sampling_params.sort_by))
      else:
        for batch in sample_outputs:
          assert isinstance(batch, list)
          batch.sort(key=lambda x: getattr(x, sampling_params.sort_by))

    return sample_outputs
