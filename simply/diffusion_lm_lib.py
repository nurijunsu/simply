"""Functions related to diffusion language model training and sampling."""

from collections.abc import Callable, MutableMapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from simply.utils import sharding as sharding_lib
from simply.utils import registry

Batch = MutableMapping[str, np.ndarray | jnp.ndarray]
CollectExtraLossFn = Callable[[Any], tuple[jnp.ndarray, dict[str, Any]]]


class AlphaSchedulerRegistry(registry.RootRegistry):
  """Registry for alpha schedulers used in diffusion language models."""
  namespace: ClassVar[str] = 'Scheduler'


@registry.AlphaSchedulerRegistry.register
class BaseAlphaScheduler:
  """Base class for alpha schedulers in diffusion language models."""

  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
    return self.alpha(t)
  
  def alpha(self, t: jnp.ndarray) -> jnp.ndarray:
    """Compute alpha(t) for a batch of timesteps t ∈ [0, 1]."""
    raise NotImplementedError
  
  def alpha_derivative(self, t: jnp.ndarray) -> jnp.ndarray:
    """Compute dα/dt for a batch of timesteps t ∈ [0, 1]."""
    raise NotImplementedError
  
  def reverse_mask_probability(self, s: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Compute the reverse mask probability from timestep t to s. Requires s < t elementwise."""
    return (1 - self(s)) / (1 - self(t))


@registry.AlphaSchedulerRegistry.register
class LinearAlphaScheduler(BaseAlphaScheduler):
  """Linear alpha scheduler: α(t) = 1 - t."""

  def alpha(self, t: jnp.ndarray) -> jnp.ndarray:
    return 1.0 - t

  def alpha_derivative(self, t: jnp.ndarray) -> jnp.ndarray:
    return -jnp.ones_like(t)


@registry.AlphaSchedulerRegistry.register
class CosineAlphaScheduler(BaseAlphaScheduler):
  """Cosine alpha scheduler: α(t) = 1 - cos(π(1-t)/2)."""

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
  scheduler = AlphaSchedulerRegistry.get(scheduler_name)

  t = time_epsilon + (1- time_epsilon) * jax.random(inputs.shape[0], rng_key, dtype=jnp.float32)
  mask_ratio = 1.0 - scheduler(t).unsqueeze(1).expand(inputs.shape[0], inputs.shape[1])

  loss_weights = batch.get('decoder_loss_weights', None)
  if loss_weights is None:
    loss_weights = jnp.ones_like(inputs, dtype=jnp.float32)
  else:
    loss_weights = loss_weights.astype(jnp.float32)

  mask_candidates = loss_weights > 0
  mask = (jax.random.uniform(rng_key, inputs.shape) < mask_ratio) & (
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
    collect_extra_loss_fn: CollectExtraLossFn | None = None,
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
  if collect_extra_loss_fn is not None and model_extra_output:
    extra_loss, extra_metric_dict = collect_extra_loss_fn(model_extra_output)
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
    collect_extra_loss_fn: CollectExtraLossFn | None = None,
) -> tuple[jnp.ndarray, dict[str, Any]]:
  return compute_diffusion_loss(
      model,
      params,
      batch,
      right_shift=right_shift,
      add_extra_loss=True,
      collect_extra_loss_fn=collect_extra_loss_fn,
  )


def compute_diffusion_eval_loss(
    model,
    params,
    batch: Batch,
    *,
    right_shift: bool = False,
    collect_extra_loss_fn: CollectExtraLossFn | None = None,
) -> tuple[jnp.ndarray, dict[str, Any]]:
  return compute_diffusion_loss(
      model,
      params,
      batch,
      right_shift=right_shift,
      add_extra_loss=False,
      collect_extra_loss_fn=collect_extra_loss_fn,
  )
