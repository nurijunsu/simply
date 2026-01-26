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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from simply import diffusion_lm_lib
from simply.utils import sampling_lib


class DiffusionLmLibTest(absltest.TestCase):

  def test_diffusion_decoding_schedule(self):
    params = diffusion_lm_lib.DiffusionSamplingParams(
        max_decode_steps=1000,
        chunk_size=128,
    )

    schedule = params.get_decoding_schedule(
        min_input_length=200, max_input_length=400
    )

    self.assertEqual(256, schedule.prefill_size)
    self.assertEqual(199, schedule.begin_position)
    self.assertEqual(1399, schedule.end_position)
    self.assertEqual(128, schedule.chunk_size)

  def test_apply_diffusion_mask_respects_loss_weights(self):
    batch = {
        'decoder_input_tokens': jnp.array(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32
        ),
        'decoder_loss_weights': jnp.array(
            [[1, 0, 1, 0], [0, 0, 1, 1]], dtype=jnp.float32
        ),
    }
    key = jax.random.key(0)
    masked = diffusion_lm_lib.apply_diffusion_mask_to_batch(
        batch,
        key,
        mask_token_id=99,
        time_epsilon=1e-5,
        scheduler_name='LinearAlphaScheduler',
    )

    np.testing.assert_equal(
        masked['decoder_diffusion_targets'],
        np.asarray(batch['decoder_input_tokens']),
    )

    loss_weights = np.asarray(batch['decoder_loss_weights'])
    masked_weights = np.asarray(masked['decoder_loss_weights'])
    masked_tokens = np.asarray(masked['decoder_input_tokens'])
    original_tokens = np.asarray(batch['decoder_input_tokens'])

    zero_mask = loss_weights == 0
    np.testing.assert_equal(masked_tokens[zero_mask], original_tokens[zero_mask])
    np.testing.assert_equal(masked_weights[zero_mask], 0)

    pos_mask = loss_weights > 0
    valid = np.logical_or(
        masked_tokens[pos_mask] == original_tokens[pos_mask],
        masked_tokens[pos_mask] == 99,
    )
    self.assertTrue(np.all(valid))
    self.assertTrue(np.all(masked_weights <= loss_weights))

  def test_continue_decode_replaces_mask_tokens(self):
    vocab_size = 16
    mask_token_id = 15
    tokens = jnp.array([[1, mask_token_id, 2, mask_token_id]], dtype=jnp.int32)

    def apply_fn(params, tokens, segment_positions=None, extra_inputs=None):
      next_ids = (tokens + 1) % vocab_size
      logits = jax.nn.one_hot(next_ids, vocab_size, dtype=jnp.float32)
      return logits, {}

    init_state = diffusion_lm_lib.DiffusionSamplingState(
        prng_key=jax.random.key(0),
        position=jnp.array(0),
        decode_state=None,
        tokens=tokens,
        token_logprobs=jnp.zeros_like(tokens, dtype=jnp.float32),
        token_scores=jnp.zeros_like(tokens, dtype=jnp.float32),
        input_lens=jnp.array([[2]], dtype=jnp.int32),
        max_decode_steps=jnp.array([[1]], dtype=jnp.int32),
        eos_ids=jnp.array([99], dtype=jnp.int32),
    )

    out_state = diffusion_lm_lib.continue_decode(
        apply_fn=apply_fn,
        params={},
        init_sampling_state=init_state,
        extra_inputs=None,
        temperature=0.0,
        mask_token_id=mask_token_id,
        pad_id=0,
        chunk_size=tokens.shape[1],
        steps=1,
        remasking='low_confidence',
        stochastic_transfer=False,
        right_shift_logits=False,
        scheduler_name='LinearAlphaScheduler',
    )

    expected = np.array(
        [[1, (mask_token_id + 1) % vocab_size, 2, (mask_token_id + 1) % vocab_size]],
        dtype=np.int32,
    )
    np.testing.assert_equal(np.asarray(out_state.tokens), expected)
    self.assertEqual(int(out_state.position), 2)

  def test_generate_with_mocked_model(self):
    class DummyInputProcessor:
      eos_ids = [99]
      pad_id = 0
      bos_id = 1

      def encode(self, chunks, max_input_len=None):
        text = sampling_lib.chunks_as_text(chunks)
        tokens = [self.bos_id] + [3] * len(text)
        if max_input_len is not None:
          tokens = tokens[-max_input_len:]
        return sampling_lib.ProcessedInput(tokens=tokens, extra_inputs=None)

      def decode(self, token_ids):
        return [
            sampling_lib.Chunk(
                type=sampling_lib.Chunk.Type.TEXT,
                content=' '.join(str(token_id) for token_id in token_ids),
            )
        ]

    class FakeModel:
      def __init__(self, vocab_size, token_id):
        self.vocab_size = vocab_size
        self.token_id = token_id

      def apply(self, params, tokens, segment_positions=None, extra_inputs=None):
        logits = jnp.full(tokens.shape + (self.vocab_size,), -1.0)
        logits = logits.at[..., self.token_id].set(1.0)
        return logits, {}

    vocab_size = 8
    token_id = 4
    mask_token_id = 7
    sampling_params = diffusion_lm_lib.DiffusionSamplingParams(
        temperature=0.0,
        max_decode_steps=1,
        num_samples=1,
        chunk_size=5,
        steps=1,
        mask_token_id=mask_token_id,
        scheduler_name='LinearAlphaScheduler',
    )

    interface = diffusion_lm_lib.DLMInterface(
        model=FakeModel(vocab_size=vocab_size, token_id=token_id),
        params={},
        input_processor=DummyInputProcessor(),
        default_sampling_params=sampling_params,
        mask_token_id=mask_token_id,
    )

    interface.mask_and_pad_to = lambda state, **kwargs: (
        diffusion_lm_lib.DiffusionSamplingState.mask_and_pad_to(state, **kwargs)
    )
    interface.decode_fn = lambda *args, **kwargs: diffusion_lm_lib.continue_decode(
        apply_fn=interface.model.apply, *args, **kwargs
    )

    outputs = interface.generate(
        'hi', prng_key=0, sampling_params=sampling_params
    )
    self.assertLen(outputs, 1)
    self.assertEqual(outputs[0].output_token_ids, [4])

  def test_mask_and_pad_to_masks_trailing_pad(self):
    tokens = jnp.array(
        [[1, 2, 0, 0, 0], [1, 2, 3, 0, 0]], dtype=jnp.int32
    )
    state = diffusion_lm_lib.DiffusionSamplingState(
        prng_key=jax.random.key(0),
        position=jnp.array(0),
        decode_state=None,
        tokens=tokens,
        token_logprobs=jnp.zeros_like(tokens, dtype=jnp.float32),
        token_scores=jnp.zeros_like(tokens, dtype=jnp.float32),
        input_lens=jnp.array([[2], [3]], dtype=jnp.int32),
        max_decode_steps=jnp.array([[1], [1]], dtype=jnp.int32),
        eos_ids=jnp.array([99], dtype=jnp.int32),
    )

    masked = state.mask_and_pad_to(
        length=5,
        mask_block_size=2,
        mask_id=99,
        pad_id=0,
        mask_all=False,
    )

    expected = np.array(
        [[1, 2, 99, 99, 0], [1, 2, 3, 0, 0]], dtype=np.int32
    )
    np.testing.assert_equal(np.asarray(masked.tokens), expected)

    masked_all = state.mask_and_pad_to(
        length=5,
        mask_block_size=2,
        mask_id=99,
        pad_id=0,
        mask_all=True,
    )
    expected_all = np.array(
        [[1, 2, 99, 99, 99], [1, 2, 3, 99, 99]], dtype=np.int32
    )
    np.testing.assert_equal(np.asarray(masked_all.tokens), expected_all)


if __name__ == "__main__":
  absltest.main()
