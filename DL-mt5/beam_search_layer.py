from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from collections import namedtuple

from TF_utils import TF_utils
import tensorflow as tf
tf_utils = TF_utils()

class BeamSearchState(namedtuple("BeamSearchState",
                                 ("inputs", "finish"))):
    pass

class BeamSearchLayer(Layer):

    def __init__(self, decoder, max_step, start_token_idx, end_token_idx, beam_size=1, **kwargs):
        self.decoder = decoder
        self.max_step = max_step
        self.beam_size = beam_size
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.pad_token_idx = start_token_idx
        super(BeamSearchLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        super(BeamSearchLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        memory = x[0][0]
        self.memory = tf_utils.merge_first_two_dims((tf_utils.tile_to_beam_size(memory, self.beam_size)))
        title_idx = x[1][0]
        # self.features['title_idx'] = tf_utils.np_merge_first_two_dims(tf_utils.np_tile_to_beam_size(title_idx, self.beam_size))
        # self.features['title_idx'] = tf_utils.merge_first_two_dims(tf_utils.tile_to_beam_size(title_idx, self.beam_size))
        self.batch_size = tf_utils.infer_shape(memory)[0]

        init_seqs = tf.fill([self.batch_size, self.beam_size, 1], self.start_token_idx,)  # [self.batch_size, self.beam_size, 1]
        init_log_probs = tf.constant([[0.] + [tf.float32.min] * (self.beam_size - 1)])
        init_log_probs = tf.tile(init_log_probs, [self.batch_size, 1])  # [batch_size,beam_size]
        init_scores = tf.zeros_like(init_log_probs)  # [batch_size,beam_size]
        fin_seqs = tf.zeros([self.batch_size, self.beam_size, 1], tf.int32)  # [batch_size,beam_size, len2]
        fin_scores = tf.fill([self.batch_size, self.beam_size], tf.float32.min)  # [batch_size,beam_size]
        fin_flags = tf.zeros([self.batch_size, self.beam_size], tf.bool)  # [batch_size,beam_size]

        state = BeamSearchState(
            inputs=(init_seqs, init_log_probs, init_scores),
            finish=(fin_flags, fin_seqs, fin_scores),
        )

        time = tf.constant(0, name="time")
        shape_invariants = BeamSearchState(
            inputs=(tf.TensorShape([None, None, None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None])),
            finish=(tf.TensorShape([None, None]),
                    tf.TensorShape([None, None, None]),
                    tf.TensorShape([None, None]))
        )

        outputs = tf.while_loop(self._is_finished, self._loop_fn, [time, state],
                                shape_invariants=[tf.TensorShape([]),
                                                  shape_invariants],
                                parallel_iterations=1,
                                back_prop=False)

        final_state = outputs[1]
        alive_seqs = final_state.inputs[0]
        alive_scores = final_state.inputs[2]
        final_flags = final_state.finish[0]
        final_seqs = final_state.finish[1]
        final_scores = final_state.finish[2]

        alive_seqs.set_shape([None, self.beam_size, None])
        final_seqs.set_shape([None, self.beam_size, None])

        final_seqs = tf.where(tf.reduce_any(final_flags, 1), final_seqs, alive_seqs) 
        final_scores = tf.where(tf.reduce_any(final_flags, 1), final_scores, alive_scores)

        self.final_seqs = final_seqs
        self.final_scores = final_scores
        return final_seqs, final_scores

    def _is_finished(self, t, s):

        log_probs = s.inputs[1]  # [batch_size, beam_size]
        finished_flags = s.finish[0]  # [batch_size, beam_size]
        finished_scores = s.finish[2]  # [batch_size, beam_size]
        # length_penalty = tf.pow(((5.0 + tf.to_float(self.max_step)) / 6.0), self.alpha)
        length_penalty = tf.compat.v1.to_float(self.max_step)
        best_alive_score = log_probs[:, 0]/length_penalty

        worst_finished_score = tf.reduce_min(finished_scores * tf.compat.v1.to_float(finished_flags),  axis=1)
        add_mask = 1.0 - tf.compat.v1.to_float(tf.reduce_any(finished_flags, 1))
        worst_finished_score += tf.float32.min * add_mask

        bound_is_met = tf.reduce_all(tf.greater(worst_finished_score, best_alive_score))
        cond = tf.logical_and(tf.less(t, self.max_step), tf.logical_not(bound_is_met))

        return cond

    def _loop_fn(self, t, s):
        outs = self._beam_search_step(t, s)
        return outs

    def _beam_search_step(self, time, state):

        seqs, log_probs = state.inputs[:2]
        flat_seqs = tf_utils.merge_first_two_dims(seqs)
        # self.features['query_input_idx'] = flat_seqs
        logits = self.decoder([self.memory, flat_seqs])
        step_log_probs = tf.nn.log_softmax(logits)
        step_log_probs = tf_utils.split_first_two_dims(step_log_probs, self.batch_size, self.beam_size)  # [batch_size, beam_zise, len2, vocab_size]
        step_log_probs = step_log_probs[:, :, -1, :]  # [batch_size, beam_zise, len2, vocab_size]
        curr_log_probs = tf.expand_dims(log_probs, 2) + step_log_probs
        # Apply length penalty
        # length_penalty = tf.pow((5.0 + tf.to_float(time + 1)) / 6.0, self.alpha)
        length_penalty = tf.compat.v1.to_float(time + 1)
        curr_scores = curr_log_probs / length_penalty
        vocab_size = curr_scores.shape[-1] or tf.shape(curr_scores)[-1]
        # Select top-k candidates
        # [self.batch_size, self.beam_size * vocab_size]
        curr_scores = tf.reshape(curr_scores, [-1, self.beam_size * vocab_size])
        # [self.batch_size, 2 * self.beam_size]
        top_scores, top_indices = tf.nn.top_k(curr_scores, k=2 * self.beam_size)
        # Shape: [self.batch_size, 2 * self.beam_size]
        beam_indices = top_indices // vocab_size   # [batch_size, beam_size]
        symbol_indices = top_indices % vocab_size  # [batch_size, beam_size]
        # Expand sequences
        # [batch_size, 2 * beam_size, time]
        candidate_seqs = tf_utils.gather_2d(seqs, beam_indices)  # [batch_size, beam_size, len_cur]
        candidate_seqs = tf.concat([candidate_seqs, tf.expand_dims(symbol_indices, 2)], 2)  # [batch_size, 2*beam_size, cur_len+1]

        # Expand sequences： time
        # Suppress finished sequences
        # print('symbol_indices:', symbol_indices)
        flags = tf.equal(symbol_indices, self.end_token_idx)
        # [batch, 2 * self.beam_size]
        alive_scores = top_scores + tf.compat.v1.to_float(flags) * tf.float32.min
        # [batch, self.beam_size]
        alive_scores, alive_indices = tf.nn.top_k(alive_scores, self.beam_size)
        alive_symbols = tf_utils.gather_2d(symbol_indices, alive_indices)
        alive_indices = tf_utils.gather_2d(beam_indices, alive_indices)
        alive_seqs = tf_utils.gather_2d(seqs, alive_indices)
        # [self.batch_size, self.beam_size, time + 1]
        alive_seqs = tf.concat([alive_seqs, tf.expand_dims(alive_symbols, 2)], 2)
        alive_log_probs = alive_scores * length_penalty

        prev_fin_flags, prev_fin_seqs, prev_fin_scores = state.finish
        # [batch, 2 * self.beam_size]
        step_fin_scores = top_scores + (1.0 - tf.compat.v1.to_float(flags)) * tf.float32.min
        # [batch, 3 * self.beam_size]
        fin_flags = tf.concat([prev_fin_flags, flags], axis=1)  #
        fin_scores = tf.concat([prev_fin_scores, step_fin_scores], axis=1)
        # [batch, self.beam_size]
        fin_scores, fin_indices = tf.nn.top_k(fin_scores, self.beam_size)
        fin_flags = tf_utils.gather_2d(fin_flags, fin_indices)
        pad_seqs = tf.fill([self.batch_size, self.beam_size, 1], tf.constant(self.pad_token_idx, tf.int32))
        prev_fin_seqs = tf.concat([prev_fin_seqs, pad_seqs], axis=2)
        fin_seqs = tf.concat([prev_fin_seqs, candidate_seqs], axis=1)
        fin_seqs = tf_utils.gather_2d(fin_seqs, fin_indices)

        new_state = BeamSearchState(
            inputs=(alive_seqs, alive_log_probs, alive_scores),
            finish=(fin_flags, fin_seqs, fin_scores),
        )

        return time + 1, new_state

    def compute_mask(self, inputs, mask=None):
        # if callable(mask):
        #     return mask(inputs, mask)
        # return mask
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])