# -*- coding: utf-8 -*-

from collections import namedtuple

from TF_utils import TF_utils
import tensorflow as tf
tf_utils = TF_utils()


class BeamSearchState(namedtuple("BeamSearchState",
                                 ("inputs", "finish"))):
    pass


class BeamSearchBatch:
    def __init__(self,
                 model,
                 sess,
                 beam_size,
                 end_id,
                 max_step,
                 normalize_by_length,
                 alpha,
                 graph):
        self.model = model
        self.beam_size = beam_size
        self.start_token_idx = 0
        self.end_token_idx = end_id
        self.pad_token_idx = end_id
        # self.id2token = vocab.idx2token
        self.max_step = max_step
        self.alpha = alpha
        self.sess = sess
        self.graph = graph
        graph.as_default()

        # placeholder
        self.features = features
        self.normalize_by_length = normalize_by_length
        self.features['title_idx'] = input_title_idx
        self.search()

    def _is_finished(self, t, s):
        '''
        :param t: time
        :param s: BeamSearchStatezi
        :return: 函数主要是判断训练是否能继续。同时满足下面两个条件循环可以继续：
        1. 当前的长度不能超过最大长度限制
        2. 本步骤中有得分高于前一步中最差的得分的序列
        '''
        log_probs = s.inputs[1]  # [batch_size, beam_size] 这里应该是排序之后的，需在_beam_search_step中验证,topk时已经处理
        finished_flags = s.finish[0]  # [batch_size, beam_size]
        finished_scores = s.finish[2]  # [batch_size, beam_size]
        # length_penalty = tf.pow(((5.0 + tf.to_float(self.max_step)) / 6.0), self.alpha)  # 长度惩罚项
        length_penalty = tf.to_float(self.max_step)
        best_alive_score = log_probs[:, 0]/length_penalty  # 当前最高的log_prob  [batch_zise]

        '''
        [0,0]如果都没有结束，worst_finished_score就仍然是tf.float32.min 
        [0,1]如果其中有一个beam_idx结束了，可以另外一个
        [1,1]如果都结束了，只能比较往后扩一个的score是否比不扩大
        '''
        worst_finished_score = tf.reduce_min(finished_scores * tf.to_float(finished_flags),  axis=1)  # 每个beam中选择最低得分
        add_mask = 1.0 - tf.to_float(tf.reduce_any(finished_flags, 1))  # beam_size维度进行逻辑或，除非每个beam都没结束，返回1，否则0
        worst_finished_score += tf.float32.min * add_mask

        bound_is_met = tf.reduce_all(tf.greater(worst_finished_score, best_alive_score))  # 除非每条数据的所有的最差>最好，此时为1
        cond = tf.logical_and(tf.less(t, self.max_step), tf.logical_not(bound_is_met))  # 每条最差>最好，返回0

        return cond

    def _loop_fn(self, t, s):
        outs = self._beam_search_step(t, s)
        return outs

    def _beam_search_step(self, time, state):
        '''
        思路：
        1. 加载当前的序列，并往前走一步看是概率值
        2. 得到之后分两部分处理：到end_token_idx的和未到的，到的从中选top_k,未到的也选top_k
        '''

        # 1. 前进一步，计算当前score并筛选top-2k，生成新的2k个序列
        seqs, log_probs = state.inputs[:2]  # 当前序列和当前序列的得分[batch_size,beam_size, cur_len]和[batch_size,beam_size]
        flat_seqs = tf_utils.merge_first_two_dims(seqs)  # 相当于将input_y reshape成 [batch_size*beam_size, cur_len]
        self.features['query_input_idx'] = flat_seqs  # 确保每次的query_input_idx的改变
        logits = self.model.decode(self.features, self.memory, False)[0]
        step_log_probs = tf.nn.log_softmax(logits)
        step_log_probs = tf_utils.split_first_two_dims(step_log_probs, self.batch_size, self.beam_size)  # [batch_size, beam_zise, len2, vocab_size]
        step_log_probs = step_log_probs[:, :, -1, :]  # [batch_size, beam_zise, len2, vocab_size]
        curr_log_probs = tf.expand_dims(log_probs, 2) + step_log_probs  # 基于当前步骤向后看一步，下一步在vocab_size上的分别
        # Apply length penalty
        # length_penalty = tf.pow((5.0 + tf.to_float(time + 1)) / 6.0, self.alpha)
        length_penalty = tf.to_float(time + 1)
        curr_scores = curr_log_probs / length_penalty
        vocab_size = curr_scores.shape[-1].value or tf.shape(curr_scores)[-1]
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

        # 2. 抑制结束的序列，从可以发展的序列中选择topk
        # Expand sequences： time
        # Suppress finished sequences
        # print('symbol_indices:', symbol_indices)
        flags = tf.equal(symbol_indices, self.end_token_idx)
        # [batch, 2 * self.beam_size]
        alive_scores = top_scores + tf.to_float(flags) * tf.float32.min  # 如果结束，当前的score将的非常小
        # [batch, self.beam_size]
        alive_scores, alive_indices = tf.nn.top_k(alive_scores, self.beam_size)  # 当前的topk
        alive_symbols = tf_utils.gather_2d(symbol_indices, alive_indices)
        alive_indices = tf_utils.gather_2d(beam_indices, alive_indices)
        alive_seqs = tf_utils.gather_2d(seqs, alive_indices)
        # [self.batch_size, self.beam_size, time + 1]
        alive_seqs = tf.concat([alive_seqs, tf.expand_dims(alive_symbols, 2)], 2)
        alive_log_probs = alive_scores * length_penalty
        # 上面选择了当前步骤最优的topk

        # 3. 从已经结束的（到end_token_idx）中选top_k
        # Select finished sequences 从本轮可以结束的序列（包括本轮和上一轮）中选择topk个得分最好的进行更新
        prev_fin_flags, prev_fin_seqs, prev_fin_scores = state.finish
        # [batch, 2 * self.beam_size]
        step_fin_scores = top_scores + (1.0 - tf.to_float(flags)) * tf.float32.min  # 屏蔽掉未结束的序列
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

    def search(self):
        # 1. 初始化状态
        memory = self.model.encode(self.features, False)[0]
        self.memory = tf_utils.merge_first_two_dims((tf_utils.tile_to_beam_size(memory, self.beam_size)))
        title_idx = self.features['title_idx']
        # self.features['title_idx'] = tf_utils.np_merge_first_two_dims(tf_utils.np_tile_to_beam_size(title_idx, self.beam_size))
        self.features['title_idx'] = tf_utils.merge_first_two_dims(tf_utils.tile_to_beam_size(title_idx, self.beam_size))
        self.batch_size = tf_utils.infer_shape(title_idx)[0]
        # self.features['query_target_idx'] = None  # 循环中的的decode部分要访问这两个值，虽然无用，但是要填充上。
        # self.features['query'] = None  # 循环中的decode部分要访问这两个值，虽然无用，但是要填充上。

        init_seqs = tf.fill([self.batch_size, self.beam_size, 1], self.start_token_idx,)  # [self.batch_size, self.beam_size, 1] 初始序列填充的是start_token_idx
        init_log_probs = tf.constant([[0.] + [tf.float32.min] * (self.beam_size - 1)])  # 初始概率 [1, self.beam_size]
        init_log_probs = tf.tile(init_log_probs, [self.batch_size, 1])  # [batch_size,beam_size]
        init_scores = tf.zeros_like(init_log_probs)  # [batch_size,beam_size]
        fin_seqs = tf.zeros([self.batch_size, self.beam_size, 1], tf.int32)  # [batch_size,beam_size, len2]
        fin_scores = tf.fill([self.batch_size, self.beam_size], tf.float32.min)  # [batch_size,beam_size]
        fin_flags = tf.zeros([self.batch_size, self.beam_size], tf.bool)  # [batch_size,beam_size]

        # print('init_seqs:看是不是int32', init_seqs)
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
        # 2. 序列扩展
        # tf.while_loop是tf里面的循环，满足条件_is_finished，就能执行循环体_loop_fn，输入的参数四
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

        final_seqs = tf.where(tf.reduce_any(final_flags, 1), final_seqs, alive_seqs)  # 结束就选择
        final_scores = tf.where(tf.reduce_any(final_flags, 1), final_scores, alive_scores)

        self.final_seqs = final_seqs
        self.final_scores = final_scores