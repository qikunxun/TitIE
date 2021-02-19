import tensorflow as tf
import modeling
import optimization

class BertModel(object):
    def __init__(self, num_classes, is_training=True, bert_config=None, learning_rate=None, dropout=None,
                num_train_steps=None, num_warmup_steps=None):
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bert_config = bert_config
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.create_placeholders(is_training)
        self.create_model_graph(is_training=is_training, num_classes=num_classes)

    def create_placeholders(self, is_training):
        self.input_ids = tf.placeholder(tf.int32, [None, None])
        self.input_mask = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
        self.segment_ids = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
        if is_training:
            self.start = tf.placeholder(tf.int32, [None])  # [batch_size]
            self.end = tf.placeholder(tf.int32, [None])  # [batch_size]
            self.reward = tf.placeholder(tf.float32, [None])
        # if self.use_weight and is_training:
        # self.weight = tf.placeholder(tf.float32, [None, 3])

    def create_feed_dict(self, batch, is_training=False):
        feed_dict = {
            self.input_ids: batch['input_ids'],
            self.input_mask: batch['input_mask'],
            self.segment_ids: batch['segment_ids']
        }

        if is_training:
            feed_dict[self.start] = batch['start']
            feed_dict[self.end] = batch['end']
            feed_dict[self.reward] = batch['reward']

        return feed_dict

    def create_model_graph(self, is_training=True, num_classes=None):
        bert_config = modeling.BertConfig.from_json_file(self.bert_config)

        # if not is_training:
        bert_config.hidden_dropout_prob = 0.0
        bert_config.attention_probs_dropout_prob = 0.0

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        # ========Prediction Layer=========
        final_hidden = model.get_sequence_output()

        final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]

        with tf.variable_scope("Answer_Pointer_Layer", reuse=tf.AUTO_REUSE):
            output_weights = tf.get_variable(
                "cls/squad/output_weights", [2, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(final_hidden,
                                         [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        logits = tf.reshape(logits, [batch_size, seq_length, 2])
        logits = tf.transpose(logits, [2, 0, 1])

        unstacked_logits = tf.unstack(logits, axis=0)

        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        self.start_probs = tf.nn.softmax(start_logits, axis=-1)
        self.end_probs = tf.nn.softmax(end_logits, axis=-1)
        self.start_predictions = tf.argmax(self.start_probs, 1)
        self.end_predictions = tf.argmax(self.end_probs, 1)
        if not is_training: return

        one_hot_start_positions = tf.one_hot(
            self.start, depth=seq_length, dtype=tf.float32)
        log_probs_start = tf.nn.log_softmax(start_logits, axis=-1)
        start_loss = -tf.reduce_mean(self.reward *  
            tf.reduce_sum(one_hot_start_positions * log_probs_start, axis=-1))

        one_hot_end_positions = tf.one_hot(
            self.end, depth=seq_length, dtype=tf.float32)
        log_probs_end = tf.nn.log_softmax(end_logits, axis=-1)
        end_loss = -tf.reduce_mean(self.reward * 
            tf.reduce_sum(one_hot_end_positions * log_probs_end, axis=-1))

        total_loss = (start_loss + end_loss) / 2.0

        self.loss = total_loss
        tvars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, var_list=tvars)
        # self.train_op = optimization.create_optimizer(
        #     self.loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, False)

class PolicyNetwork(object):
    def __init__(self, num_classes, is_training=True, bert_config=None, learning_rate=None, dropout=None,
                 num_train_steps=None, num_warmup_steps=None):
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bert_config = bert_config
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.create_placeholders(is_training)
        self.create_model_graph(is_training=is_training, num_classes=num_classes)

    def create_placeholders(self, is_training):
        self.input_ids = tf.placeholder(tf.int32, [None, None])
        self.input_mask = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
        self.segment_ids = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
        if is_training:
            self.truth = tf.placeholder(tf.int32, [None])  # [batch_size]
            self.reward = tf.placeholder(tf.float32, [])
        # if self.use_weight and is_training:
        # self.weight = tf.placeholder(tf.float32, [None, 3])

    def create_feed_dict(self, batch, is_training=False, is_training_pn=False):
        feed_dict = {
            self.input_ids: batch['input_ids'],
            self.input_mask: batch['input_mask'],
            self.segment_ids: batch['segment_ids']
        }

        if is_training:
            feed_dict[self.truth] = batch['label']
            feed_dict[self.reward] = batch['reward']

        return feed_dict

    def create_model_graph(self, is_training=True, num_classes=None):
        bert_config = modeling.BertConfig.from_json_file(self.bert_config)
        # if not is_training:
        bert_config.hidden_dropout_prob = 0.0
        bert_config.attention_probs_dropout_prob = 0.0
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        # ========Prediction Layer=========
        match_representation = model.get_pooled_output()

        match_dim = int(match_representation.shape[1])
        # with tf.variable_scope("Classification_Layer", reuse=tf.AUTO_REUSE):
        #     w_0 = tf.get_variable("w_0", [match_dim, num_classes], dtype=tf.float32,
        #                           initializer=tf.truncated_normal_initializer(stddev=0.02))
        #     b_0 = tf.get_variable("b_0", [num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            output_weights = tf.get_variable(
                "output_weights", [num_classes, match_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [num_classes], initializer=tf.zeros_initializer())

        # if is_training:
        #     match_representation = tf.nn.dropout(match_representation, 1 - self.dropout)
        logits = tf.matmul(match_representation, output_weights, transpose_b=True) + output_bias

        self.vec = match_representation
        self.prob = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.prob, 1)

        if not is_training: return

        gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix)
        self.loss = tf.reduce_mean(self.reward * self.loss)

        tvars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, var_list=tvars)
        # self.train_op = optimization.create_optimizer(
        #     self.loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, False)
