#! -*- coding: utf-8 -*-

from __future__ import print_function
import json
import os
os.environ['TF_KERAS'] = '1'
import numpy as np
import tensorflow as tf
import tokenization
# import tensorflow.keras as K
from tensorflow.keras.layers import Dense, Lambda
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss, Input
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import SpTokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn import metrics
from sklearn.utils import shuffle
from beam_search_layer import BeamSearchLayer

tf.logging.set_verbosity(tf.logging.INFO)

max_title_length = 64
batch_size = 8
max_epochs = 2
max_epochs_training = 3
warmup_proportion = 0.1
num_classes = 4
dropout = 0.1
learning_rate = 2e-4
radio = 0.7
episodes = 2
max_query_length = 13
min_query_length = 5

pretrained_path_t5 = './mt5_base'
config_path_t5 = os.path.join(pretrained_path_t5, 'mt5_base_config.json')
checkpoint_path_t5 = os.path.join(pretrained_path_t5, 'model.ckpt-1000000')
spm_path = os.path.join(pretrained_path_t5, 'sentencepiece_cn.model')
keep_tokens_path = os.path.join(pretrained_path_t5, 'sentencepiece_cn_keep_tokens.json')

pretrained_path_bert = '../span_extraction/chinese_L-12_H-768_A-12'
pretrained_path_pn = '../span_extraction/output'
config_path_bert = os.path.join(pretrained_path_bert, 'bert_config.json')
checkpoint_path_bert = os.path.join(pretrained_path_bert, 'bert_model.ckpt')
checkpoint_path_pn = os.path.join(pretrained_path_pn, 'model.ckpt-143874')
vocab_path_bert = os.path.join(pretrained_path_bert, 'vocab.txt')

data_path = './data'
data_file_train = os.path.join(data_path, 'train.txt')
data_file_valid = os.path.join(data_path, 'valid.txt')
data_file_predict = os.path.join(data_path, 'predict.txt')
data_file_un = os.path.join(data_path, 'train_un.txt')
valid_file_for_pretrain_policy_network = os.path.join(data_path, 'valid_selection.txt')
save_data_file_path = os.path.join(data_path, 'train_save.txt')

model_path = './model'
model_save_path = os.path.join(model_path, 'best_model.weights')
model_save_path_pn = os.path.join(model_path, 'best_model_pn.weights')
model_save_path_ori = os.path.join(model_path, 'ori_model.weights')

tokenizer_bert = tokenization.FullTokenizer(
        vocab_file=vocab_path_bert, do_lower_case=True)

def process_query(query, max_seq_length=(max_query_length + 2)):

    tokens_a = tokenizer_bert.tokenize(query)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)


    input_ids = tokenizer_bert.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, segment_ids


def load_data(filename):
    D = []
    i = 0
    with open(filename, encoding='utf-8') as f:
        for l in tqdm(f):
            item  = l.strip().split('\t')
            query = item[0]
            title = item[1]
            D.append(('s' + str(i), query, title, 1.))
            i += 1
            # if len(D) > 100: break
    np.random.shuffle(D)
    # D = D[:10000]
    return D

def load_valid_data(filename):
    D = []
    i = 0
    with open(filename, encoding='utf-8') as f:
        for l in tqdm(f):
            item  = l.strip().split('\t')
            query = item[0].split('||')
            title = item[1]
            D.append(('v' + str(i), query, title))
            i += 1
            # if len(D) > 100: break
    # np.random.shuffle(D)
    # D = D[:10000]
    return D

def load_data_un(filename):
    D = []
    i = 0
    with open(filename, encoding='utf-8') as f:
        for l in tqdm(f):
            title = l.strip().split('\t')[1]
            D.append(('u' + str(i), title))
            i += 1
            # if len(D) > 100: break
    np.random.shuffle(D)
    # D = D[:10000]
    return D

def load_valid_data_for_pn(filename):
    indices, segments, labels = [], [], []
    with open(filename, encoding='utf-8') as f:
        for l in tqdm(f):
            item = l.strip().split('\t')
            query = item[0]
            label = int(item[1])
            input_ids, segment_ids = process_query(query)
            indices.append(input_ids)
            segments.append(segment_ids)
            labels.append(label)
            # if len(labels) > 100: break
    indices = np.array(indices)
    segments = np.array(segments)
    labels = np.array(labels)
    return [indices, segments], labels

def extend_data(data_ori, data_new):
    data_ori.extend(data_new)
    return data_ori

train_data = load_data(data_file_train)
valid_data = load_data(data_file_valid)
test_data = load_data(data_file_valid)
data_un = load_data_un(data_file_un)
valid_x, valid_y = load_valid_data_for_pn(valid_file_for_pretrain_policy_network)

tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
keep_tokens = json.load(open(keep_tokens_path))


class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_c_token_ids, batch_t_token_ids = [], []
        for is_end, (id, query, title) in self.sample(random):
            c_token_ids, _ = tokenizer.encode(title, maxlen=max_title_length)
            t_token_ids, _ = tokenizer.encode(query, maxlen=max_query_length)
            batch_c_token_ids.append(c_token_ids)
            batch_t_token_ids.append([0] + t_token_ids)
            if len(batch_c_token_ids) == self.batch_size or is_end:
                batch_c_token_ids = sequence_padding(batch_c_token_ids)
                batch_t_token_ids = sequence_padding(batch_t_token_ids)
                yield [batch_c_token_ids, batch_t_token_ids], None
                batch_c_token_ids, batch_t_token_ids = [], []

class data_generator_pn(DataGenerator):
    def __iter__(self, random=False):
        indices, segments, labels = [], [], []
        for is_end, (id, query, label, reward) in self.sample(random):
            input_ids, segment_ids = process_query(query)
            indices.append(input_ids)
            segments.append(segment_ids)
            one_hot = np.eye(4)[label] * reward
            labels.append(one_hot)
            if len(indices) == self.batch_size or is_end:
                yield [np.array(indices), np.array(segments)], np.array(labels)
                indices, segments, labels = [], [], []

class CrossEntropy(Loss):
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_true = y_true[:, 1:]
        y_mask = K.cast(mask[1], K.floatx())[:, :-1]
        y_pred = y_pred[:, :-1]
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

with tf.variable_scope('t5'):
    t5 = build_transformer_model(
        config_path=config_path_t5,
        checkpoint_path=checkpoint_path_t5,
        keep_tokens=keep_tokens,
        model='t5.1.1',
        return_keras_model=False,
        name='T5',
    )

    encoder = t5.encoder
    decoder = t5.decoder
    model_t5 = t5.model
    model_t5.summary()
    outputs = BeamSearchLayer(decoder, max_query_length, 0, tokenizer._token_end_id, beam_size=1)([encoder.outputs, encoder.inputs])
    model_search = Model(model_t5.inputs, outputs)

    output = CrossEntropy(1)([model_t5.inputs[1], model_t5.outputs[0]])

    model = Model(model_t5.inputs, output)
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))
    model.save_weights(model_save_path_ori)

with tf.variable_scope('bert'):
    bert = build_transformer_model(
        config_path=config_path_bert,
        checkpoint_path=checkpoint_path_pn,
        model='bert',
        return_keras_model=False,
        sequence_length=(max_query_length + 2)
    )

    dropout_rate = 0.1
    inputs = bert.inputs[:2]
    context_emb = bert.model.output

    cls_emb = Lambda(lambda x: x[:, 0])(context_emb)

    cls_dense = Dense(
        units=768,
        activation='tanh',
        kernel_initializer=bert.initializer
    )
    cls_emb = cls_dense(cls_emb)

    output_dense = Dense(
        units=4,
        activation='softmax',
        kernel_initializer=bert.initializer
    )
    outputs = output_dense(cls_emb)
    pooler_weights = tf.train.load_variable(checkpoint_path_pn, 'bert/pooler/dense/kernel')
    K.set_value(cls_dense.kernel, pooler_weights)
    pooler_bias = tf.train.load_variable(checkpoint_path_pn, 'bert/pooler/dense/bias')
    K.set_value(cls_dense.bias, pooler_bias)
    #
    output_weights = tf.train.load_variable(checkpoint_path_pn, 'output_weights')
    K.set_value(output_dense.kernel, np.transpose(output_weights))
    output_bias = tf.train.load_variable(checkpoint_path_pn, 'output_bias')
    K.set_value(output_dense.bias, np.transpose(output_bias))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate * 0.01)
    model_pn = Model(inputs, outputs)
    model_pn.summary()
    model_pn.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model_pn.save_weights(model_save_path_pn)
y_pred = model_pn.predict(valid_x, batch_size=batch_size, verbose=1)
y_pred = y_pred.argmax(axis=-1)
tf.logging.info(metrics.classification_report(valid_y, y_pred))

class Autoquery(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        c_encoded = inputs[0]
        return decoder.predict([c_encoded, output_ids], steps=1)[:, -1]

    def generate(self, text, topk=1):
        c_token_ids, _ = tokenizer.encode(text, maxlen=max_title_length)
        c_encoded = encoder.predict(np.array([c_token_ids]))[0]
        output_ids, score = self.random_sample([c_encoded], 1)
        output_ids = output_ids[0]
        return tokenizer.decode([int(i) for i in output_ids]), score

    def generate_batch(self, texts):
        c_token_ids = []
        for text in texts:
            c_token_id, _ = tokenizer.encode(text, maxlen=max_title_length)
            while len(c_token_id) < max_title_length:
                c_token_id.append(0)
            c_token_ids.append(c_token_id)
        c_token_ids = np.array(c_token_ids)
        output_ids, scores_ = model_search.predict([c_token_ids, c_token_ids], batch_size=batch_size)
        queries = []
        scores = []
        for i in range(output_ids.shape[0]):
            output_id = output_ids[i, 0, :]
            query = tokenizer.decode([int(j) for j in output_id])
            queries.append(query)
            score = scores_[i][0]
            scores.append(score)
        return queries, scores

autoquery = Autoquery(start_id=0, end_id=tokenizer._token_end_id, maxlen=max_query_length)


class Evaluator(keras.callbacks.Callback):

    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights(model_save_path)
        metrics['best_bleu'] = self.best_bleu
        tf.logging.info('valid_data:' + str(metrics))

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for query, title in tqdm(data):
            total += 1
            query = ' '.join(query).lower()
            pred_query = ' '.join(autoquery.generate(title,
                                                     topk=topk)).lower()
            if pred_query.strip():
                scores = self.rouge.get_scores(hyps=pred_query, refs=query)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[query.split(' ')],
                    hypothesis=pred_query.split(' '),
                    smoothing_function=self.smooth
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


if __name__ == '__main__':

    evaluator = Evaluator()
    rouge = Rouge()
    for i in range(max_epochs):
        tf.logging.info('start training model')
        train_generator = data_generator(train_data, batch_size)
        if i > 0: model.load_weights(model_save_path)
        model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=max_epochs_training,
            # callbacks=[evaluator]
        )
        model.save_weights(model_save_path)
        tf.logging.info('start evaluating model')
        titles = []
        for id, query, title in tqdm(valid_data):
            titles.append(title)
        pred_queries, pred_scores = autoquery.generate_batch(titles)
        rouge_1, rouge_2, rouge_l, total = 0, 0, 0, 0
        for k, (id, query, title) in tqdm(enumerate(valid_data)):
            query = ' '.join(query).lower()
            pred_query = pred_queries[k]
            if len(pred_query) < 5 or len(pred_query) > 13:
                total += 1
                continue
            pred_query = ' '.join(pred_query).lower()
            if pred_query.strip():
                scores = rouge.get_scores(hyps=pred_query, refs=query)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                total += 1
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        tf.logging.info('rouge: {}\t{}\t{}'.format(rouge_1, rouge_2, rouge_l))

        train_data_u_sample = []
        titles = []
        for item in tqdm(data_un):
            title = item[1]
            titles.append(title)
        pred_queries, pred_scores = autoquery.generate_batch(titles)
        for k, item in tqdm(enumerate(data_un)):
            id = item[0]
            title = item[1]
            pred_query = pred_queries[k]
            score = pred_scores[k]
            train_data_u_sample.append((id, pred_query, title, score))

        train_data_u_sample = sorted(train_data_u_sample, key=lambda item: item[-1], reverse=True)
        train_data_u_sample = train_data_u_sample[:int(len(train_data_u_sample) * radio)]
        for episode in range(episodes):
            train_data_u_sample = shuffle(train_data_u_sample)
            tf.logging.info('start selecting sample')
            indices, segments = [], []
            tf.logging.info('len data u:' + str(len(train_data_u_sample)))
            for item in train_data_u_sample:
                query = item[1]
                input_ids, segment_ids = process_query(query)
                indices.append(input_ids)
                segments.append(segment_ids)
            indices = np.array(indices)
            segments = np.array(segments)
            valid_x_pn = [indices, segments]
            valid_pred_pn = model_pn.predict(valid_x_pn, batch_size=batch_size, verbose=1)
            valid_pred_pn = valid_pred_pn.argmax(axis=-1)
            num_expamles_rand = len(train_data_u_sample)
            assert valid_pred_pn.shape[0] == num_expamles_rand
            rewards = []
            train_data_pn = []
            for start in tqdm(range(0, num_expamles_rand, batch_size)):
                tf.logging.info('start training sub sample')
                end = start + batch_size
                batch = train_data_u_sample[start: end]
                batch_pred = valid_pred_pn[start: end]
                batch_new = []
                for k in range(len(batch)):
                    if batch_pred[k] == 0:
                        batch_new.append((batch[k][0], batch[k][1], batch[k][2]))
                if len(batch_new) == 0:
                    rewards.append(0)
                    for k in range(len(batch)):
                        train_data_pn.append((batch[k][0], batch[k][1], batch_pred[k], 0))
                    continue
                train_generator = data_generator(batch_new, len(batch))
                model.load_weights(model_save_path_ori)
                model.fit(
                    train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=max_epochs_training * 3,
                    # callbacks=[evaluator]
                )
                titles = []
                for id, query, title in tqdm(valid_data):
                    titles.append(title)
                pred_queries, pred_scores = autoquery.generate_batch(titles)
                rouge_1, rouge_2, rouge_l, total = 0, 0, 0, 0
                for k, (id, query, title) in tqdm(enumerate(valid_data)):
                    query = ' '.join(query).lower()
                    pred_query = pred_queries[k]
                    if len(pred_query) < 5 or len(pred_query) > 13:
                        total += 1
                        continue
                    pred_query = ' '.join(pred_query).lower()
                    if pred_query.strip():
                        scores = rouge.get_scores(hyps=pred_query, refs=query)
                        rouge_1 += scores[0]['rouge-1']['f']
                        rouge_2 += scores[0]['rouge-2']['f']
                        rouge_l += scores[0]['rouge-l']['f']
                        total += 1
                rouge_1 /= total
                rouge_2 /= total
                rouge_l /= total
                reward = (rouge_1 + rouge_2 + rouge_l) / 3
                tf.logging.info('Reward for valid se sub: %.3f' % reward)
                rewards.append(reward)
                for k in range(len(batch)):
                    train_data_pn.append((batch[k][0], batch[k][1], batch_pred[k], reward))
            rewards = np.array(rewards)
            train_data_pn_new = []
            for it in train_data_pn:
                reward_new = (it[3] - rewards.mean()) / np.sqrt(rewards.var())
                train_data_pn_new.append((it[0], it[1], it[2], reward_new))
            tf.logging.info('start training policy network')
            model_pn.load_weights(model_save_path_pn)
            train_generator_pn = data_generator_pn(train_data_pn_new, batch_size)
            model_pn.fit(
                train_generator_pn.forfit(),
                steps_per_epoch=len(train_generator_pn),
                epochs=1,
                # callbacks=[evaluator]
            )
            y_pred = model_pn.predict(valid_x, batch_size=batch_size, verbose=1)
            y_pred = y_pred.argmax(axis=-1)
            tf.logging.info(metrics.classification_report(valid_y, y_pred))
            model_pn.save_weights(model_save_path_pn)
        indices, segments = [], []
        for item in train_data_u_sample:
            query = item[1]
            input_ids, segment_ids = process_query(query)
            indices.append(input_ids)
            segments.append(segment_ids)
        indices = np.array(indices)
        segments = np.array(segments)
        prediction_un = model_pn.predict([indices, segments], batch_size=batch_size, verbose=1)
        prediction_un = prediction_un.argmax(axis=-1)
        remove_index = set()
        train_data_u_new = []
        for j, item in enumerate(train_data_u_sample):
            if prediction_un[j] == 0 or prediction_un[j] == 3:
                id = item[0]
                query = item[1]
                title = item[2]
                train_data.append((id, query, title))
                remove_index.add(id)
            else:
                id = item[0]
                title = item[2]
                train_data_u_new.append((id, title))
        tf.logging.info(remove_index)
        with open(save_data_file_path, mode='w') as fw:
            for item in train_data:
                # if index in remove_index:
                try:
                    index = item[0]
                    query = item[1]
                    title = item[2]
                    fw.write('{}\t{}\t{}\n'.format(index, query, title))
                except Exception as e:
                    tf.logging.info(e)
        data_un = train_data_u_new

    train_generator = data_generator(train_data, batch_size)
    model.load_weights(model_save_path)
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=max_epochs_training,
        # callbacks=[evaluator]
    )
    titles = []
    for id, query, title in tqdm(valid_data):
        titles.append(title)
    pred_queries, pred_scores = autoquery.generate_batch(titles)
    rouge_1, rouge_2, rouge_l, total = 0, 0, 0, 0
    for k, (id, query, title) in tqdm(enumerate(valid_data)):
        query = ' '.join(query).lower()
        pred_query = pred_queries[k]
        if len(pred_query) < 5 or len(pred_query) > 13:
            total += 1
            continue
        pred_query = ' '.join(pred_query).lower()
        if pred_query.strip():
            scores = rouge.get_scores(hyps=pred_query, refs=query)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            total += 1
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    tf.logging.info('rouge: {}\t{}\t{}'.format(rouge_1, rouge_2, rouge_l))

model.save_weights(model_save_path)
