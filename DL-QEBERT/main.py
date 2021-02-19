import os
import json
import tokenization
import numpy as np
import tensorflow as tf
import modeling
import collections
from data_processor import get_seen_data, get_unseen_data, get_pretrain_data
from Model import BertModel, PolicyNetwork
from sklearn.utils import shuffle
from sklearn import metrics
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

tf.logging.set_verbosity(tf.logging.INFO)
max_seq_length = 64
batch_size = 128
max_epochs = 3
max_epochs_training = 5
warmup_proportion = 0.1
num_classes = 3
dropout = 0.1
learning_rate = 1e-4
radio = 0.1
episodes = 3
max_answer_length = 13
min_answer_length = 5

pretrained_path = '../chinese_L-12_H-768_A-12/'
pretrained_path_pn = '../output/'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
checkpoint_path_pn = os.path.join(pretrained_path_pn, 'model.ckpt-143874')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

data_path = '../dataset'
data_file_train_seen = os.path.join(data_path, 'train.txt')
data_file_valid_seen = os.path.join(data_path, 'valid.txt')
data_file_train_unseen = os.path.join(data_path, 'unlabeled_titles.txt')
# data_file_valid_unseen = os.path.join(data_path, 'test.txt')
train_file_for_pretrain_policy_network = os.path.join(data_path, 'train_pn.txt')
valid_file_for_pretrain_policy_network = os.path.join(data_path, 'valid_pn_new.txt')
remove_file_path = os.path.join(data_path, 'train_data_sub.txt')

model_path = '../model'
model_save_path_se = os.path.join(model_path, 'model_se.ckpt')
model_save_path_pn = os.path.join(model_path, 'model_pn.ckpt')

train_data_seen = get_seen_data(data_file_train_seen, vocab_path, is_training=True, max_seq_length=max_seq_length)
tf.logging.info('Input data shape: {}'.format(len(train_data_seen)))
valid_data_seen = get_seen_data(data_file_valid_seen, vocab_path, max_seq_length=max_seq_length)
tf.logging.info('Input data shape: {}'.format(len(valid_data_seen)))

train_data_unseen = get_unseen_data(data_file_train_unseen, vocab_path, max_seq_length=max_seq_length)
tf.logging.info('Input data shape: {}'.format(len(train_data_unseen)))

valid_data_pretrain = get_pretrain_data(valid_file_for_pretrain_policy_network, vocab_path, max_seq_length=max_answer_length + 2)
tf.logging.info('Input data shape: {}'.format(len(valid_data_seen)))

train_data_s = train_data_seen.copy()
valid_data_s = valid_data_seen
train_data_u = train_data_unseen.copy()

g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    global_step = tf.train.get_or_create_global_step()
    train_examples = len(train_data_s)
    num_expamles_valid_pretrain = len(valid_data_pretrain)
    num_train_steps = int(
        train_examples / batch_size * max_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    train_graph_pn = PolicyNetwork(num_classes,
                            is_training=True,
                            bert_config=config_path,
                            dropout=dropout,
                            learning_rate=learning_rate * 0.1,
                            num_train_steps=num_train_steps,
                            num_warmup_steps=num_warmup_steps)
    valid_graph_pn = PolicyNetwork(num_classes, bert_config=config_path, is_training=False)
    tvars = tf.trainable_variables()
    # initialized_variable_names = {}
    scaffold_fn = None
    init_checkpoint = checkpoint_path_pn
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                               init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    initializer_pn = tf.global_variables_initializer()
    vars_ = {}
    for var in tf.global_variables():
        if "Adam" in var.name: continue
        if 'power' in var.name: continue
        # if var.name.startswith("bert"): continue
        vars_[var.name.split(":")[0]] = var
    saver_pn = tf.train.Saver(vars_)
    sess_pn = tf.Session(graph=g1)
    sess_pn.run(initializer_pn)
    saver_pn.restore(sess_pn, init_checkpoint)
    # for epoch in range(max_epochs_training):
    #     data_s = shuffle(train_data_pretrain)
    #     num_expamles_pretrain = len(train_data_pretrain)
    #     total_loss = 0
    #     for start in tqdm(range(0, num_expamles_pretrain, batch_size)):
    #         end = start + batch_size
    #         input_ids_batch, segment_ids_batch, input_mask_batch, label_batch = [], [], [], []
    #         for item in train_data_pretrain[start: end]:
    #             input_ids_batch.append(item['input_ids'])
    #             segment_ids_batch.append(item['segment_ids'])
    #             input_mask_batch.append(item['input_mask'])
    #             label_batch.append(item['label'])
    #         batch = {'input_ids': np.array(input_ids_batch),
    #                  'segment_ids': np.array(segment_ids_batch),
    #                  'input_mask': np.array(input_mask_batch),
    #                  'label': np.array(label_batch),
    #                  'reward': 1
    #                  }
    #         feed_dict = train_graph_pn.create_feed_dict(batch, is_training=True)
    #         _, loss_value = sess_pn.run(
    #             [train_graph_pn.train_op, train_graph_pn.loss],
    #             feed_dict=feed_dict)
    #         total_loss += loss_value
    #     tf.logging.info('Epoch %d: loss = %.4f' % (epoch, total_loss / (int(num_expamles_pretrain / batch_size) + 1)))
    # tf.logging.info('train model done')
    # ########################### Evaluation on valid data ############################
    num_expamles_valid_pretrain = len(valid_data_pretrain)
    valid_pred = []
    valid_true = []
    for start in tqdm(range(0, num_expamles_valid_pretrain, batch_size)):
        end = start + batch_size
        input_ids_valid_batch, segment_ids_valid_batch, input_mask_valid_batch = [], [], []
        for item in valid_data_pretrain[start: end]:
            input_ids_valid_batch.append(item['input_ids'])
            segment_ids_valid_batch.append(item['segment_ids'])
            input_mask_valid_batch.append(item['input_mask'])
            valid_true.append(item['label'])
        batch = {'input_ids': np.array(input_ids_valid_batch),
                 'segment_ids': np.array(segment_ids_valid_batch),
                 'input_mask': np.array(input_mask_valid_batch)
                 }
        feed_dict = valid_graph_pn.create_feed_dict(batch, is_training=False)
        pred, prob = sess_pn.run([valid_graph_pn.predictions, valid_graph_pn.prob], feed_dict=feed_dict)
        valid_pred.extend(pred.tolist())
    tf.logging.info(metrics.classification_report(y_true=valid_true, y_pred=valid_pred))
    saver_pn.save(sess_pn, model_save_path_pn)

rouge = Rouge()
smooth = SmoothingFunction().method1

with g2.as_default():
    global_step = tf.train.get_or_create_global_step()
    train_examples = len(train_data_s)
    num_train_steps = int(
        train_examples / batch_size * max_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    train_graph = BertModel(num_classes,
                            is_training=True,
                            bert_config=config_path,
                            dropout=dropout,
                            learning_rate=learning_rate,
                            num_train_steps=num_train_steps,
                            num_warmup_steps=num_warmup_steps)
    valid_graph = BertModel(num_classes, bert_config=config_path, is_training=False)
    tvars = tf.trainable_variables()
    # initialized_variable_names = {}
    scaffold_fn = None
    init_checkpoint = checkpoint_path
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                               init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    initializer = tf.global_variables_initializer()
    vars_ = {}
    for var in tf.global_variables():
        # if "word_embedding" in var.name: continue
        # if var.name.startswith("bert"): continue
        vars_[var.name.split(":")[0]] = var
    saver = tf.train.Saver(vars_)
    sess = tf.Session(graph=g2)
    sess.run(initializer)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def calculate_score(y_true, y_pred):
    rouge_1_all = []
    rouge_2_all = []
    rouge_l_all = []
    bleu_1_all = []
    bleu_2_all = []
    bleu_3_all = []
    bleu_4_all = []
    em_all = []
    for i in range(len(y_pred)):
        if len(y_pred[i]) < 5 or len(y_pred[i]) > 13:
            rouge_1_all.append(0)
            rouge_2_all.append(0)
            rouge_l_all.append(0)
            bleu_1_all.append(0)
            bleu_2_all.append(0)
            bleu_3_all.append(0)
            bleu_4_all.append(0)
            em_all.append(0)
            continue
        pred_query = ' '.join(y_pred[i]).lower()
        queries_ = y_true[i]
        rouge_1_per = []
        rouge_2_per = []
        rouge_l_per = []
        bleu_1_per = []
        bleu_2_per = []
        bleu_3_per = []
        bleu_4_per = []
        em_per = []
        for query_ in queries_:
            query_ = ' '.join(query_).lower()
            if pred_query.strip():
                score = rouge.get_scores(hyps=pred_query, refs=query_)
                rouge_1 = score[0]['rouge-1']['f']
                rouge_2 = score[0]['rouge-2']['f']
                rouge_l = score[0]['rouge-l']['f']
                rouge_1_per.append(rouge_1)
                rouge_2_per.append(rouge_2)
                rouge_l_per.append(rouge_l)
                bleu_1 = sentence_bleu(references=[query_.split(' ')], hypothesis=pred_query.split(' '),
                                     weights=[1.], smoothing_function=smooth)
                bleu_2 = sentence_bleu(references=[query_.split(' ')], hypothesis=pred_query.split(' '),
                                       weights=[1 / 2, 1 / 2], smoothing_function=smooth)
                bleu_3 = sentence_bleu(references=[query_.split(' ')], hypothesis=pred_query.split(' '),
                                       weights=[1 / 3, 1 / 3, 1 / 3], smoothing_function=smooth)
                bleu_4 = sentence_bleu(references=[query_.split(' ')], hypothesis=pred_query.split(' '),
                                       weights=[1 / 4, 1 / 4, 1 / 4, 1 / 4], smoothing_function=smooth)
                bleu_1_per.append(bleu_1)
                bleu_2_per.append(bleu_2)
                bleu_3_per.append(bleu_3)
                bleu_4_per.append(bleu_4)
                em = 0
                if query_ == pred_query:
                    em = 1
                em_per.append(em)
        rouge_1_per = np.max(np.array(rouge_1_per))
        rouge_2_per = np.max(np.array(rouge_2_per))
        rouge_l_per = np.max(np.array(rouge_l_per))
        bleu_1_per = np.max(np.array(bleu_1_per))
        bleu_2_per = np.max(np.array(bleu_2_per))
        bleu_3_per = np.max(np.array(bleu_3_per))
        bleu_4_per = np.max(np.array(bleu_4_per))
        em_per = np.max(np.array(em_per))

        rouge_1_all.append(float(rouge_1_per))
        rouge_2_all.append(float(rouge_2_per))
        rouge_l_all.append(float(rouge_l_per))

        bleu_1_all.append(float(bleu_1_per))
        bleu_2_all.append(float(bleu_2_per))
        bleu_3_all.append(float(bleu_3_per))
        bleu_4_all.append(float(bleu_4_per))

        em_all.append(float(em_per))
    return [np.array(rouge_1_all).mean(), np.array(rouge_2_all).mean(),
            np.array(rouge_l_all).mean(), np.array(bleu_1_all).mean(),
            np.array(bleu_2_all).mean(), np.array(bleu_3_all).mean(),
            np.array(bleu_4_all).mean(), np.array(em_all).mean()]

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_nbest_result(item, n_best_size=10):
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
    prelim_predictions = []
    start_indexes = _get_best_indexes(item.start_prob, n_best_size)
    end_indexes = _get_best_indexes(item.end_prob, n_best_size)
    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            bl = False
            if start_index >= len(item.tokens):
                bl = True
            if end_index >= len(item.tokens):
                bl = True
            if start_index not in item.token_to_orig_map:
                bl = True
            if end_index not in item.token_to_orig_map:
                bl = True
            if not item.token_is_max_context.get(start_index, False):
                bl = True
            if end_index < start_index:
                bl = True
            length = end_index - start_index + 1
            if length > max_answer_length:
                bl = True
            if bl:
                start_index = 0
                end_index = 0
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=item.unique_id,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=item.start_prob[start_index],
                    end_logit=item.end_prob[end_index]))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)
    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_index", "end_index", "prob", "feature_index"])

    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = item.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = item.token_to_orig_map[pred.start_index]
            orig_doc_end = item.token_to_orig_map[pred.end_index]
            orig_tokens = item.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            # tok_text = " ".join(tok_text.split())
            # orig_text = " ".join(orig_tokens)

            final_text = ''.join(item.doc_tokens[orig_doc_start:(orig_doc_end + 1)])

        else:
            final_text = ""

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_index=pred.start_index,
                end_index=pred.end_index,
                prob=(pred.start_logit + pred.end_logit) / 2,
                feature_index=item.unique_id))
    return nbest

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_path, do_lower_case=True)

def process_text(text):
    tokens_a = tokenizer.tokenize(text)
    if len(tokens_a) > max_answer_length - 2:
        tokens_a = tokens_a[0:(max_answer_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_answer_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_answer_length
    assert len(input_mask) == max_answer_length
    assert len(segment_ids) == max_answer_length
    return input_ids, segment_ids, input_mask

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

config.gpu_options.allow_growth = True
for i in range(max_epochs):
    sess.run(initializer)
    if i > 0: saver.restore(sess, model_save_path_se)
    # training
    ########################### train model on seend data ############################
    # total_loss = 0
    for epoch in range(max_epochs_training):
        total_loss = 0
        data_s = shuffle(train_data_s)
        num_expamles = len(train_data_s)
        for start in tqdm(range(0, num_expamles, batch_size)):
            end = start + batch_size
            input_ids_batch, segment_ids_batch, input_mask_batch, \
            start_position_batch, end_position_batch, reward_batch = [], [], [], [], [], []
            for item in train_data_s[start: end]:
                input_ids_batch.append(item.input_ids)
                segment_ids_batch.append(item.segment_ids)
                input_mask_batch.append(item.input_mask)
                start_position_batch.append(item.start_position)
                end_position_batch.append(item.end_position)
                reward_batch.append(item.reward)
            batch = {'input_ids': np.array(input_ids_batch),
                     'segment_ids': np.array(segment_ids_batch),
                     'input_mask': np.array(input_mask_batch),
                     'start': np.array(start_position_batch),
                     'end': np.array(end_position_batch),
                     'reward': np.array(reward_batch)
                     }
            feed_dict = train_graph.create_feed_dict(batch, is_training=True)
            _, loss_value = sess.run([train_graph.train_op, train_graph.loss],
                feed_dict=feed_dict)
            total_loss += loss_value
        tf.logging.info('Epoch %d: loss = %.4f' % (epoch, total_loss / (int(num_expamles / batch_size) + 1)))
    tf.logging.info('train model done')
    ########################### Evaluation on valid data ############################
    tf.logging.info('start predict on valid data')
    num_expamles_valid_s = len(valid_data_s)
    valid_pred_start = []
    valid_pred_end = []
    valid_true_start = []
    valid_true_end = []
    for start in tqdm(range(0, num_expamles_valid_s, batch_size)):
        end = start + batch_size
        input_ids_valid_batch, segment_ids_valid_batch, input_mask_valid_batch = [], [], []
        for item in valid_data_s[start: end]:
            input_ids_valid_batch.append(item.input_ids)
            segment_ids_valid_batch.append(item.segment_ids)
            input_mask_valid_batch.append(item.input_mask)
            valid_true_start.append(item.start_position)
            valid_true_end.append(item.end_position)
        batch = {'input_ids': np.array(input_ids_valid_batch),
                 'segment_ids': np.array(segment_ids_valid_batch),
                 'input_mask': np.array(input_mask_valid_batch)
                 }
        feed_dict = valid_graph.create_feed_dict(batch, is_training=False)
        start_predictions, end_predictions = sess.run([valid_graph.start_predictions, valid_graph.end_predictions], feed_dict=feed_dict)
        valid_pred_start.extend(start_predictions.tolist())
        valid_pred_end.extend(end_predictions.tolist())
    y_true = []
    y_pred = []
    assert len(valid_pred_start) == len(valid_data_s)
    for j, item in enumerate(valid_data_s):
        try:
            orig_doc_start = item.token_to_orig_map[valid_pred_start[j]]
            orig_doc_end = item.token_to_orig_map[valid_pred_end[j]]
            pred_tokens = ''.join(item.doc_tokens[orig_doc_start:(orig_doc_end + 1)])
        except Exception as e:
            pred_tokens = ''
        y_pred.append(pred_tokens)
        y_true.append(item.answer)
    score = calculate_score(y_true, y_pred)
    tf.logging.info('Score for valid total: %s' % str(score))
    saver.save(sess, model_save_path_se)

    ########################### prediction on unseend data ############################
    tf.logging.info('start predicting on unseen data')
    num_expamles_unseen = len(train_data_u)
    train_data_u_sample = []
    for start in tqdm(range(0, num_expamles_unseen, batch_size)):
        end = start + batch_size
        input_ids_batch, segment_ids_batch, input_mask_batch = [], [], []
        for item in train_data_u[start: end]:
            input_ids_batch.append(item.input_ids)
            segment_ids_batch.append(item.segment_ids)
            input_mask_batch.append(item.input_mask)

        batch = {
            'input_ids': np.array(input_ids_batch),
            'segment_ids': np.array(segment_ids_batch),
            'input_mask': np.array(input_mask_batch)
        }
        feed_dict = valid_graph.create_feed_dict(batch, is_training=False)
        start_probs, end_probs = sess.run([valid_graph.start_probs, valid_graph.end_probs],
                                                      feed_dict=feed_dict)

        for k, item in enumerate(train_data_u[start: end]):
            start_index = start_probs[k].argmax()
            end_index = end_probs[k].argmax()
            if start_index >= len(item.tokens):
                continue
            if end_index >= len(item.tokens):
                continue
            if start_index not in item.token_to_orig_map:
                continue
            if end_index not in item.token_to_orig_map:
                continue
            if not item.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length < min_answer_length:
                continue
            if length > max_answer_length:
                continue
            item.start_prob = start_probs[k][start_index]
            item.end_prob = end_probs[k][end_index]
            item.start_position = start_index
            item.end_position = end_index
            item.prediction = item.input_ids[start_index: end_index + 1]
            train_data_u_sample.append(item)

            # if len(train_data_u_sample) < 5:
            #     tf.logging.info("input_ids in predicting on unseen data: %s" % " ".join([str(x) for x in item.input_ids]))
            #     tf.logging.info(
            #         "start index in predicting on unseen data: %s" % str(start_index))
            #     tf.logging.info(
            #         "end index in predicting on unseen data: %s" % str(end_index))


    tf.logging.info('train data u sample num:' + str(len(train_data_u_sample)))
    train_data_u_sample = sorted(train_data_u_sample, key=lambda item: (item.start_prob + item.end_prob) / 2, reverse=True)

    train_data_u_sample = train_data_u_sample[:int(len(train_data_u_sample) * radio)]

    tf.logging.info('start training policy network')
    for episode in range(episodes):
        data_u_sample = shuffle(train_data_u_sample)
        num_expamles_rand = len(train_data_u_sample)

        tf.logging.info('start selecting sample')
        scores = []
        predictions = []
        for start in tqdm(range(0, num_expamles_rand, batch_size)):
            end = start + batch_size
            input_ids_batch, segment_ids_batch, input_mask_batch = [], [], []
            for item in train_data_u_sample[start: end]:
                input_ids = [101]
                input_ids.extend(item.prediction)
                input_ids.append(102)
                segment_ids = [0] * len(input_ids)
                input_mask = [1] * len(input_ids)
                assert len(input_ids) == len(segment_ids)
                assert len(input_ids) == len(input_mask)
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                item.input_ids_pn = input_ids
                item.segment_ids_pn = segment_ids
                item.input_mask_pn = input_mask

                input_ids_batch.append(input_ids)
                segment_ids_batch.append(segment_ids)
                input_mask_batch.append(input_mask)
                # if len(input_ids_batch) < 5:
                #     tf.logging.info("input_ids in selecting: %s" % " ".join([str(x) for x in input_ids]))
                #     tf.logging.info(
                #         "input_mask in selecting: %s" % " ".join([str(x) for x in input_mask]))
                #     tf.logging.info(
                #         "segment_ids in selecting: %s" % " ".join([str(x) for x in segment_ids]))

            batch = {
                    'input_ids': np.array(input_ids_batch),
                    'segment_ids': np.array(segment_ids_batch),
                    'input_mask': np.array(input_mask_batch)
                    }
            feed_dict = valid_graph_pn.create_feed_dict(batch, is_training=False)
            vec, prob = sess_pn.run([valid_graph_pn.vec, valid_graph_pn.prob], feed_dict=feed_dict)
            prediction = prob.argmax(axis=-1)
            predictions.append(prediction)
            for k, item in enumerate(train_data_u_sample[start: end]):
                item.is_trainable = prediction[k]
                item.reward = prob[k][prediction[k]]

        tf.logging.info('start training sub sample')

        for start in tqdm(range(0, num_expamles_rand, batch_size)):
            end = start + batch_size
            sess.run(initializer)
            sample_input_ids_batch, sample_segment_ids_batch, \
                sample_input_mask_batch, sample_start_position_batch, \
                sample_end_position_batch, reward_batch = [], [], [], [], [], []
            for item in train_data_u_sample[start: end]:
                if item.is_trainable == 0:
                    sample_input_ids_batch.append(item.input_ids)
                    sample_segment_ids_batch.append(item.segment_ids)
                    sample_input_mask_batch.append(item.input_mask)
                    sample_start_position_batch.append(item.start_position)
                    sample_end_position_batch.append(item.end_position)
                    reward_batch.append(item.reward)
            if len(sample_input_ids_batch) == 0:
                scores.append(0)
                continue
            batch = {'input_ids': np.array(sample_input_ids_batch),
                     'segment_ids': np.array(sample_segment_ids_batch),
                     'input_mask': np.array(sample_input_mask_batch),
                     'start': np.array(sample_start_position_batch),
                     'end': np.array(sample_end_position_batch),
                     'reward': np.array(reward_batch)
                     }
            for j in range(max_epochs_training * 2):
                feed_dict = train_graph.create_feed_dict(batch, is_training=True)
                _, loss_value = sess.run([train_graph.train_op, train_graph.loss],
                    feed_dict=feed_dict)
            tf.logging.info('start evaluating')

            num_expamles_valid_s = len(valid_data_s)
            valid_pred_start = []
            valid_pred_end = []
            valid_true_start = []
            valid_true_end = []
            for start_ in range(0, num_expamles_valid_s, batch_size):
                end_ = start_ + batch_size
                input_ids_valid_batch, segment_ids_valid_batch, input_mask_valid_batch = [], [], []
                for item in valid_data_s[start_: end_]:
                    input_ids_valid_batch.append(item.input_ids)
                    segment_ids_valid_batch.append(item.segment_ids)
                    input_mask_valid_batch.append(item.input_mask)
                    valid_true_start.append(item.start_position)
                    valid_true_end.append(item.end_position)
                batch = {'input_ids': np.array(input_ids_valid_batch),
                         'segment_ids': np.array(segment_ids_valid_batch),
                         'input_mask': np.array(input_mask_valid_batch)
                         }
                feed_dict = valid_graph.create_feed_dict(batch, is_training=False)
                start_predictions, end_predictions = sess.run(
                    [valid_graph.start_predictions, valid_graph.end_predictions], feed_dict=feed_dict)
                valid_pred_start.extend(start_predictions.tolist())
                valid_pred_end.extend(end_predictions.tolist())

            y_true = []
            y_pred = []
            assert len(valid_pred_start) == len(valid_data_s)
            for j, item in enumerate(valid_data_s):
                try:
                    orig_doc_start = item.token_to_orig_map[valid_pred_start[j]]
                    orig_doc_end = item.token_to_orig_map[valid_pred_end[j]]
                    pred_tokens = ''.join(item.doc_tokens[orig_doc_start:(orig_doc_end + 1)])
                except Exception as e:
                    pred_tokens = ''
                y_pred.append(pred_tokens)
                y_true.append(item.answer)
            score = calculate_score(y_true, y_pred)
            tf.logging.info('Score for valid se sub: %s' % str(score))
            reward = (score[0] + score[1] + score[2]) / 3
            scores.append(reward)
        scores = np.array(scores)
        tf.logging.info('start training policy network')
        saver_pn.restore(sess_pn, model_save_path_pn)
        for k, start in enumerate(tqdm(range(0, num_expamles_rand, batch_size))):
            end = start + batch_size
            input_ids_batch, segment_ids_batch, input_mask_batch = [], [], []
            for item in train_data_u_sample[start: end]:
                input_ids_batch.append(item.input_ids_pn)
                segment_ids_batch.append(item.segment_ids_pn)
                input_mask_batch.append(item.input_mask_pn)
            batch = {'input_ids': np.array(input_ids_batch),
                     'segment_ids': np.array(segment_ids_batch),
                     'input_mask': np.array(input_mask_batch),
                     'label': np.array(predictions[k]),
                     'reward': (scores[k] - scores.mean()) / np.sqrt(scores.var())
                     }
            feed_dict = train_graph_pn.create_feed_dict(batch, is_training=True)
            _, prob = sess_pn.run([train_graph_pn.train_op, train_graph_pn.prob], feed_dict=feed_dict)
    saver_pn.save(sess_pn, model_save_path_pn)
    tf.logging.info('start selecting subsample for next training epoch')

    for start in tqdm(range(0, num_expamles_rand, batch_size)):
        end = start + batch_size
        input_ids_batch, segment_ids_batch, input_mask_batch = [], [], []
        for item in train_data_u_sample[start: end]:
            input_ids_batch.append(item.input_ids_pn)
            segment_ids_batch.append(item.segment_ids_pn)
            input_mask_batch.append(item.input_mask_pn)
        batch = {'input_ids': np.array(input_ids_batch),
                 'segment_ids': np.array(segment_ids_batch),
                 'input_mask': np.array(input_mask_batch)
                 }
        feed_dict = valid_graph_pn.create_feed_dict(batch, is_training=False)
        vec, prob = sess_pn.run([valid_graph_pn.vec, valid_graph_pn.prob], feed_dict=feed_dict)
        prediction = prob.argmax(axis=-1)

        for k, item in enumerate(train_data_u_sample[start: end]):
            item.is_trainable = prediction[k]
            item.reward = prob[k][prediction[k]]
    remove_index = set()
    for item in train_data_u_sample:
        if item.is_trainable == 0:
            train_data_s.append(item)
            index = item.unique_id
            remove_index.add(index)
    print(remove_index)
    train_data_u_new = []
    for item in train_data_u:
        if item.unique_id not in remove_index:
            train_data_u_new.append(item)

    with open(remove_file_path, mode='w') as fw:
        for item in train_data_s:
            index = item.unique_id
            # if index in remove_index:
            try:
                orig_doc_start = item.token_to_orig_map[item.start_position]
                orig_doc_end = item.token_to_orig_map[item.end_position]
                pred_tokens = ''.join(item.doc_tokens[orig_doc_start:(orig_doc_end + 1)])
                orig_tokens = ''.join(item.doc_tokens)
                fw.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(index, orig_doc_start, orig_doc_end, pred_tokens, orig_tokens, item.reward))
            except Exception as e:
                print(e)
    train_data_u = train_data_u_new

    ########################### Evaluate policy network ############################
    tf.logging.info('Evaluate policy network')
    valid_pred = []
    valid_true = []
    for start in tqdm(range(0, num_expamles_valid_pretrain, batch_size)):
        end = start + batch_size
        input_ids_valid_batch, segment_ids_valid_batch, input_mask_valid_batch = [], [], []
        for item in valid_data_pretrain[start: end]:
            input_ids_valid_batch.append(item['input_ids'])
            segment_ids_valid_batch.append(item['segment_ids'])
            input_mask_valid_batch.append(item['input_mask'])
            valid_true.append(item['label'])
        batch = {'input_ids': np.array(input_ids_valid_batch),
                 'segment_ids': np.array(segment_ids_valid_batch),
                 'input_mask': np.array(input_mask_valid_batch)
                 }
        feed_dict = valid_graph_pn.create_feed_dict(batch, is_training=False)
        pred, prob = sess_pn.run([valid_graph_pn.predictions, valid_graph_pn.prob], feed_dict=feed_dict)
        valid_pred.extend(pred.tolist())
    tf.logging.info(metrics.classification_report(y_true=valid_true, y_pred=valid_pred))

########################### Training on total data ############################
tf.logging.info('Training on total data')
with tf.Session(graph=g2) as sess:
    sess.run(initializer)
    saver.restore(sess, model_save_path_se)
    # total_loss = 0
    for epoch in range(max_epochs_training):
        total_loss = 0
        data_s = shuffle(train_data_s)
        num_expamles = len(train_data_s)
        for start in tqdm(range(0, num_expamles, batch_size)):
            end = start + batch_size
            input_ids_batch, segment_ids_batch, input_mask_batch, \
            start_position_batch, end_position_batch, reward_batch = [], [], [], [], [], []
            for item in train_data_s[start: end]:
                input_ids_batch.append(item.input_ids)
                segment_ids_batch.append(item.segment_ids)
                input_mask_batch.append(item.input_mask)
                start_position_batch.append(item.start_position)
                end_position_batch.append(item.end_position)
                reward_batch.append(item.reward)
            batch = {'input_ids': np.array(input_ids_batch),
                     'segment_ids': np.array(segment_ids_batch),
                     'input_mask': np.array(input_mask_batch),
                     'start': np.array(start_position_batch),
                     'end': np.array(end_position_batch),
                     'reward': np.array(reward_batch)
                     }
            feed_dict = train_graph.create_feed_dict(batch, is_training=True)
            _, loss_value = sess.run([train_graph.train_op, train_graph.loss],
                feed_dict=feed_dict)
            total_loss += loss_value
        tf.logging.info('Epoch %d: loss = %.4f' % (epoch, total_loss / (int(num_expamles / batch_size) + 1)))
    tf.logging.info('train model done')
    ########################### Evaluation on valid data ############################
    tf.logging.info('start predict on valid data')
    num_expamles_valid_s = len(valid_data_s)
    valid_pred_start = []
    valid_pred_end = []
    valid_true_start = []
    valid_true_end = []
    for start in tqdm(range(0, num_expamles_valid_s, batch_size)):
        end = start + batch_size
        input_ids_valid_batch, segment_ids_valid_batch, input_mask_valid_batch = [], [], []
        for item in valid_data_s[start: end]:
            input_ids_valid_batch.append(item.input_ids)
            segment_ids_valid_batch.append(item.segment_ids)
            input_mask_valid_batch.append(item.input_mask)
            valid_true_start.append(item.start_position)
            valid_true_end.append(item.end_position)
        batch = {'input_ids': np.array(input_ids_valid_batch),
                 'segment_ids': np.array(segment_ids_valid_batch),
                 'input_mask': np.array(input_mask_valid_batch)
                 }
        feed_dict = valid_graph.create_feed_dict(batch, is_training=False)
        start_predictions, end_predictions = sess.run([valid_graph.start_predictions, valid_graph.end_predictions], feed_dict=feed_dict)
        valid_pred_start.extend(start_predictions.tolist())
        valid_pred_end.extend(end_predictions.tolist())
    y_true = []
    y_pred = []
    assert len(valid_pred_start) == len(valid_data_s)
    for j, item in enumerate(valid_data_s):
        try:
            orig_doc_start = item.token_to_orig_map[valid_pred_start[j]]
            orig_doc_end = item.token_to_orig_map[valid_pred_end[j]]
            pred_tokens = ''.join(item.doc_tokens[orig_doc_start:(orig_doc_end + 1)])
        except Exception as e:
            pred_tokens = ''
        y_pred.append(pred_tokens)
        y_true.append(item.answer)
    score = calculate_score(y_true, y_pred)

    tf.logging.info('Score for valid total: %s' % str(score))
    saver.save(sess, model_save_path_se)
