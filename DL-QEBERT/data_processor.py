import tokenization
import tensorflow as tf
from tqdm import tqdm
import collections
import six
import numpy as np

np.random.seed(777)
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 prediction=None,
                 start_logit=None,
                 end_logit=None,
                 is_trainable=None,
                 input_ids_pn=None,
                 input_mask_pn=None,
                 segment_ids_pn=None,
                 doc_tokens=None,
                 reward=None,
                 answer=None
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.prediction = prediction
        self.start_logit = start_logit
        self.end_logit = end_logit
        self.is_trainable = is_trainable
        self.input_ids_pn = input_ids_pn
        self.segment_ids_pn = segment_ids_pn
        self.input_mask_pn = input_mask_pn
        self.doc_tokens = doc_tokens
        self.reward = reward
        self.answer = answer

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

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

def get_seen_data(data_file, vocab_file, is_training=False, do_lower_case=True, max_seq_length=128, doc_stride=512, dual_sentence=False):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    data = []
    with open(data_file, mode='r') as fd:
        for line in tqdm(fd.readlines()):
            if line.startswith('query'): continue
            item = line.strip().split('\t')
            # if len(item) != 2: continue
            paragraph_text = item[1]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                # if is_whitespace(c):
                #     prev_is_whitespace = True
                # else:
                #     if prev_is_whitespace:
                #         doc_tokens.append(c)
                #     else:
                #         doc_tokens[-1] += c
                #     prev_is_whitespace = False
                if not is_whitespace(c):
                    doc_tokens.append(c)
                char_to_word_offset.append(len(doc_tokens) - 1)
            qas_id = 's' + str(len(data))
            answer = item[0]
            if is_training:
                if answer not in paragraph_text: continue
                orig_answer_text = answer
                answer_offset = paragraph_text.index(answer)
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length - 1]
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = "".join(
                    doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = "".join(
                    tokenization.whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    print("Could not find answer: '%s' vs. '%s'",
                                       actual_text, cleaned_answer_text)
                    continue

            unique_id = qas_id

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []

            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            if is_training:
                tok_start_position = None
                tok_end_position = None

                tok_start_position = orig_to_tok_index[start_position]
                if end_position < len(doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)


                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

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

                start_position = None
                end_position = None
                if is_training:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = 1
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset


                if len(data) < 5:
                    tf.logging.info("*** Example ***")
                    tf.logging.info("unique_id: %s" % (unique_id))
                    tf.logging.info("example_index: %s" % (len(data)))
                    tf.logging.info("doc_span_index: %s" % (doc_span_index))
                    tf.logging.info("tokens: %s" % " ".join(
                        [tokenization.printable_text(x) for x in tokens]))
                    tf.logging.info("token_to_orig_map: %s" % " ".join(
                        ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                    tf.logging.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                    ]))
                    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    tf.logging.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    tf.logging.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

                    if is_training:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        tf.logging.info("start_position: %d" % (start_position))
                        tf.logging.info("end_position: %d" % (end_position))
                        tf.logging.info(
                            "answer: %s" % (tokenization.printable_text(answer_text)))
                query = None
                if not is_training:
                    query = answer.split('||')
                feature = InputFeatures(
                    unique_id=unique_id,
                    example_index=len(data),
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    doc_tokens=doc_tokens,
                    reward=1.,
                    answer=query
                )

            data.append(feature)
            # if len(data) > 200: break
    # if len(data) > 20000: data = np.random.choice(data, 10000, replace=False).tolist()
    return data

def get_unseen_data(data_file, vocab_file, do_lower_case=True, max_seq_length=128, doc_stride=512, dual_sentence=False):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    data = []
    with open(data_file, mode='r') as fd:
        for line in tqdm(fd.readlines()):
            if line.startswith('title'): continue
            if line.startswith('query'): continue
            item = line.strip().split('\t')
            paragraph_text = item[1]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                # if is_whitespace(c):
                #     prev_is_whitespace = True
                # else:
                #     if prev_is_whitespace:
                #         doc_tokens.append(c)
                #     else:
                #         doc_tokens[-1] += c
                #     prev_is_whitespace = False
                if not is_whitespace(c):
                    doc_tokens.append(c)
                char_to_word_offset.append(len(doc_tokens) - 1)

            qas_id = 'u' + str(len(data))

            unique_id = qas_id

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)


            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

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


                if len(data) < 5:
                    tf.logging.info("*** Example ***")
                    tf.logging.info("unique_id: %s" % (unique_id))
                    tf.logging.info("example_index: %s" % (len(data)))
                    tf.logging.info("doc_span_index: %s" % (doc_span_index))
                    tf.logging.info("tokens: %s" % " ".join(
                        [tokenization.printable_text(x) for x in tokens]))
                    tf.logging.info("token_to_orig_map: %s" % " ".join(
                        ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                    tf.logging.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                    ]))
                    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    tf.logging.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    tf.logging.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

                feature = InputFeatures(
                    unique_id=unique_id,
                    example_index=len(data),
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    doc_tokens=doc_tokens)

            data.append(feature)
            # if len(data) > 200: break
    # data = np.random.choice(data, 100000, replace=False).tolist()
    return data

def get_pretrain_data(data_file, vocab_file, do_lower_case=True, max_seq_length=128, dual_sentence=False):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    data = []
    with open(data_file, mode='r') as fd:
        for line in tqdm(fd.readlines()):
            if line.startswith('query'): continue
            item = line.strip().split('\t')
            query = item[0]
            # title = item[1]
            state = item[1]
            label = int(state)
            tokens_a = tokenizer.tokenize(query)
            tokens_b = None
            # if dual_sentence:
            #     tokens_b = tokenizer.tokenize(title)

            if dual_sentence:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

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

            if len(data) < 5:
                tf.logging.info("*** Example ***")
                tf.logging.info("example_index: %s" % (len(data)))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))

                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                tf.logging.info("label: %s" % label)

            data.append({'index': 's' + str(len(data)), 'label': label, 'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids})
            # if len(data) > 100: break

    return data
