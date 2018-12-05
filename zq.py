import re
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import pandas as pd
import argparse
import collections
import multiprocessing
import logging
import random

from general_utils import flatten_json, free_text_to_span, normalize_text, build_embedding, load_glove_vocab, pre_proc, get_context_span, find_answer_span, feature_gen, token2id

parser = argparse.ArgumentParser(
    description='Preprocessing train + dev files, about 15 minutes to run on Servers.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words.'
                         'Otherwise consider question words first.')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--no_match', action='store_true',
                    help='do not extract the three exact matching features.')
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, embedding init, etc.')

args = parser.parse_args()
trn_file = 'CoQA/sampleTrain.json'
dev_file = 'CoQA/sampleDev.json'
wv_file = args.wv_file
wv_dim = args.wv_dim


random.seed(args.seed)
np.random.seed(args.seed)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing... (using {} threads)'.format(args.threads))

# glove_vocab = load_glove_vocab(wv_file, wv_dim) # return a "set" of vocabulary
# log.info('glove loaded.')

#===============================================================
#=================== Work on training data =====================
#===============================================================

def proc_train(ith, article):
    rows = []
    context = article['story']

    for j, (question, answers) in enumerate(zip(article['questions'], article['answers'])):
        gold_answer = answers['input_text'] #str
        span_answer = answers['span_text'] #str

        answer, char_i, char_j = free_text_to_span(gold_answer, span_answer)
        answer_choice = 0 if answer == '__NA__' else\
                        1 if answer == '__YES__' else\
                        2 if answer == '__NO__' else\
                        3 # Not a yes/no question

        if answer_choice == 3:
            answer_start = answers['span_start'] + char_i
            answer_end = answers['span_start'] + char_j
        else:
            answer_start, answer_end = -1, -1

        rationale = answers['span_text']
        rationale_start = answers['span_start']
        rationale_end = answers['span_end']

        q_text = question['input_text']
        if j > 0:
            q_text = article['answers'][j-1]['input_text'] + " // " + q_text #考虑前一个的答案吗

        rows.append((ith, q_text, answer, answer_start, answer_end, rationale, rationale_start, rationale_end, answer_choice))
    return rows, context

train, train_context = flatten_json(trn_file, proc_train)
train = pd.DataFrame(train, columns=['context_idx', 'question', 'answer', 'answer_start', 'answer_end', 'rationale', 'rationale_start', 'rationale_end', 'answer_choice'])
log.info('train json data flattened.')
log.info("train = ")
print(len(train))
print(train)