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
import pickle
import os
from allennlp.modules.elmo import batch_to_ids
from general_utils import flatten_json, free_text_to_span, normalize_text, build_embedding, load_glove_vocab, pre_proc, get_context_span, find_answer_span, feature_gen, token2id

parser = argparse.ArgumentParser(
    description='Preprocessing train + dev files, about 15 minutes to run on Servers.'
)
parser.add_argument('--wv_file', default='/data1/zhengquan/data/wordvecs/glove.840B.300d.txt',
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
nlp = spacy.load('en', disable=['parser'])

random.seed(args.seed)
np.random.seed(args.seed)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing... (using {} threads)'.format(args.threads))

glove_vocab = load_glove_vocab(wv_file, wv_dim) # return a "set" of vocabulary
log.info('glove loaded.')

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

print(train)
log.info("huangxiayudidit")
exit(90)

log.info('train json data flattened.')
# log.info("train = ")
# print(len(train))
# print(train)

trC_iter = (pre_proc(c) for c in train_context)
trQ_iter = (pre_proc(q) for q in train.question)
trC_docs = [doc for doc in nlp.pipe(trC_iter, batch_size=64, n_threads=args.threads)] #这一节的操作是为了分词吗, 除了分词也把每个单词的tag和ner种类都识别出来了
trQ_docs = [doc for doc in nlp.pipe(trQ_iter, batch_size=64, n_threads=args.threads)]


# tokens
trC_tokens = [[normalize_text(w.text) for w in doc] for doc in trC_docs]
trQ_tokens = [[normalize_text(w.text) for w in doc] for doc in trQ_docs]
trC_unnorm_tokens = [[w.text for w in doc] for doc in trC_docs]
log.info('All tokens for training are obtained.')

train_context_span = [get_context_span(a, b) for a, b in zip(train_context, trC_unnorm_tokens)]
#get_context_span()#各个context_token这个list中的token对应于context这个字符串中的，单词首尾的下标
#train_context_span 这个list中的每个元素是[[],[],[]]一句话作为list，这句话里面的每个单词再是一个二元素的list

ans_st_token_ls, ans_end_token_ls = [], []
for ans_st, ans_end, idx in zip(train.answer_start, train.answer_end, train.context_idx):#ans_st,ans_end也是char base的
    ans_st_token, ans_end_token = find_answer_span(train_context_span[idx], ans_st, ans_end) #这个就是我关心的关键，如何将char_base的index，转化为token_base的，有可能返回-1，-1哦
    #这里的逻辑很简单，真正神奇的事情已经发生过了，就是白天看的
    #在free_text_to_span()函数中两个函数的配合使用
    #其一len_preserved_normalize_answer()在去掉每一个标点或者不想要的字符的时候，用空白符填补，确保返回的字符串的长度不变，并且里面的每个单词所处的位置不变
    #其二split_with_span()在这个里面使用了re.finditer()函数，返回了每个匹配的start和end
    #zip(*[(m.group(0), (m.start(), m.end()-1)) for m in re.finditer(r'\S+', s)])#\S匹配非空白字符
    #返回的东西用zip(*返回的东西)裹一下，这样外面就得到了list[str],list[span()]
    ans_st_token_ls.append(ans_st_token)
    ans_end_token_ls.append(ans_end_token)

ration_st_token_ls, ration_end_token_ls = [], []
for ration_st, ration_end, idx in zip(train.rationale_start, train.rationale_end, train.context_idx):#与得到答案的首尾token base的首尾index 的方法一样，得到rationale的token base的首尾span
    ration_st_token, ration_end_token = find_answer_span(train_context_span[idx], ration_st, ration_end)
    ration_st_token_ls.append(ration_st_token)
    ration_end_token_ls.append(ration_end_token)

train['answer_start_token'], train['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
train['rationale_start_token'], train['rationale_end_token'] = ration_st_token_ls, ration_end_token_ls

initial_len = len(train)

train.dropna(inplace=True) # modify self DataFrame
#dropna是什么操作啊，删除缺失数据
log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(train), initial_len))
log.info('answer span for training is generated.')

# features
trC_tags, trC_ents, trC_features = feature_gen(trC_docs, train.context_idx, trQ_docs, args.no_match)
#目前查看feature_gen()函数中，产生的feature都是什么

log.info('features for training is generated: {}, {}, {}'.format(len(trC_tags), len(trC_ents), len(trC_features)))
# log.info(str(len(trC_tags))) #样例的数目 3
# log.info(str(len(trC_ents))) #样例的数目 3
# log.info(str(len(trC_features))) #问题的数目 38
# exit(67)

# glove_vocab = load_glove_vocab(wv_file, wv_dim) # return a "set" of vocabulary
# if not os.path.exists('/data1/zhengquan/data/temp'):
#     os.mkdir('/data1/zhengquan/data/temp')
# pickle.dump(glove_vocab,open('/data1/zhengquan/data/temp/glove_vocab.pkl',"wb"))
glove_vocab = pickle.load(open("/data1/zhengquan/data/temp/glove_vocab.pkl","rb"))
log.info('glove loaded.')
def build_train_vocab(questions, contexts): # vocabulary will also be sorted accordingly
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    else:
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in glove_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in glove_vocab],
                        key=counter.get, reverse=True)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab {1}/{0} OOV {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    vocab.insert(2, "<S>")
    vocab.insert(3, "</S>")
    return vocab

# vocab
tr_vocab = build_train_vocab(trQ_tokens, trC_tokens)#不在glove词典中的词就作为unk看待了 ,#tr_vocab实际上是一个list


trC_ids = token2id(trC_tokens, tr_vocab, unk_id=1)
trQ_ids = token2id(trQ_tokens, tr_vocab, unk_id=1) #转化为id表示了
trQ_tokens = [["<S>"] + doc + ["</S>"] for doc in trQ_tokens]
trQ_ids = [[2] + qsent + [3] for qsent in trQ_ids]
print("trQ_ids = ")
print(trQ_ids[:10])

# tags
a = list(nlp.tagger.labels)
# print("a = ",a)
# print(len(a))
vocab_tag = [''] + list(nlp.tagger.labels)
# 这些tag的意义还不知道，有50个tag，不可能是人工写规则，即便是使用，也是在后端用word embedding的方式，弄一个tag embedding
# print("vocab_tag = ",vocab_tag)
# print(len(vocab_tag))
# exit(789)
trC_tag_ids = token2id(trC_tags, vocab_tag) #Context中的每个单词对应到id之后，没有单词是有tag的，这个tag也对应到tag_id上
# entities
vocab_ent = list(set([ent for sent in trC_ents for ent in sent])) #在context中所出现的所有的命名实体类型全部归入到list中
# print("vocab_ent = ",vocab_ent)
# exit(789)

trC_ent_ids = token2id(trC_ents, vocab_ent, unk_id=0) #这里怎么还会需要unk_id

log.info('Found {} POS tags.'.format(len(vocab_tag)))
log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
log.info('vocabulary for training is built.')
# exit(89)
# 接下来该看怎么获得向量的了


tr_embedding = build_embedding(wv_file, tr_vocab, wv_dim) #tr_vocab实际上是一个list ，返回的是一个numpy的array
log.info('got embedding matrix for training.')

meta = {
    'vocab': tr_vocab,
    'embedding': tr_embedding.tolist()
}
with open('CoQA/train_meta.msgpack', 'wb') as f:
    msgpack.dump(meta, f)

prev_CID, first_question = -1, []
for i, CID in enumerate(train.context_idx):
    if not (CID == prev_CID):
        first_question.append(i)
    prev_CID = CID


result = {
    'question_ids': trQ_ids,
    'context_ids': trC_ids,
    'context_features': trC_features, # exact match, tf
    'context_tags': trC_tag_ids, # POS tagging
    'context_ents': trC_ent_ids, # Entity recognition
    'context': train_context, #list(str)
    'context_span': train_context_span, # train_context_span 这个list中的每个元素是[[],[],[]]一句话作为list，这句话里面的每个单词再是一个二元素的list
    '1st_question': first_question, # 更短的list，长度只有数据条目数
    'question_CID': train.context_idx.tolist(),
    'question': train.question.tolist(),
    'answer': train.answer.tolist(),
    'answer_start': train.answer_start_token.tolist(),
    'answer_end': train.answer_end_token.tolist(),
    'rationale_start': train.rationale_start_token.tolist(),
    'rationale_end': train.rationale_end_token.tolist(),
    'answer_choice': train.answer_choice.tolist(),
    'context_tokenized': trC_tokens,
    'question_tokenized': trQ_tokens
}
with open('CoQA/train_data.msgpack', 'wb') as f:
    msgpack.dump(result, f)

log.info('saved training to disk.')

#==========================================================
#=================== Work on dev data =====================
#==========================================================

def proc_dev(ith, article):
    rows = []
    context = article['story']

    for j, (question, answers) in enumerate(zip(article['questions'], article['answers'])):
        gold_answer = answers['input_text']
        span_answer = answers['span_text']

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
            q_text = article['answers'][j-1]['input_text'] + " // " + q_text

        rows.append((ith, q_text, answer, answer_start, answer_end, rationale, rationale_start, rationale_end, answer_choice))
    return rows, context

dev, dev_context = flatten_json(dev_file, proc_dev)
dev = pd.DataFrame(dev, columns=['context_idx', 'question', 'answer', 'answer_start', 'answer_end', 'rationale', 'rationale_start', 'rationale_end', 'answer_choice'])
log.info('dev json data flattened.')

print(dev)

devC_iter = (pre_proc(c) for c in dev_context)
devQ_iter = (pre_proc(q) for q in dev.question)
devC_docs = [doc for doc in nlp.pipe(
    devC_iter, batch_size=64, n_threads=args.threads)]
devQ_docs = [doc for doc in nlp.pipe(
    devQ_iter, batch_size=64, n_threads=args.threads)]

# tokens
devC_tokens = [[normalize_text(w.text) for w in doc] for doc in devC_docs]
devQ_tokens = [[normalize_text(w.text) for w in doc] for doc in devQ_docs]
devC_unnorm_tokens = [[w.text for w in doc] for doc in devC_docs]
log.info('All tokens for dev are obtained.')

dev_context_span = [get_context_span(a, b) for a, b in zip(dev_context, devC_unnorm_tokens)]
log.info('context span for dev is generated.')

ans_st_token_ls, ans_end_token_ls = [], []
for ans_st, ans_end, idx in zip(dev.answer_start, dev.answer_end, dev.context_idx):
    ans_st_token, ans_end_token = find_answer_span(dev_context_span[idx], ans_st, ans_end)
    ans_st_token_ls.append(ans_st_token)
    ans_end_token_ls.append(ans_end_token)

ration_st_token_ls, ration_end_token_ls = [], []
for ration_st, ration_end, idx in zip(dev.rationale_start, dev.rationale_end, dev.context_idx):
    ration_st_token, ration_end_token = find_answer_span(dev_context_span[idx], ration_st, ration_end)
    ration_st_token_ls.append(ration_st_token)
    ration_end_token_ls.append(ration_end_token)

dev['answer_start_token'], dev['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
dev['rationale_start_token'], dev['rationale_end_token'] = ration_st_token_ls, ration_end_token_ls

initial_len = len(dev)
dev.dropna(inplace=True) # modify self DataFrame
log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(dev), initial_len))
log.info('answer span for dev is generated.')

# features
devC_tags, devC_ents, devC_features = feature_gen(devC_docs, dev.context_idx, devQ_docs, args.no_match)
log.info('features for dev is generated: {}, {}, {}'.format(len(devC_tags), len(devC_ents), len(devC_features)))

def build_dev_vocab(questions, contexts): # most vocabulary comes from tr_vocab
    existing_vocab = set(tr_vocab)
    new_vocab = list(set([w for doc in questions + contexts for w in doc if w not in existing_vocab and w in glove_vocab]))
    vocab = tr_vocab + new_vocab
    log.info('train vocab {0}, total vocab {1}'.format(len(tr_vocab), len(vocab)))
    return vocab

# vocab
dev_vocab = build_dev_vocab(devQ_tokens, devC_tokens) # tr_vocab is a subset of dev_vocab
devC_ids = token2id(devC_tokens, dev_vocab, unk_id=1)
devQ_ids = token2id(devQ_tokens, dev_vocab, unk_id=1)
devQ_tokens = [["<S>"] + doc + ["</S>"] for doc in devQ_tokens]
devQ_ids = [[2] + qsent + [3] for qsent in devQ_ids]
print(devQ_ids[:10])
# tags
devC_tag_ids = token2id(devC_tags, vocab_tag) # vocab_tag same as training
# entities
devC_ent_ids = token2id(devC_ents, vocab_ent, unk_id=0) # vocab_ent same as training
log.info('vocabulary for dev is built.')

dev_embedding = build_embedding(wv_file, dev_vocab, wv_dim)
# tr_embedding is a submatrix of dev_embedding
log.info('got embedding matrix for dev.')

# don't store row name in csv
#dev.to_csv('QuAC_data/dev.csv', index=False, encoding='utf8')

meta = {
    'vocab': dev_vocab,
    'embedding': dev_embedding.tolist()
}
with open('CoQA/dev_meta.msgpack', 'wb') as f:
    msgpack.dump(meta, f)

prev_CID, first_question = -1, []
for i, CID in enumerate(dev.context_idx):
    if not (CID == prev_CID):
        first_question.append(i)
    prev_CID = CID

result = {
    'question_ids': devQ_ids,
    'context_ids': devC_ids,
    'context_features': devC_features, # exact match, tf
    'context_tags': devC_tag_ids, # POS tagging
    'context_ents': devC_ent_ids, # Entity recognition
    'context': dev_context,
    'context_span': dev_context_span,
    '1st_question': first_question,
    'question_CID': dev.context_idx.tolist(),
    'question': dev.question.tolist(),
    'answer': dev.answer.tolist(),
    'answer_start': dev.answer_start_token.tolist(),
    'answer_end': dev.answer_end_token.tolist(),
    'rationale_start': dev.rationale_start_token.tolist(),
    'rationale_end': dev.rationale_end_token.tolist(),
    'answer_choice': dev.answer_choice.tolist(),
    'context_tokenized': devC_tokens,
    'question_tokenized': devQ_tokens
}
with open('CoQA/dev_data.msgpack', 'wb') as f:
    msgpack.dump(result, f)

log.info('saved dev to disk.')
