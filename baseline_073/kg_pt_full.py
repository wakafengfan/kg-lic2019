import collections
import json
import logging
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
from random import choice

import gensim
import numpy as np
import pyhanlp
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertAdam
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from tqdm import tqdm

from configuration.config import data_dir, bert_vocab_path, bert_data_path, bert_model_path

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

hidden_size = 768
batch_size = 64
epoch_num = 10

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


train_data = json.load(open(data_dir + '/train_data_me_2.json'))
dev_data = json.load(open(data_dir + '/dev_data_me_2.json'))
info_ = pickle.load((Path(data_dir)/'info_.pkl').open('rb'))
predicates = info_['predicates']
sp2o = info_['sp2o']
spo_total = info_['spo_total']
s_ac = pickle.load((Path(data_dir)/'s_ac.json').open())
o_ac = pickle.load((Path(data_dir)/'o_ac.json').open())

id2predicate, predicate2id = json.load(open(data_dir + '/all_50_schemas_me.json'))
id2predicate = {int(i): j for i, j in id2predicate.items()}

id2char, char2id = json.load(open(data_dir + '/all_chars_me.json'))
num_classes = len(id2predicate)




def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


bert_vocab = load_vocab(bert_vocab_path)

wv_model = gensim.models.KeyedVectors.load_word2vec_format(Path(data_dir)/'tencent_embed_for_el2019')
word2vec = wv_model.wv.syn0
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1,word_size)), np.zeros((1,word_size)),word_size])
id2word = {i+2:j for i,j in wv_model.wv.index2word}
word2id = {j:i for i, j in id2word.items()}
def tokenize(text_in):
    return [i.word for i in pyhanlp.HanLP.segment(text_in)]
def seq2vec(token_ids):
    V = []
    for s in token_ids:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return


def seq_padding(X):
    ML = max(map(len, X))
    return [x + [0] * (ML - len(x)) for x in X]

def random_generate(d, spo_list_key):
    r = np.random.random()
    if r > 0.5:
        return d
    else:
        k = np.random.randint(len(d[spo_list_key]))
        spi = d[spo_list_key][k]
        k = np.random.randint(len(predicates[spi[1]]))
        spo = predicates[spi[1]][k]
        F = lambda s: s.replace(spi[0], spo[0]).replace(spi[2], spo[2])
        text = F(d['text'])
        spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d[spo_list_key]]
        return {'text': text, spo_list_key: spo_list}

def spo_searcher(text_in, text_idx=None):
    R = set()
    for s in s_ac.iter(text_in):
        for o in o_ac.iter(text_in):
            if (s[1], o[1]) in sp2o:
                for p in sp2o[(s[1],o[1])]:
                    if text_idx is None:
                        R.add((s[1],p,o[1]))
                    elif spo_total[(s[1],p,o[1])] - {text_idx}:
                        R.add((s[1],p,o[1]))
    return list(R)



class data_generator:
    def __init__(self, data):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        # while True:
        idxs = list(range(len(self.data)))
        np.random.shuffle(idxs)
        T, S1, S2, K1, K2, O1, O2, TM, TS, TT = [], [], [], [], [], [], [], [], [],[]
        for i in idxs:
            d = self.data[i]
            # text = d['text']
            # text = re.sub(r'\s+', '', text)
            text_tokens = d['text_words']
            text = ''.join(text_tokens)
            items = {}
            for sp in d['spo_list']:
                subjectid = text.find(sp[0])
                objectid = text.find(sp[2])
                if subjectid != -1 and objectid != -1:
                    key = (subjectid, subjectid + len(sp[0]))
                    if key not in items:
                        items[key] = []
                    items[key].append((objectid,
                                       objectid + len(sp[2]),
                                       predicate2id[sp[1]]))
            if items:
                text_ids = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text]
                text_mask = [1] * len(text_ids)
                T.append(text_ids)
                TM.append(text_mask)

                text_token_ids = [word2id.get(w, 1) for w in text_tokens]
                TT.append(text_token_ids)

                s1, s2 = [0] * len(text), [0] * len(text)
                for j in items:
                    s1[j[0]] = 1
                    s2[j[1] - 1] = 1
                k1, k2 = choice(list(items.keys()))
                o1, o2 = [0] * len(text), [0] * len(text)  # 0是unk类（共49+1个类）
                for j in items[(k1, k2)]:
                    o1[j[0]] = j[2]
                    o2[j[1] - 1] = j[2]

                S1.append(s1)
                S2.append(s2)
                K1.append(k1)
                K2.append(k2-1)
                O1.append(o1)
                O2.append(o2)
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = torch.tensor(seq_padding(T), dtype=torch.long)
                    TM = torch.tensor(seq_padding(TM), dtype=torch.long)
                    TS = torch.zeros(*T.size(), dtype=torch.long)
                    TT = torch.tensor(seq2vec(TT), dtype=torch.long)

                    S1 = torch.tensor(seq_padding(S1), dtype=torch.float32)
                    S2 = torch.tensor(seq_padding(S2), dtype=torch.float32)
                    O1 = torch.tensor(seq_padding(O1))
                    O2 = torch.tensor(seq_padding(O2))
                    K1, K2 = torch.tensor(K1), torch.tensor(K2)
                    yield [T, S1, S2, K1, K2, O1, O2, TM, TS, TT], None
                    T, S1, S2, K1, K2, O1, O2, TM, TS,TT = [], [], [], [], [], [], [], [], [],[]


train_D = data_generator(train_data)


class SubjectModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SubjectModel, self).__init__(config)
        self.bert = BertModel(config)

        self.wv_linear = nn.Linear(in_features=200, out_features=hidden_size)

        self.linear1 = nn.Linear(in_features=hidden_size, out_features=1)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=1)

        self.apply(self.init_bert_weights)

    def forward(self, input_id, token_type_id, input_mask, input_wv):
        input_wv = self.wv_linear(input_wv)
        encoder_layers, _ = self.bert(input_id, token_type_id, input_mask, output_all_encoded_layers=False)
        x = input_wv + encoder_layers #[b,s,h]

        ps1 = torch.sigmoid(self.linear1(x).squeeze(-1))  # [b,s,h]->[b,s,1]->[b,s]
        ps2 = torch.sigmoid(self.linear2(x).squeeze(-1))

        return ps1, ps2, encoder_layers


def gather(indexs, mat):
    tmp_list = []
    batch_size = mat.size(0)
    for i in range(batch_size):
        tmp_list.append(mat[i][indexs[i]])
    return torch.stack(tmp_list)


class ObjectModel(nn.Module):
    def __init__(self):
        super(ObjectModel, self).__init__()
        self.linear11 = nn.Linear(in_features=hidden_size*3, out_features=hidden_size*3//2)
        self.linear12 = nn.Linear(in_features=hidden_size*3//2, out_features=num_classes + 1)
        self.linear21 = nn.Linear(in_features=hidden_size*3, out_features=hidden_size*3//2)
        self.linear22 = nn.Linear(in_features=hidden_size*3//2, out_features=num_classes + 1)

    def forward(self, x_b, k1, k2):
        k1 = gather(k1, x_b)
        k2 = gather(k2, x_b)  # [b,h]

        k = torch.cat([k1, k2], dim=1)  # [b,h*2]
        h = torch.cat([x_b, k.unsqueeze(1).to(torch.float32).expand(x_b.size(0), x_b.size(1), k.size(1))], dim=2)  # [b,s,h*3]

        x_l1 = self.linear11(h)
        x_l2 = self.linear21(h)

        po1 = self.linear12(x_l1)  # [b,s,num_class]
        po2 = self.linear22(x_l2)

        return po1, po2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
logger.info(f'use {n_gpu} gpu')

subject_model = SubjectModel.from_pretrained(pretrained_model_name_or_path=bert_model_path, cache_dir=bert_data_path)
object_model = ObjectModel()

subject_model.to(device)
object_model.to(device)
if n_gpu > 1:
    torch.cuda.manual_seed_all(42)

    logger.info(f'let us use {n_gpu} gpu')
    subject_model = torch.nn.DataParallel(subject_model)
    object_model = torch.nn.DataParallel(object_model)


# loss
b_loss_func = nn.BCELoss(reduction='none')
loss_func = nn.CrossEntropyLoss(reduction='none')

# optim
param_optimizer = list(subject_model.named_parameters()) + list(object_model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

learning_rate = 5e-5
warmup_proportion = 0.1
num_train_optimization_steps = len(train_data) // batch_size * epoch_num
logger.info(f'num_train_optimization: {num_train_optimization_steps}')

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)


def extract_items(text_in, text_words):
    R = []
    _s = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text_in]
    _sw = [word2id.get(w, 1) for w in text_words.split()]
    _input_mask = [1] * len(_s)
    _s = torch.tensor([_s], dtype=torch.long, device=device)
    _input_mask = torch.tensor([_input_mask], dtype=torch.long, device=device)
    _segment_ids = torch.zeros(*_s.size(), dtype=torch.long, device=device)

    with torch.no_grad():
        _k1, _k2, _t_b = subject_model(_s, _segment_ids, _input_mask)  # _k1:[b,s]

    _k1, _k2 = _k1[0, :], _k2[0, :]
    for i, _kk1 in enumerate(_k1):
        if _kk1 > 0.5:
            _subject = ''
            for j, _kk2 in enumerate(_k2[i:]):
                if _kk2 > 0.5:
                    _subject = text_in[i: i + j + 1]
                    break
            if _subject:
                _kk1, _kk2 = torch.tensor([i]), torch.tensor([i + j])
                with torch.no_grad():
                    _o1, _o2 = object_model(_t_b, _kk1, _kk2)  # [b,s,50]
                _o1, _o2 = torch.argmax(_o1[0], 1), torch.argmax(_o2[0], 1)
                _o1 = _o1.detach().cpu().numpy()
                _o2 = _o2.detach().cpu().numpy()
                for m, _oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for n, _oo2 in enumerate(_o2[m:]):
                            if _oo2 == _oo1:
                                _object = text_in[m: m + n + 1]
                                _predicate = id2predicate[_oo1]
                                R.append((_subject, _predicate, _object))
                                break
    return list(set(R))


err_log = (Path(data_dir) / 'err_log.json').open('w')
err_dict = defaultdict(list)

best_score = 0
best_epoch = 0
for e in range(epoch_num):
    subject_model.train()
    object_model.train()
    batch_idx = 0
    tr_total_loss = 0
    dev_total_loss = 0

    for batch in train_D:
        batch_idx += 1

        batch = tuple(t.to(device) for t in batch[0])
        T, S1, S2, K1, K2, O1, O2, TM, TS,TT = batch
        pred_s1, pred_s2, x_bert = subject_model(T, TS, TM,TT)
        pred_o1, pred_o2 = object_model(x_bert, K1, K2)

        s1_loss = b_loss_func(pred_s1, S1)  # [b,s]
        s2_loss = b_loss_func(pred_s2, S2)

        x_mask_ = 1 - TM
        x_mask_ = x_mask_.type(torch.ByteTensor)
        x_mask_ = x_mask_.to(device)

        s1_loss.masked_fill_(x_mask_, 0)
        s2_loss.masked_fill_(x_mask_, 0)

        o1_loss = loss_func(pred_o1.permute(0, 2, 1), O1)  # [b,s]
        o2_loss = loss_func(pred_o2.permute(0, 2, 1), O2)

        o1_loss.masked_fill_(x_mask_, 0)
        o2_loss.masked_fill_(x_mask_, 0)

        total_ele = torch.sum(TM)
        s1_loss = torch.sum(s1_loss) / total_ele
        s2_loss = torch.sum(s2_loss) / total_ele
        o1_loss = torch.sum(o1_loss) / total_ele
        o2_loss = torch.sum(o2_loss) / total_ele

        tmp_loss = (s1_loss + s2_loss) + (o1_loss + o2_loss)

        if n_gpu > 1:
            tmp_loss = tmp_loss.mean()

        tmp_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        tr_total_loss += tmp_loss.item()
        if batch_idx % 100 == 0:
            logger.info(f'Epoch:{e} - batch:{batch_idx}/{train_D.steps} - loss: {tr_total_loss / batch_idx:.8f}')

    subject_model.eval()
    object_model.eval()
    A, B, C = 1e-10, 1e-10, 1e-10
    for d in tqdm(iter(dev_data)):
        R = set(extract_items(d['text'], d['text_words']))
        T = set([tuple(i) for i in d['spo_list']])
        A += len(R & T)
        B += len(R)
        C += len(T)

        if R != T:
            err_dict['err'].append({'text': d['text'],
                                    'spo_list': d['spo_list'],
                                    'predict': list(R)})

    f1, precision, recall = 2 * A / (B + C), A / B, A / C
    if f1 > best_score:
        best_score = f1
        best_epoch = e

        json.dump(err_dict, err_log, ensure_ascii=False)


    logger.info(
        f'Epoch:{e}-precision:{precision:.4f}-recall:{recall:.4f}-f1:{f1:.4f} - best f1: {best_score:.4f} - best epoch:{best_epoch}')
