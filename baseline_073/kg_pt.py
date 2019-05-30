import json
import logging
from random import choice

import gensim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from configuration.config import data_dir, model_dir

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


train_data = json.load(open(data_dir + '/train_data_me.json'))
dev_data = json.load(open(data_dir + '/dev_data_me.json'))

id2predicate, predicate2id = json.load(open(data_dir + '/all_50_schemas_me.json'))
id2predicate = {int(i):j for i,j in id2predicate.items()}

id2char, char2id = json.load(open(data_dir + '/all_chars_me.json'))

hidden_size = 256
num_classes = len(id2predicate)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]


class data_generator:
    def __init__(self, data, batch_size=64):
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
        T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []
        for i in idxs:
            d = self.data[i]
            text = d['text']
            items = {}
            for sp in d['spo_list']:
                subjectid = text.find(sp[0])
                objectid = text.find(sp[2])
                if subjectid != -1 and objectid != -1:
                    key = (subjectid, subjectid+len(sp[0]))
                    if key not in items:
                        items[key] = []
                    items[key].append((objectid,
                                       objectid+len(sp[2]),
                                       predicate2id[sp[1]]))
            if items:
                T.append([char2id.get(c, 1) for c in text]) # 1是unk，0是padding
                s1, s2 = [0] * len(text), [0] * len(text)
                for j in items:
                    s1[j[0]] = 1
                    s2[j[1]-1] = 1
                k1, k2 = choice(list(items.keys()))
                o1, o2 = [0] * len(text), [0] * len(text) # 0是unk类（共49+1个类）
                for j in items[(k1, k2)]:
                    o1[j[0]] = j[2]
                    o2[j[1]-1] = j[2]
                S1.append(s1)
                S2.append(s2)
                K1.append(k1)
                K2.append(k2-1)
                O1.append(o1)
                O2.append(o2)
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = torch.tensor(seq_padding(T))
                    S1 = torch.FloatTensor(seq_padding(S1))
                    S2 = torch.FloatTensor(seq_padding(S2))
                    O1 = torch.tensor(seq_padding(O1))
                    O2 = torch.tensor(seq_padding(O2))
                    K1, K2 = torch.tensor(K1), torch.tensor(K2)
                    yield [T, S1, S2, K1, K2, O1, O2], None
                    T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []

train_D = data_generator(train_data)

def load_char_embedding():
    f = data_dir + '/w2v_char_py3_baidu_1112'
    char_embed_model = gensim.models.KeyedVectors.load(f)
    char_embed = np.random.random((len(id2char)+2, 150))
    for c, cid in char2id.items():
        if c in char_embed_model.vocab:
            char_embed[cid] = char_embed_model.word_vec(c)
    return char_embed

pretrained_embeddings = torch.tensor(load_char_embedding(), dtype=torch.float32)

class SubjectModel(nn.Module):
    def __init__(self):
        super(SubjectModel, self).__init__()
        # self.embeddings = nn.Embedding(len(char2id)+2, 150)
        self.embeddings = nn.Embedding(len(char2id)+2, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)

        self.conv = nn.Conv1d(in_channels=hidden_size*2,
                              out_channels=hidden_size,
                              kernel_size=3,
                              padding=1)

        self.linear1 = nn.Linear(in_features=hidden_size, out_features=1)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=1)

        self._init_params()

    def _init_params(self):
        nn.init.orthogonal_(getattr(self.lstm1, 'weight_hh_l0'))
        nn.init.kaiming_normal_(getattr(self.lstm1, 'weight_ih_l0'))
        nn.init.constant_(getattr(self.lstm1, 'bias_hh_l0'), 0)
        nn.init.constant_(getattr(self.lstm1, 'bias_ih_l0'), 0)
        getattr(self.lstm1, f'bias_hh_l{0}').chunk(4)[1].fill_(1)

        nn.init.orthogonal_(getattr(self.lstm2, 'weight_hh_l0'))
        nn.init.kaiming_normal_(getattr(self.lstm2, 'weight_ih_l0'))
        nn.init.constant_(getattr(self.lstm2, 'bias_hh_l0'), 0)
        nn.init.constant_(getattr(self.lstm2, 'bias_ih_l0'), 0)
        getattr(self.lstm2, f'bias_hh_l{0}').chunk(4)[1].fill_(1)

        nn.init.orthogonal_(getattr(self.lstm1, f'weight_hh_l{0}_reverse'))
        nn.init.kaiming_normal_(getattr(self.lstm1, f'weight_ih_l{0}_reverse'))
        nn.init.constant_(getattr(self.lstm1, f'bias_hh_l{0}_reverse'), val=0)
        nn.init.constant_(getattr(self.lstm1, f'bias_ih_l{0}_reverse'), val=0)
        getattr(self.lstm1, f'bias_hh_l{0}_reverse').chunk(4)[1].fill_(1)
        
        nn.init.orthogonal_(getattr(self.lstm2, f'weight_hh_l{0}_reverse'))
        nn.init.kaiming_normal_(getattr(self.lstm2, f'weight_ih_l{0}_reverse'))
        nn.init.constant_(getattr(self.lstm2, f'bias_hh_l{0}_reverse'), val=0)
        nn.init.constant_(getattr(self.lstm2, f'bias_ih_l{0}_reverse'), val=0)
        getattr(self.lstm2, f'bias_hh_l{0}_reverse').chunk(4)[1].fill_(1)

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)

        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x_idx):
        x_masks = torch.eq(x_idx, 0)
        x_masks.requires_grad = False

        x_embed = self.embeddings(x_idx)
        # x_embed = F.dropout(x_embed, p=0.25, training=self.training)

        t, _ = self.lstm1(x_embed)
        t, _ = self.lstm2(t)  # [b,s,h]

        t_max = F.max_pool1d(t.masked_fill(x_masks.unsqueeze(2), -1e10).permute(0,2,1), kernel_size=t.size(1))
        t_max = t_max.squeeze(-1).unsqueeze(1)  # [b,1,h]

        t_concat = torch.cat([t, t_max.expand_as(t)], dim=-1)  # [b,s,h*2]
        t_conv = F.relu(self.conv(t_concat.permute(0, 2, 1))).permute(0,2,1)  # [b,s,h]

        ps1 = torch.sigmoid(self.linear1(t_conv).squeeze(-1))  # [b,s,h]->[b,s,1]->[b,s]
        ps2 = torch.sigmoid(self.linear2(t_conv).squeeze(-1))

        return ps1, ps2, t, t_concat


def gather(indexs, mat):
    tmp_list = []
    batch_size = mat.size(0)
    for i in range(batch_size):
        tmp_list.append(mat[i][indexs[i]])
    return torch.stack(tmp_list)


class ObjectModel(nn.Module):
    def __init__(self):
        super(ObjectModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=hidden_size*4,
                              out_channels=hidden_size,
                              kernel_size=3,
                              padding=1)
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=num_classes+1)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=num_classes+1)

        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)

        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, t, t_concat, k1, k2):
        k1 = gather(k1, t)
        k2 = gather(k2, t)  # [b,h]

        k = torch.cat([k1, k2], dim=1)  # [b,h*2]
        h = torch.cat([t_concat, k.unsqueeze(1).to(torch.float32).expand_as(t_concat)], dim=2)  # [b,s,h*4]

        h_conv = F.relu(self.conv(h.permute(0,2,1))).permute(0,2,1)  # [b,s,h]

        po1 = self.linear1(h_conv)  # [b,s,num_class]
        po2 = self.linear2(h_conv)

        return po1, po2


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()

subject_model = SubjectModel()
object_model = ObjectModel()

subject_model.to(device)
object_model.to(device)
if n_gpu > 1:
    logger.info(f'let us use {n_gpu} gpu')
    subject_model = torch.nn.DataParallel(subject_model)
    object_model = torch.nn.DataParallel(object_model)

# loss
b_loss_func = nn.BCELoss()
loss_func = nn.CrossEntropyLoss()

params = list(subject_model.parameters()) + list(object_model.parameters())
optim = optim.Adam(params, lr=0.001)

def extract_items(text_in):
    R = []
    _s = [char2id.get(c, 1) for c in text_in]
    _s = torch.tensor([_s])
    with torch.no_grad():
        _k1, _k2, _t, _t_concat = subject_model(_s.to(device))
    _k1, _k2 = _k1[0, :], _k2[0, :]
    for i,_kk1 in enumerate(_k1):
        if _kk1 > 0.5:
            _subject = ''
            for j,_kk2 in enumerate(_k2[i:]):
                if _kk2 > 0.5:
                    _subject = text_in[i: i+j+1]
                    break
            if _subject:
                _kk1, _kk2 = torch.tensor([i]), torch.tensor([i+j])
                with torch.no_grad():
                    _o1, _o2 = object_model(_t, _t_concat, _kk1, _kk2)  # [b,s,50]
                _o1, _o2 = torch.argmax(_o1[0], 1), torch.argmax(_o2[0], 1)
                _o1 = _o1.detach().cpu().numpy()
                _o2 = _o2.detach().cpu().numpy()
                for i,_oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j,_oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = text_in[i: i+j+1]
                                _predicate = id2predicate[_oo1]
                                R.append((_subject, _predicate, _object))
                                logger.info(f'{_subject} {_predicate} {_object}')
                                break
    return list(set(R))



best_score = 0
best_epoch = 0
for e in range(10):
    subject_model.train()
    object_model.train()
    batch_idx = 0
    tr_total_loss = 0
    dev_total_loss = 0

    for batch in train_D:
        batch_idx += 1
        batch = tuple(t.to(device) for t in batch[0])
        T, S1, S2, K1, K2, O1, O2 = batch
        pred_s1, pred_s2, x_lstm2_, x_concat_ = subject_model(T)
        pred_o1, pred_o2 = object_model(x_lstm2_, x_concat_, K1, K2)

        s1_loss = b_loss_func(pred_s1, S1)
        s2_loss = b_loss_func(pred_s2, S2)

        o1_loss = loss_func(pred_o1.permute(0,2,1), O1)  # [b,s]
        o2_loss = loss_func(pred_o2.permute(0,2,1), O2)

        tmp_loss = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)

        if n_gpu > 1:
            tmp_loss = tmp_loss.mean()

        tr_total_loss += tmp_loss.item()

        optim.zero_grad()
        tmp_loss.backward()
        optim.step()

        if batch_idx % 100 == 0:
            logger.info(f'Epoch:{e} - batch:{batch_idx}/{train_D.steps} - loss: {tr_total_loss/batch_idx:.4f}')

    subject_model.eval()
    object_model.eval()
    A, B, C = 1e-10, 1e-10, 1e-10
    for d in tqdm(iter(dev_data)):
        R = set(extract_items(d['text']))
        T = set([tuple(i) for i in d['spo_list']])
        A += len(R & T)
        B += len(R)
        C += len(T)

    f1, precision, recall = 2 * A / (B + C), A / B, A / C
    if f1 > best_score:
        best_score = f1
        best_epoch = e

        # s_model_to_save = subject_model.module if hasattr(subject_model, 'module') else subject_model
        # o_model_to_save = object_model.module if hasattr(object_model, 'module') else object_model

        # torch.save(s_model_to_save.state_dict(), model_dir + '/subject_model.pt')
        # torch.save(o_model_to_save.state_dict(), model_dir + '/object_model.pt')

    logger.info(f'Epoch:{e}-precision:{precision:.4f}-recall:{recall:.4f}-f1:{f1:.4f} - best f1: {best_score:.4f} - best epoch:{best_epoch}')
















































