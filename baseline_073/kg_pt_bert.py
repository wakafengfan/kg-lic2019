import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertAdam
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from tqdm import tqdm

from configuration.config import data_dir, bert_vocab_path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

train_data = json.load(open(data_dir + '/train_data_me.json'))
dev_data = json.load(open(data_dir + '/dev_data_me.json'))

id2predicate, predicate2id = json.load(open(data_dir + '/all_50_schemas_me.json'))
id2predicate = {int(i): j for i, j in id2predicate.items()}

id2char, char2id = json.load(open(data_dir + '/all_chars_me.json'))

hidden_size = 256
num_classes = len(id2predicate)
batch_size = 64
epoch_num = 6
MAX_SEQ_LENGTH = 160

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class BertSequenceLabeling(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSequenceLabeling, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, ):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        return pooled_output


class InputFeature:
    def __init__(self, input_id, segment_ids, input_mask):
        self.input_id = input_id
        self.segment_ids = segment_ids
        self.input_mask = input_mask


def convert_example_to_feature(texts, max_seq_length=MAX_SEQ_LENGTH):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_vocab_path, do_lower=True)

    features = []

    for idx, text in enumerate(texts):
        text = [c for c in text]
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']

        segment_ids = [0] * len(tokens_a)
        input_mask = [1] * len(tokens_a)

        input_id = tokenizer.convert_tokens_to_ids(tokens_a)

        # zero-padding
        padding = [0] * (max_seq_length - len(input_id))
        input_id += padding
        segment_ids += padding
        input_mask += padding

        assert len(input_id) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        if idx < 5:
            logger.info("*** Examples ***")
            logger.info("Input id: %s" % " ".join([str(x) for x in input_id]))
            logger.info("Segment id: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("Input mask: %s" % " ".join([str(x) for x in input_mask]))

        features.append(InputFeature(input_id=input_id, segment_ids=segment_ids, input_mask=input_mask))

    return features

train_features = convert_example_to_feature()


class SubjectModel(nn.Module):
    def __init__(self):
        super(SubjectModel, self).__init__()
        self.embeddings = nn.Embedding(len(char2id) + 2, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)

        self.conv = nn.Conv1d(in_channels=hidden_size * 2,
                              out_channels=hidden_size,
                              kernel_size=3,
                              padding=1)

        self.linear1 = nn.Linear(in_features=hidden_size, out_features=1)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x_idx, x_length):
        x_masks = torch.eq(x_idx, 0)
        x_masks.requires_grad = False

        x_embed = self.embeddings(x_idx)
        x_embed = F.dropout(x_embed, p=0.25, training=self.training)

        packed = torch.nn.utils.rnn.pack_padded_sequence(x_embed, x_length, batch_first=True)
        t, _ = self.lstm1(packed)
        t, _ = self.lstm2(t)  # [b,s,h]
        t, _ = torch.nn.utils.rnn.pad_packed_sequence(t, batch_first=True)

        t_max = F.max_pool1d(t.masked_fill(x_masks.unsqueeze(2), -1e10).permute(0, 2, 1), kernel_size=t.size(1))
        t_max = t_max.squeeze(-1).unsqueeze(1)  # [b,1,h]

        t_concat = torch.cat([t, t_max.expand_as(t)], dim=-1)  # [b,s,h*2]
        t_conv = F.relu(self.conv(t_concat.permute(0, 2, 1))).permute(0, 2, 1)  # [b,s,h]

        ps1 = torch.sigmoid(self.linear1(t_conv).squeeze(-1))  # [b,s,h]->[b,s,1]->[b,s]
        ps2 = torch.sigmoid(self.linear2(t_conv).squeeze(-1))

        return ps1, ps2, t, t_concat, x_masks


def gather(indexs, mat):
    tmp_list = []
    batch_size = mat.size(0)
    for i in range(batch_size):
        tmp_list.append(mat[i][indexs[i]])
    return torch.stack(tmp_list)


class ObjectModel(nn.Module):
    def __init__(self):
        super(ObjectModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=hidden_size * 4,
                              out_channels=hidden_size,
                              kernel_size=3,
                              padding=1)
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=num_classes + 1)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=num_classes + 1)

    def forward(self, t, t_concat, k1, k2):
        k1 = gather(k1, t)
        k2 = gather(k2, t)  # [b,h]

        k = torch.cat([k1, k2], dim=1)  # [b,h*2]
        h = torch.cat([t_concat, k.unsqueeze(1).to(torch.float32).expand_as(t_concat)], dim=2)  # [b,s,h*4]

        h_conv = F.relu(self.conv(h.permute(0, 2, 1))).permute(0, 2, 1)  # [b,s,h]

        po1 = self.linear1(h_conv)  # [b,s,num_class]
        po2 = self.linear2(h_conv)

        return po1, po2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
b_loss_func = nn.BCELoss(reduction='none')
loss_func = nn.CrossEntropyLoss(reduction='none')

# optim
param_optimizer = list(subject_model.named_parameters() + object_model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

learning_rate = 5e-5
warmup_proportion = 0.1
num_train_optimization_steps = len(train_data) / batch_size * epoch_num

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)

def extract_items(text_in):
    R = []
    _s = [char2id.get(c, 1) for c in text_in]
    _l = torch.tensor([len(_s)])
    _s = torch.tensor([_s])
    with torch.no_grad():
        _k1, _k2, _t, _t_concat, _t_mask = subject_model(_s.to(device), _l.to(device))
        _k1.masked_fill_(_t_mask, 0)
        _k2.masked_fill_(_t_mask, 0)

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
                    _o1, _o2 = object_model(_t, _t_concat, _kk1, _kk2)  # [b,s,50]
                    _o1.masked_fill_(_t_mask.unsqueeze(2), 0)
                    _o2.masked_fill_(_t_mask.unsqueeze(2), 0)
                _o1, _o2 = torch.argmax(_o1[0], 1), torch.argmax(_o2[0], 1)
                _o1 = _o1.detach().cpu().numpy()
                _o2 = _o2.detach().cpu().numpy()
                for i, _oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j, _oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = text_in[i: i + j + 1]
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
        T, S1, S2, K1, K2, O1, O2, Ls = batch
        pred_s1, pred_s2, x_lstm2_, x_concat_, x_mask_ = subject_model(T, Ls)
        pred_o1, pred_o2 = object_model(x_lstm2_, x_concat_, K1, K2)

        s1_loss = b_loss_func(pred_s1, S1)  # [b,s]
        s2_loss = b_loss_func(pred_s2, S2)

        s1_loss.masked_fill_(x_mask_, 0)
        s2_loss.masked_fill_(x_mask_, 0)

        o1_loss = loss_func(pred_o1.permute(0, 2, 1), O1)  # [b,s]
        o2_loss = loss_func(pred_o2.permute(0, 2, 1), O2)

        o1_loss.masked_fill_(x_mask_, 0)
        o2_loss.masked_fill_(x_mask_, 0)

        total_ele = torch.sum(1 - x_mask_)
        s1_loss = torch.sum(s1_loss) / total_ele
        s2_loss = torch.sum(s2_loss) / total_ele
        o1_loss = torch.sum(o1_loss) / total_ele
        o2_loss = torch.sum(o2_loss) / total_ele

        tmp_loss = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)

        # if n_gpu > 1:
        #     tmp_loss = tmp_loss.mean()

        tr_total_loss += tmp_loss.item()

        optim.zero_grad()
        tmp_loss.backward()
        optim.step()

        if batch_idx % 100 == 0:
            logger.info(f'Epoch:{e} - batch:{batch_idx}/{train_D.steps} - loss: {tr_total_loss / batch_idx:.8f}')

    subject_model.eval()
    object_model.eval()
    A, B, C = 1e-10, 1e-10, 1e-10
    for d in tqdm(iter(dev_data)):
        R = set(extract_items(d['text']))
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

        # s_model_to_save = subject_model.module if hasattr(subject_model, 'module') else subject_model
        # o_model_to_save = object_model.module if hasattr(object_model, 'module') else object_model

        # torch.save(s_model_to_save.state_dict(), model_dir + '/subject_model.pt')
        # torch.save(o_model_to_save.state_dict(), model_dir + '/object_model.pt')

    logger.info(
        f'Epoch:{e}-precision:{precision:.4f}-recall:{recall:.4f}-f1:{f1:.4f} - best f1: {best_score:.4f} - best epoch:{best_epoch}')
