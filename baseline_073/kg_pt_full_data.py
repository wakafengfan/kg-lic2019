import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import ahocorasick
from tqdm import tqdm

from configuration.config import data_dir

train_data = json.load(open(data_dir + '/train_data_me_2.json'))
dev_data = json.load(open(data_dir + '/dev_data_me_2.json'))

predicates = defaultdict(list)  # 格式：{predicate: [(subject, predicate, object)]}


def repair(d):
    # 歌手、作词、作曲三个schema的subject如果是专辑名，则删除该spo

    d['text'] = d['text'].lower()
    something = re.findall(u'《([^《》]*?)》', d['text'])  # 提取书名号里面内容
    something = [s.strip() for s in something]
    zhuanji = []
    gequ = []
    for sp in d['spo_list']:
        sp[0] = sp[0].strip(u'《》').strip().lower()
        sp[2] = sp[2].strip(u'《》').strip().lower()
        for some in something:
            if sp[0] in some and d['text'].count(sp[0]) == 1:
                sp[0] = some
        if sp[1] == '所属专辑':
            zhuanji.append(sp[2])
            gequ.append(sp[0])
    spo_list = []
    for sp in d['spo_list']:
        if sp[1] in ['歌手', '作词', '作曲']:
            if sp[0] in zhuanji and sp[0] not in gequ:
                continue
        spo_list.append(tuple(sp))
    d['spo_list'] = spo_list


# train data + repair
for d in train_data:
    repair(d)
    for sp in d['spo_list']:
        predicates[sp[1]].append(sp)
# dev data + repair
for d in dev_data:
    repair(d)

# 远程监督的ac automation
s_ac = ahocorasick.Automaton()
o_ac = ahocorasick.Automaton()
sp2o = defaultdict(set)
spo_total = defaultdict(set)

for i, d in tqdm(enumerate(train_data), desc='构建三元组搜索器'):
    for s, p, o in d['spo_list']:
        s_ac.add_word(s, s)
        o_ac.add_word(o, o)
        sp2o[(s, o)].add(p)
        spo_total[(s, p, o)].add(i)

pickle.dump(s_ac, (Path(data_dir) / 's_ac.pkl').open('wb'))
pickle.dump(o_ac, (Path(data_dir) / 'o_ac.pkl').open('wb'))

json.dump(train_data, (Path(data_dir) / 'train_data_me_3.json').open('w'), indent=4, ensure_ascii=False)
json.dump(dev_data, (Path(data_dir) / 'dev_data_me_3.json').open('w'), indent=4, ensure_ascii=False)

info_ = {
    'predicates': predicates,
    'sp2o': sp2o,
    'spo_total': spo_total
}

pickle.dump(info_, (Path(data_dir) / 'info_.pkl').open('wb'))
