import json
import shutil
import tarfile
import tempfile
from pathlib import Path

# import pyhanlp
from tqdm import tqdm

from configuration.config import data_dir, tencent_w2v_path
word_set = set()

if not (Path(data_dir)/'train_data_me_2.json').exists():

    train_data = []
    for doc in tqdm(json.load((Path(data_dir)/'train_data_me.json').open())):
        text = doc['text']
        text_word = [i.word for i in pyhanlp.HanLP.segment(text)]
        doc.update({'text_words':' '.join(text_word)})
        train_data.append(doc)

        word_set.update(text_word)

    train_upt_path = (Path(data_dir)/'train_data_me_2.json').open('w')
    json.dump(train_data, train_upt_path, indent=4, ensure_ascii=False)
else:
    for doc in tqdm(json.load((Path(data_dir)/'train_data_me_2.json').open())):
        text_word = doc['text_words'].split()

        word_set.update(text_word)

if not (Path(data_dir)/'dev_data_me_2.json').exists():
    dev_data = []
    for doc in tqdm(json.load((Path(data_dir)/'dev_data_me.json').open())):
        text = doc['text']
        text_word = [i.word for i in pyhanlp.HanLP.segment(text)]
        doc.update({'text_words':' '.join(text_word)})
        dev_data.append(doc)

        word_set.update(text_word)

    dev_upt_path = (Path(data_dir)/'dev_data_me_2.json').open('w')
    json.dump(dev_data, dev_upt_path, indent=4, ensure_ascii=False)
else:
    for doc in tqdm(json.load((Path(data_dir)/'dev_data_me_2.json').open())):
        text_word = doc['text_words'].split()

        word_set.update(text_word)

print(f'total word: {len(word_set)}')  # 253590

# get tencent embedding from .gz
tmpdir = tempfile.mkdtemp()
with tarfile.open(Path(tencent_w2v_path)/'Tencent_AILab_ChineseEmbedding.tar.gz', 'r:gz') as archive:
    
    import os
    
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(archive, tmpdir)

serialization_dir = tmpdir
for fn in Path(serialization_dir).iterdir():
    print(fn.name)
    if 'Tencent_AILab_ChineseEmbedding' in fn.name:
        break
raw_tencent_embeds = fn.open(errors='ignore')
upt_tencent_embeds = (Path(data_dir)/'Tencent_AILab_ChineseEmbedding_gl.txt').open('w')
if tmpdir:
    shutil.rmtree(tmpdir)
first_line = raw_tencent_embeds.readline()
print(first_line)


# filter tencent embedding to glove format
word_cnt = 0
for line in tqdm(raw_tencent_embeds):
    l = line.strip().split()
    if len(l) != 201 or l[0] not in word_set:
        continue
    word_cnt += 1
    upt_tencent_embeds.write(line)

print(word_cnt)  #190855

