import shutil
from pathlib import Path

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

from configuration.config import data_dir

glove_file = Path(data_dir)/'Tencent_AILab_ChineseEmbedding_gl.txt'
glove_file = datapath(glove_file)

w2v_file = get_tmpfile(Path(data_dir)/'tmpfile')
glove2word2vec(glove_file, w2v_file)

m = gensim.models.KeyedVectors.load_word2vec_format(w2v_file)
m.save(str(Path(data_dir)/'tencen_embed_for_el2019'))

shutil.rmtree(Path(data_dir)/'tmpfile')