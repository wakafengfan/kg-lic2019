import os


ROOT_PATH = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

# data
data_dir = os.path.join(ROOT_PATH, "data")
model_dir = os.path.join(ROOT_PATH, "model")

bert_vocab_path = os.path.join(data_dir, 'bert_data', 'bert-base-chinese-vocab.txt')
bert_model_path = os.path.join(data_dir, 'bert_data', 'bert-base-chinese.tar.gz')