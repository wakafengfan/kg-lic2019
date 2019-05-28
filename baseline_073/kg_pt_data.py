import json

train_data = json.load(open('../datasets/train_data_me.json'))
dev_data = json.load(open('../datasets/dev_data_me.json'))
id2predicate, predicate2id = json.load(open('../datasets/all_50_schemas_me.json'))
id2predicate = {int(i):j for i,j in id2predicate.items()}
id2char, char2id = json.load(open('../datasets/all_chars_me.json'))


