import pandas as pd
import os

from nltk.corpus import wordnet as wn

DATA_PATH = 'data/WN18RR'

def generate_sentence_func(df):
    head_description = wn.synset(df['head']).definition()
    relation_text = df['relation'].replace('_', ' ').strip()
    tail_description = wn.synset(df['tail']).definition()
    sentences = '[CLS] ' + head_description + ' [SEP] ' + relation_text + ' [SEP] ' + tail_description + ' [SEP]'
    return sentences


def prepare_WN18RR(path='data/datasets_knowledge_embedding/WN18RR/text'):
    for txt in ['train', 'test', 'valid']:
        filepath = os.path.join(path, txt + '.txt')
        ori_df = pd.read_csv(filepath, delimiter='\t', names=['head', 'relation', 'tail']).head()
        ori_df['sentence'] = ori_df.apply(generate_sentence_func, axis=1)
        ori_df['head'] = ori_df['head'].apply(lambda string: string.split('.')[0])
        # ori_df['relation'] = ori_df['relation'].apply(lambda string: string.replace('_', ' ').strip())
        ori_df['tail'] = ori_df['tail'].apply(lambda string: string.split('.')[0])
        ori_df.to_csv(os.path.join(DATA_PATH, txt + '.tsv'), sep=',', index=False)


if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)
    prepare_WN18RR()
