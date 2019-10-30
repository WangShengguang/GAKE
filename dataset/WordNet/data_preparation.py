import pandas as pd
import os
from tqdm import tqdm

from nltk.corpus import wordnet as wn

DATA_PATH = 'data/WN18RR'


def generate_sentence_func(df):
    head_description = wn.synset(df['head']).definition()
    relation_text = df['relation'].replace('_', ' ').strip()
    tail_description = wn.synset(df['tail']).definition()
    sentences = '[CLS] ' + head_description + ' [SEP] ' + \
        relation_text + ' [SEP] ' + tail_description + ' [SEP]'
    return sentences


def prepare_WN18RR(path='data/datasets_knowledge_embedding/WN18RR/text', verbose=True):
    for txt in ['train', 'test', 'valid']:
        if verbose:
            print('Processing', txt)
        filepath = os.path.join(path, txt + '.txt')
        ori_df = pd.read_csv(filepath, delimiter='\t', names=[
                             'head', 'relation', 'tail'])
        if verbose:
            tqdm.pandas()
            ori_df['sentence'] = ori_df.progress_apply(
                generate_sentence_func, axis=1)
        else:
            ori_df['sentence'] = ori_df.apply(generate_sentence_func, axis=1)

        ori_df.to_csv(os.path.join(DATA_PATH, txt + '.csv'),
                      sep=',', index=False)


if __name__ == "__main__":
    import nltk
    print('Check WordNet')
    print(nltk.download('wordnet'))
    print('Processing WN18RR')
    os.makedirs(DATA_PATH, exist_ok=True)
    prepare_WN18RR()
