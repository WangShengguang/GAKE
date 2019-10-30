import pandas as pd
import os
from tqdm import tqdm

import json

DATA_PATH = 'data/FB15K-237'


def load_entity2wikidata(path='data/datasets_knowledge_embedding/FB15k-237/entity2wikidata.json'):
    with open(path, 'r') as stream:
        return json.load(stream)


entity2wikidata = load_entity2wikidata()


def generate_sentence_func(df):
    try:
        head_description = entity2wikidata[df['head']]['description']
        tail_description = entity2wikidata[df['tail']]['description']
    except KeyError:
        # key not in entity2wikidata
        head_description = None
        tail_description = None
    relation_text = df['relation'].replace('/', ' ').strip()
    try:
        sentences = '[CLS] ' + head_description + ' [SEP] ' + \
            relation_text + ' [SEP] ' + tail_description + ' [SEP]'
    except:
        # head_description or tail_description is None
        sentences = None
    return sentences


def prepare_FB15K_237(path='data/datasets_knowledge_embedding/FB15k-237', verbose=True):

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

        # drop the data which can't construct sentence
        ori_df = ori_df.dropna()

        ori_df.to_csv(os.path.join(DATA_PATH, txt + '.csv'),
                      sep=',', index=False)


if __name__ == "__main__":
    print('Processing FB15K-237')
    os.makedirs(DATA_PATH, exist_ok=True)
    prepare_FB15K_237()
