import pandas as pd
import os
from tqdm import tqdm
import json

from dataset.common import get_id_table_from_set, sentence_to_words

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
    entitiy_set = set()
    relation_set = set()
    word_set = set()

    for txt in ['train', 'test', 'valid']:
        if verbose:
            print('Processing', txt)
        filepath = os.path.join(path, txt + '.txt')
        df = pd.read_csv(filepath, delimiter='\t', names=[
            'head', 'relation', 'tail'])

        if verbose:
            tqdm.pandas()
            df['sentence'] = df.progress_apply(
                generate_sentence_func, axis=1)
        else:
            df['sentence'] = df.apply(generate_sentence_func, axis=1)

        # drop the data which can't construct sentence
        df = df.dropna()

        entitiy_set.update(df['head'].to_list() + df['tail'].to_list())
        relation_set.update(df['relation'].to_list())
        word_set.update([word for sent_list in df['sentence'].apply(
            sentence_to_words).to_list() for word in sent_list])

        df.to_csv(os.path.join(DATA_PATH, txt + '.csv'),
                  sep=',', index=False)

    print('Generating id tables')
    get_id_table_from_set(entitiy_set, f'{DATA_PATH}/entity2id.json')
    get_id_table_from_set(relation_set, f'{DATA_PATH}/relation2id.json')
    word_set.discard('[CLS]')
    word_set.discard('[SEP]')
    get_id_table_from_set(word_set, f'{DATA_PATH}/word2id.json',
                          '[PAD]', ['[CLS]', '[SEP]', '[UNK]', '[MASK]'])


if __name__ == "__main__":
    print('Processing FB15K-237')
    os.makedirs(DATA_PATH, exist_ok=True)
    prepare_FB15K_237()
