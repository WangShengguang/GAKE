import pandas as pd
import os
from tqdm import tqdm

from nltk.corpus import wordnet as wn

from dataset.common import get_id_table_from_set, sentence_to_words

DATA_PATH = 'data/WN18RR'


def generate_sentence_func(df):
    head_description = wn.synset(df['head']).definition()
    relation_text = df['relation'].replace('_', ' ').strip()
    tail_description = wn.synset(df['tail']).definition()
    sentences = '[CLS] ' + head_description + ' [SEP] ' + \
        relation_text + ' [SEP] ' + tail_description + ' [SEP]'
    return sentences


def prepare_WN18RR(path='data/datasets_knowledge_embedding/WN18RR/text', verbose=True):
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
    import nltk
    print('Check WordNet')
    print(nltk.download('wordnet'))
    print('Processing WN18RR')
    os.makedirs(DATA_PATH, exist_ok=True)
    prepare_WN18RR()
