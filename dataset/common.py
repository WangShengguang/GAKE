import json

from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()


def sentence_to_words(sentence):
    sentence = sentence.lstrip('[CLS]').strip()
    sent_li = sentence.split('[SEP]')
    words_list = []
    for s in sent_li:
        words_list.extend(tokenizer.tokenize(s) + ['[SEP]'])
    return ['[CLS]'] + words_list


def get_id_table_from_set(item_set, save_file='', pos0=None, preserve=None):
    table = dict()
    if pos0:
        table[pos0] = 0
    if preserve:
        for item in preserve:
            table[item] = len(table)
    for item in sorted(list(item_set)):
        table[item] = len(table)

    if save_file:
        with open(save_file, 'w') as stream:
            json.dump(table, stream, indent=4)

    return table
