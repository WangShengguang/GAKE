import pandas as pd
import os

DATA_PATH = 'data/WN18RR'

def load_WN18RR(path='data/datasets_knowledge_embedding/WN18RR/text'):
    for txt in ['train', 'test', 'valid']:
        filepath = os.path.join(path, txt + '.txt')
        ori_df = pd.read_csv(filepath, delimiter='\t', names=['head', 'relation', 'tail'])
        # TODO generate sentence using description

        ori_df.to_csv(os.path.join(DATA_PATH, txt + '.tsv'), sep='\t', index=False)

if __name__ == "__main__":
    os.makedirs(DATA_PATH)
    load_WN18RR()
