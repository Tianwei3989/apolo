import pandas as pd
import os
from tqdm import tqdm

df_artemis = pd.read_csv(os.path.join('./data/artemis_index', 'artemis_dataset_release_v0.csv'))

for split in ['train','val','test']:
    print('Processing', split, 'set...')
    df = pd.read_json(os.path.join('./data/artemis_index', split + '_index.json'))
    df_0 = pd.DataFrame()

    for i in tqdm(range(df.shape[0])):
        artemis_idx = df['artemis_id'][i]
        for j in artemis_idx:
            df_0 = pd.concat([df_0, df_artemis[df_artemis.index == j]])

    df_0['utterance'] = df_0['utterance'] + ' '
    aggregation_functions = {'art_style	': 'first',
                             'utterance': 'sum',
                             'repetition': 'first',
                             }

    df_1 = df_0.groupby([df_0['painting'], df_0['emotion']]).aggregate(aggregation_functions).reset_index()

    df_1.to_json(os.path.join('./data', split + '.json'),orient='records')