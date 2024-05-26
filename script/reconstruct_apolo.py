import pandas as pd
import os

df_artemis = pd.read_csv(os.path.join('./data', 'artemis_dataset_release_v0.csv'))
df_apolo = pd.read_json(os.path.join('./data', 'apolo.json'))

emotions = []
utterances = []

for i in range(df_apolo.shape[0]):
    utterance = []
    artemis_idx = df_apolo['artemis_id'].iloc[i]

    for j in artemis_idx:
        sample_artemis = df_artemis.iloc[j]
        utterance.append(sample_artemis['utterance'])

    emotions.append(df_artemis.iloc[artemis_idx[0]]['emotion'])
    utterances.append(utterance)

df_apolo['emotion'] = emotions
df_apolo['utterances'] = utterances

df_apolo[df_apolo['split']=='val'].to_json(os.path.join('./data', 'apolo_val.json'),orient='records')
df_apolo[df_apolo['split']=='test'].to_json(os.path.join('./data', 'apolo_test.json'),orient='records')