import pandas as pd

members_small_df = pd.read_csv('dataset/kkbox/members_small.csv')
train_df = pd.read_csv('dataset/kkbox/train.csv')

memid_list = members_small_df['msno'].tolist()
train_df = train_df[train_df['msno'].isin(memid_list)]
train_df.to_csv('dataset/kkbox/train_small.csv', index=False)

songs_df = pd.read_csv('dataset/songs_merged.csv')
songs_df = songs_df[songs_df['song_id'].isin(train_df['song_id'].unique())]
songs_df.to_csv('dataset/songs_merged_small.csv', index=False)