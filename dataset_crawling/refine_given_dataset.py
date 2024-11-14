import pandas as pd
import hydra
from collections import Counter

@hydra.main(config_path='../config', config_name='packed', version_base='1.1')
def main(config):
    songs_df = pd.read_csv(config.kkbox.songs_csv_fn)
    songs_extra_df = pd.read_csv(config.kkbox.song_extra_info_csv_fn)

    songs_df = songs_df.merge(songs_extra_df, on='song_id', how='left')
    
    # Drop duplicates
    in_csv_col_names = config.csv_column_names.song
    songs_df = songs_df.drop_duplicates(subset=[in_csv_col_names.title, in_csv_col_names.artist], keep='first')

    # Leave only songs that are in train or test
    train_df = pd.read_csv(config.kkbox.train_csv_fn)
    counter = Counter(train_df['song_id'].values)
    songs_df['train_count'] = songs_df['song_id'].map(counter)
    songs_df = songs_df.sort_values(by='train_count', ascending=False).reset_index(drop=True)

    test_df = pd.read_csv(config.kkbox.test_csv_fn)
    songs_df['is_test'] = songs_df['song_id'].isin(test_df['song_id'])

    songs_df = songs_df[(songs_df['train_count'] > 0) | (songs_df['is_test'] == True)]
    songs_df.to_csv(config.kkbox.songs_to_crawl_csv_fn, index=False)

    # Leave only the songs that have name and artist_name
    songs_df = songs_df.dropna(subset=['name'])
    songs_df = songs_df.dropna(subset=['artist_name'])


if __name__ == '__main__':
    main()