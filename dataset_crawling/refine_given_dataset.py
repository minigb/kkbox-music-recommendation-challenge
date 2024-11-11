import pandas as pd
import hydra

@hydra.main(config_path='../config', config_name='packed', version_base='1.1')
def main(config):
    songs_df = pd.read_csv(config.kkbox.songs_csv_fn)
    songs_extra_df = pd.read_csv(config.kkbox.song_extra_info_csv_fn)

    songs_df = songs_df.merge(songs_extra_df, on='song_id', how='left')
    songs_df.to_csv(config.kkbox.songs_merged_csv_fn)


if __name__ == '__main__':
    main()