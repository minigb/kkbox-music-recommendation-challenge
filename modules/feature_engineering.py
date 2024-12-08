import pandas as pd

class FeatureEngineering:
    def __init__(self, data, config):
        self.data = data # interaction data
        self.config = config

    def run(self):
        for method_name in dir(self):
            if method_name.startswith("run_"):
                method = getattr(self, method_name)
                if callable(method):
                    method()
        return self.data
    
    def run_refine_composer(self):
        if not self.config.feature_engineering.run_composer:
            return
        column_name = 'composer'
        composers_list = [sorted(str(composers).split('| ')) for composers in self.data[column_name].tolist()]
        # max_num = max([len(composers) for composers in composers_list])
        max_num = self.config.feature_engineering.max_composers
        for i in range(max_num):
            self.data[f'{column_name}_{i}'] = [composers[i] if len(composers) > i else None for composers in composers_list]
        self.data.drop(columns=[column_name], inplace=True)
    
    def run_refine_lyricist(self):
        if not self.config.feature_engineering.run_lyricist:
            return
        column_name = 'lyricist'
        composers_list = [sorted(str(composers).split('| ')) for composers in self.data[column_name].tolist()]
        # max_num = max([len(composers) for composers in composers_list])
        max_num = self.config.feature_engineering.max_lyricist
        for i in range(max_num):
            self.data[f'{column_name}_{i}'] = [elem[i] if len(elem) > i else None for elem in composers_list]
        self.data.drop(columns=[column_name], inplace=True)

    def run_refine_genre_id(self):
        if not self.config.feature_engineering.run_genre_id:
            return
        column_name = 'genre_ids'
        genre_ids = [sorted(str(elem).split('|')) for elem in self.data[column_name].tolist()]
        # max_num = max([len(composers) for composers in genre_ids])
        max_num = self.config.feature_engineering.max_genre_ids
        for i in range(max_num):
            self.data[f'{column_name}_{i}'] = [elem[i] if len(elem) > i else None for elem in genre_ids]
        self.data.drop(columns=[column_name], inplace=True)

    def run_get_member_register_duration(self):
        if not self.config.feature_engineering.run_register_duration:
            return
        exp_col_df = pd.to_datetime(self.data['registration_init_time'], format='%Y%m%d', errors='coerce')
        reg_col_df = pd.to_datetime(self.data['expiration_date'], format='%Y%m%d', errors='coerce')
        self.data['member_duration'] = (exp_col_df - reg_col_df).dt.days.astype(int)