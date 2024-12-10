import pandas as pd

class FeatureEngineering:
    def __init__(self, data, config):
        self.data = data # user-item data
        self.config = config

    def _check_with_default_as_false(self, key):
        if key in self.config.feature_engineering.keys() and not (self.config.feature_engineering).to_dict()[key]:
            return False
        return True
    
    def _check_with_default_as_true(self, key):
        if key in self.config.feature_engineering.keys() and (self.config.feature_engineering).to_dict()[key]:
            return True
        return False

    def run(self):
        for method_name in dir(self):
            if method_name.startswith("run_"):
                method = getattr(self, method_name)
                if callable(method):
                    method()
        return self.data
    
    def run_refine_composer(self):
        if not self._check_with_default_as_false('run_composer'):
            return
        column_name = 'composer'
        if self._check_with_default_as_true('do_sort'):
            composers_list = [sorted(str(composers).split('| ')) for composers in self.data[column_name].tolist()]
        else: # default is to sort
            composers_list = [str(composers).split('| ') for composers in self.data[column_name].tolist()]
        # max_num = max([len(composers) for composers in composers_list])
        max_num = self.config.feature_engineering.max_composers
        for i in range(max_num):
            self.data[f'{column_name}_{i}'] = [composers[i] if len(composers) > i else None for composers in composers_list]
        self.data.drop(columns=[column_name], inplace=True)
    
    def run_refine_lyricist(self):
        if not self._check_with_default_as_false('run_lyricist'):
            return
        column_name = 'lyricist'
        if self._check_with_default_as_true('do_sort'):
            composers_list = [sorted(str(composers).split('| ')) for composers in self.data[column_name].tolist()]
        else:
            composers_list = [str(composers).split('| ') for composers in self.data[column_name].tolist()]
        # max_num = max([len(composers) for composers in composers_list])
        max_num = self.config.feature_engineering.max_lyricist
        for i in range(max_num):
            self.data[f'{column_name}_{i}'] = [elem[i] if len(elem) > i else None for elem in composers_list]
        self.data.drop(columns=[column_name], inplace=True)

    def run_refine_genre_id(self):
        if not self._check_with_default_as_false('run_genre_id'):
            return
        column_name = 'genre_ids'
        if self._check_with_default_as_true('do_sort'):
            genre_ids = [sorted(str(elem).split('|')) for elem in self.data[column_name].tolist()]
        else:
            genre_ids = [str(elem).split('|') for elem in self.data[column_name].tolist()]
        # max_num = max([len(composers) for composers in genre_ids])
        max_num = self.config.feature_engineering.max_genre_ids
        for i in range(max_num):
            self.data[f'{column_name}_{i}'] = [elem[i] if len(elem) > i else None for elem in genre_ids]
        self.data.drop(columns=[column_name], inplace=True)

    def run_get_member_register_duration(self):
        if not self._check_with_default_as_false('run_register_duration'):
            return
        column_name = 'member_duration'
        if column_name in self.data.columns:
            return
        exp_col_df = pd.to_datetime(self.data['registration_init_time'], format='%Y%m%d', errors='coerce')
        reg_col_df = pd.to_datetime(self.data['expiration_date'], format='%Y%m%d', errors='coerce')
        self.data[column_name] = (exp_col_df - reg_col_df).dt.days.astype(int)

    def run_set_numeric_features(self):
        # Run run_get_member_register_duration function first since it needs datetime info
        self.run_get_member_register_duration()
        self.data['registration_init_time'] = pd.to_datetime(self.data['registration_init_time'], format='%Y%m%d', errors='coerce')
        self.data['expiration_date'] = pd.to_datetime(self.data['expiration_date'], format='%Y%m%d', errors='coerce')

        numeric_columns = ['song_length', 'bd', 'registration_init_time', 'expiration_date']
        for column in numeric_columns:
            if column not in self.data.columns:
                self.data[column] = None
            self.data[column] = self.data[column].fillna(0).astype(int)