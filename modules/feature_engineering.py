import pandas as pd

class FeatureEngineering:
    def __init__(self, data, config):
        self.data = data # interaction data
        self.config = config
    
    def refine_composer(self):
        composers_list = [sorted(str(composers).split('| ')) for composers in self.data['composer'].tolist()]
        # max_composers = max([len(composers) for composers in composers_list])
        max_composers = self.config.feature_engineering.max_composers
        for i in range(max_composers):
            self.data[f'composer_{i}'] = [composers[i] if len(composers) > i else None for composers in composers_list]
        self.data.drop(columns=['composer'], inplace=True)

    def run(self):
        self.refine_composer()
        return self.data