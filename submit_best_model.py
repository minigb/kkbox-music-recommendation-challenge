import os

import hydra
from utils import load_json
from pathlib import Path

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    best_model_json = load_json(config.best_model.json_path)
    best_model_output_dir = best_model_json['output_dir']
    date_from_output_dir = best_model_output_dir.split('/')[-1]

    submit_csv_path = Path(best_model_output_dir) / Path('submission.csv')
    assert submit_csv_path.exists(), f"submission.csv not found in {best_model_output_dir}"

    os.system(f"kaggle competitions submit -c {config.kaggle.competition} -f {submit_csv_path} -m 'Best model: {date_from_output_dir}'")

if __name__ == "__main__":
    main()