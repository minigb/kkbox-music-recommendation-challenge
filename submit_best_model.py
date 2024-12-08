import os
import hydra
from pathlib import Path

from modules.predict import run_inference
from utils import load_json


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    best_model_json = load_json(config.best_model.json_path)
    best_model_output_dir = best_model_json['output_dir']
    date_from_output_dir = best_model_output_dir.split('/')[-1]

    fn = config.output.submission_path.split('/')[-1]
    submit_csv_path = Path(best_model_output_dir) / fn
    if not submit_csv_path.exists():
        run_inference(config)
    assert submit_csv_path.exists(), f"submission.csv not found in {best_model_output_dir}"

    os.system(f"kaggle competitions submit -c kkbox-music-recommendation-challenge -f {submit_csv_path} -m 'Best model: {date_from_output_dir}'")

if __name__ == "__main__":
    main()