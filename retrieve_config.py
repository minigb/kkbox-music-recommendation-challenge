from utils import fetch_config_from_wandb
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import load_conf, save_conf

def retrieve_config(config):
    output_dir = Path(config.output.main_dir)
    for dir_name in tqdm(list(output_dir.iterdir())):
        if not dir_name.is_dir():
            continue

        wandb_runname = dir_name.name
        config.wandb.name = wandb_runname

        try:
            config_fetched = fetch_config_from_wandb(config.wandb)
        except ValueError as e:
            print(e)
            continue
        
        config_path = dir_name / 'config.yaml'
        save_conf(config_fetched, config_path)

def refine_dirs(current_config):
    # skip if it is already ran
    for existing_out_dirs in Path(current_config.output.main_dir).iterdir():
        if not existing_out_dirs.is_dir():
            continue
        config_path = existing_out_dirs / 'config.yaml'
        saved_config = load_conf(config_path)
        # saved_config = OmegaConf.create(config_path)
        saved_config_dict = OmegaConf.to_container(saved_config, resolve=True)
        if len(saved_config_dict) == 0:
            import shutil
            shutil.rmtree(existing_out_dirs)
        # is_same = True
        # for key in keys:
        #     if config_dict[key] != saved_config_dict[key]:
        #         is_same = False
        #         break
        # if is_same:
        #     continue



if __name__ == "__main__":
    config = OmegaConf.load('config.yaml')
    retrieve_config(config)

    refine_dirs(config)
