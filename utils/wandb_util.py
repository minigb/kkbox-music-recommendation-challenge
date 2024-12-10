import wandb
from omegaconf import OmegaConf


def fetch_config_from_wandb(wandb_config):
    # Initialize the wandb API
    api = wandb.Api()

    # Fetch all runs in the project
    runs = api.runs(path=f"{wandb_config.entity}/{wandb_config.project}")

    # Find the specific run by name
    for run in runs:
        if run.name == wandb_config.name:
            plain_config = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(run.config), resolve=True))
            return plain_config
        
    raise ValueError(f"Run with name '{wandb_config.name}' not found in project '{wandb_config.project}'.")