"""
Trains a paired sequence generative model.
"""

import hydra
import os
import pytorch_lightning as pl
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from igcraft.model import load_model, init_model, get_next_run_name

# Register a new resolver for summation in the config
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))


@hydra.main(version_base=None, config_path="../config")
def main(cfg: DictConfig) -> None:
    """Starts a training run."""
    pl.seed_everything(123)

    # Load the model/datamodule
    if cfg.checkpoint.config_path is not None:
        logger.info(
            f"Loading model from pre-existing config at {cfg.checkpoint.config_path}..."
        )
        datamodule, model, cfg = load_model(cfg)
    else:
        logger.info("Initialising model...")
        datamodule, model = init_model(cfg)

    # Save the merged config (this just re-saves the input config if not checkpoint config was provided)
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.run.dir
    OmegaConf.save(cfg, os.path.join(run_dir, ".hydra", "config.yaml"))

    # Get the next run name
    run_name = get_next_run_name(
        cfg.wandb.entity, cfg.wandb.project, cfg.wandb.run_name
    )
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Initialize WandB
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=run_name,
        tags=cfg.wandb.tags,
        config=config_dict,
    )

    # Load the trainer
    trainer = instantiate(cfg.trainer)

    # If a checkpoint was provided, pass to the trainer if not resetting optimizer
    if cfg.checkpoint.reset_optimizer:
        ckpt_path = None
    else:
        ckpt_path = cfg.checkpoint.checkpoint_path

    logger.info("Starting training...")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
