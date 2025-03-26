"""Model-related utilities."""

from copy import deepcopy

import hydra
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule

from .config import CheckpointSettings
from .datamodule import BaseDatamodule


def load_model(
    cfg: DictConfig,
    compile: bool = True,
) -> tuple[BaseDatamodule, LightningModule, DictConfig]:
    """
    Loads a model and its datamodule from a training or testing configuration
    with checkpointing information. The input config object needs to have a :code:`checkpoint`
    key containing the checkpoint information, with the field :code:`config_path`
    specifying the path to the configuration file and the field :code:`checkpoint_path`
    specifying the path to the checkpoint file (optional). When loaded, the config at
    :code:`config_path` should have a :code:`model` key with the same structure as :code:`ModelSettings`.

    The input config is used to override the configuration loaded from the config path in the checkpoint
    file, so additional arguments under the :code:`model` key can be provided to override the loaded
    model configuration.

    :param cfg: Configuration object containing a :code:`checkpoint` field.
    :param compile: Whether to compile the model after loading.
    :return: The datamodule and model loaded from the checkpoint, as well as the merged configuration.
    """
    if cfg.checkpoint is None:
        raise ValueError("A checkpoint must be provided to load a model.")

    checkpoint = CheckpointSettings(**cfg.checkpoint)

    if checkpoint.config_path is None:
        raise ValueError("A configuration file must be provided to load a model.")

    # Load the model configuration and merge with the existing configuration
    existing_cfg = OmegaConf.load(checkpoint.config_path)
    cfg = OmegaConf.merge(existing_cfg, cfg)

    # Load from the checkpoint file if it is provided
    if checkpoint.checkpoint_path is not None:

        # Detach the datamodule from the model config for loading
        model_cfg = deepcopy(cfg.model)
        datamodule_cfg = deepcopy(model_cfg.datamodule)
        del model_cfg.datamodule

        # Initialise the model and datamodule
        datamodule = hydra.utils.instantiate(datamodule_cfg)
        model = hydra.utils.instantiate(model_cfg)

        checkpoint = torch.load(checkpoint.checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if compile:
            model.compile_network()

    # Otherwise just initialise the model from scratch
    else:
        logger.warning("No checkpoint provided. Initialising model from scratch.")
        datamodule, model = init_model(cfg, compile=compile)

    return datamodule, model, cfg


def init_model(
    cfg: DictConfig,
    compile: bool = True,
) -> tuple[BaseDatamodule, LightningModule]:
    """
    Initialises a model and its datamodule from a train settings configuration.

    :param cfg: Configuration object for the run. This should have a key :code:`model`
        containing the model configuration with the structure specified in :code:`ModelSettings`.
    :param compile: Whether to compile the model after initialisation.
    :return: The datamodule and model loaded from the configuration.
    """
    # Detach the datamodule from the model config
    model_cfg = deepcopy(cfg.model)
    datamodule_cfg = deepcopy(model_cfg.datamodule)
    del model_cfg.datamodule

    # Initialise the datamodule
    datamodule = hydra.utils.instantiate(datamodule_cfg)

    # Initialise the model and run dummy forward pass to initialise model parameters
    model = hydra.utils.instantiate(model_cfg)

    # Compile the model's network
    if compile:
        model.compile_network()

    return datamodule, model


def get_next_run_name(entity: str, project: str, run_name: str) -> str:
    """
    For an input run name in the given entity/project, returns a unique run name
    by appending an integer to the run name.

    :param entity: The name of the entity on WandB - either your username or the ground organisation.
    :param project: The name of the project on WandB.
    :param run_name: The name of the run on WandB.
    :return: A unique run name that does not already exist in the project.
    """
    # Initialize a temporary run to query existing runs
    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project}")

    # Count existing runs that start with the base_name
    existing_runs = [run for run in runs if run.name.startswith(run_name)]

    # Get the maximum index of existing runs
    existing_run_idxs = []
    for run in existing_runs:
        if run.name.startswith(run_name):
            name_split = run.name.split("-")

            if len(name_split) == 1:
                continue

            if name_split[-1].isdigit():
                idx = int(name_split[-1])
                existing_run_idxs.append(idx)

    max_idx = max(existing_run_idxs) if existing_run_idxs else 0

    # Append an integer to the input run name
    new_run_name = f"{run_name}-{max_idx + 1}"
    return new_run_name
