"""
Performs inpainting with IgCraft for an input set of IMGT regions.
"""

import json
import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from igcraft.model import load_model, init_model

# Register a new resolver for summation in the config
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))


@hydra.main(version_base=None, config_path="../config")
def main(cfg: DictConfig) -> None:
    """
    Runs inpainting for an input set of sequences and IMGT regions, saving the
    per-region inpainted sequences to a JSON file.
    """

    # Load the model/datamodule
    if cfg.checkpoint.config_path is not None:
        logger.info(f"Loading model from {cfg.checkpoint.config_path}...")
        datamodule, model, cfg = load_model(cfg)
    else:
        logger.info("No checkpoint provided, testing randomly initialized model...")
        datamodule, model = init_model(cfg)

    logger.info(f"Using {cfg.device} device for sampling...")
    model = model.to(cfg.device)
    model.eval()

    # Save the merged config
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.run.dir
    OmegaConf.save(cfg, os.path.join(run_dir, ".hydra", "config.yaml"))

    seed_everything(cfg.seed)

    # Obtain a dataset from the datamodule
    dataset = datamodule.get_dataset(cfg.sequences_csv)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False
    )

    logger.info(f"Inpainting {len(dataset)} VH/VL pairs...")

    # Instantiate the sampler
    sampler = instantiate(cfg.sampler)

    # Get the batch sizes for the dataloader
    batch_sizes = [
        min(cfg.batch_size, len(dataset) - i)
        for i in range(0, len(dataset), cfg.batch_size)
    ]

    inpaint_sequences = []
    for batch, batch_size in zip(dataloader, batch_sizes):
        batch = model.transfer_batch_to_device(batch, model.device, 0)

        seq_data, _ = batch
        true_sequences = datamodule.data_to_sequences(seq_data)
        batch_inpaint_sequences = [{"true": seqs} for seqs in true_sequences]

        for region in cfg.regions:

            if region is None:
                region = []

            cond_mask = datamodule.get_imgt_inpaint_mask(
                seq_data, region, batch_size, reveal_pads=cfg.fix_length
            )

            with torch.no_grad():
                samples = sampler(
                    model,
                    cond_x=batch,
                    cond_mask=cond_mask,
                    fix_length=cfg.fix_length,
                    progress_bar=False,
                )

            sequences = datamodule.data_to_sequences(samples)

            if isinstance(region, str):
                region_key = region
            else:
                region_key = "/".join(region)

            for i, seqs in enumerate(sequences):
                batch_inpaint_sequences[i][f"{region_key}-inpaint"] = seqs

        inpaint_sequences.extend(batch_inpaint_sequences)

    # Write the inpainted IMGT regions to a JSON file
    with open(Path(cfg.out_dir) / "inpaint_sequences.json", "w") as f:
        json.dump(inpaint_sequences, f, indent=4)

    logger.info(
        f"Run finished! Saved inpainted sequences to {cfg.out_dir}/inpaint_sequences.json."
    )


if __name__ == "__main__":
    main()
