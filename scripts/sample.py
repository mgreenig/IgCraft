"""
Samples from a paired sequence generative model.
"""

import os
from pathlib import Path

import hydra
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
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
    Runs unconditional sampling and saves a FASTA file where VH/VL chains are
    separated by a ":" character.
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

    # Save the merged config (this just re-saves the input config if not checkpoint config was provided)
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.run.dir
    OmegaConf.save(cfg, os.path.join(run_dir, ".hydra", "config.yaml"))

    seed_everything(cfg.seed)

    logger.info(f"Sampling {cfg.num_samples} VH/VL pairs...")

    # Instantiate the sampler
    sampler = instantiate(cfg.sampler)

    # Run unconditional sampling
    with torch.no_grad():
        sequences = []
        for i in range(0, cfg.num_samples, cfg.batch_size):
            batch_size = min(cfg.batch_size, cfg.num_samples - i)
            sampled_batch = sampler(
                model, num_samples=batch_size, progress_bar=cfg.progress_bar
            )

            seqs = datamodule.data_to_sequences(sampled_batch, split_by_region=False)

            sequences.extend(seqs)

    logger.info(
        f"Saving sequences to FASTA file at {Path(cfg.out_dir) / 'samples.fasta'}..."
    )

    # Write the (paired) generated sequences to a FASTA file
    seq_records = [
        SeqRecord(Seq(seq), id=str(i), description="")
        for i, seq in enumerate(sequences, start=1)
    ]

    with open(Path(cfg.out_dir) / "samples.fasta", "w") as f:
        SeqIO.write(seq_records, f, "fasta")


if __name__ == "__main__":
    main()
