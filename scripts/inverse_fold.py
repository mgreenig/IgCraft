"""
Performs inverse folding with IgCraft for an input set of PDB files.
"""

import os
from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from igcraft.data.pdb import AntibodyPDBData
from igcraft.model import load_model, init_model
from igcraft.model.datamodule import PairedStructureDatamodule

# Register a new resolver for summation in the config
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))


@hydra.main(version_base=None, config_path="../config", config_name="inverse_fold")
def main(cfg: DictConfig) -> None:
    """
    Runs inverse folding for an input set of PDB files.
    """

    # Load the model/datamodule
    if cfg.checkpoint.config_path is not None:
        logger.info(f"Loading model from {cfg.checkpoint.config_path}...")
        datamodule, model, cfg = load_model(cfg)
    else:
        logger.info("No checkpoint provided, testing randomly initialized model...")
        datamodule, model = init_model(cfg)

    # The datamodule must be a PairedStructureDatamodule
    if not isinstance(datamodule, PairedStructureDatamodule):
        raise ValueError(
            "The datamodule for inverse folding must be a PairedStructureDatamodule."
        )

    logger.info(f"Using {cfg.device} device for sampling...")
    model = model.to(cfg.device)
    model.eval()

    # Save the merged config
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.run.dir
    OmegaConf.save(cfg, os.path.join(run_dir, ".hydra", "config.yaml"))

    seed_everything(cfg.seed)

    # Check the PDB path is valid
    if cfg.pdb_path is None:
        raise ValueError("The pdb_path must be provided.")

    pdb_path = Path(cfg.pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Path {pdb_path} does not exist.")
    elif pdb_path.is_dir():
        pdb_files = list(pdb_path.glob("*.pdb")) + list(pdb_path.glob("*.cif"))
    else:
        pdb_files = [pdb_path]

    # Load in the chain map if not None
    if cfg.chain_map is not None:
        chain_map_df = pd.read_csv(cfg.chain_map, header=None)
        chain_map = defaultdict(set)
        for row in chain_map_df.values:
            if len(row) != 2:
                raise ValueError(
                    "Chain map file must have two columns, the first containing the PDB ID and the second "
                    "containing the chain ID (as a single string <VH>-<VL>)."
                )
            pdb_id, chain_id = row
            chain = tuple(chain_id.split("-"))
            if len(chain) != 2:
                raise ValueError(
                    "The chain ID in the chain_map must be provided as a single string <VH>-<VL>."
                )
            chain_map[pdb_id].add(chain)

    else:
        chain_map = None

    # Load the PDB data into a list
    ids = []
    dataset = []
    for file in pdb_files:
        pdb_data = AntibodyPDBData.from_pdb(file)

        if pdb_data is None:
            continue

        pdb_id = Path(file).stem
        if chain_map is not None and pdb_id in chain_map:
            keep_chains = chain_map[pdb_id]
        else:
            keep_chains = None

        data = datamodule.pdb_to_data(pdb_data, keep_chains=keep_chains)

        if not data:
            if keep_chains is not None:
                logger.warning(
                    f"Could not find the specified chains {list(keep_chains)} in the PDB file {file}."
                )
            else:
                logger.warning(f"Could not find any valid VH/VL pairs in the PDB file.")

        for (vh_id, vl_id), chain_data in data.items():
            for _ in range(cfg.n_sequences):
                ids.append((pdb_id, vh_id, vl_id))
                dataset.append(chain_data)

    # Pass the list directly to the datamodule
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False
    )

    logger.info(f"Inverse folding {len(dataset)} VH/VL pairs...")

    # Instantiate the sampler
    sampler = instantiate(cfg.sampler)

    # Get the batch sizes for the dataloader
    batch_sizes = [
        min(cfg.batch_size, len(dataset) - i)
        for i in range(0, len(dataset), cfg.batch_size)
    ]

    inverse_fold_sequences = defaultdict(list)

    # Add IDs
    for pdb_id, vh_id, vl_id in ids:
        inverse_fold_sequences[f"pdb_id"].append(pdb_id)
        inverse_fold_sequences[f"vh_id"].append(vh_id)
        inverse_fold_sequences[f"vl_id"].append(vl_id)

    # Sample sequences and add per-region
    for batch, batch_size in zip(dataloader, batch_sizes):
        batch = model.transfer_batch_to_device(batch, model.device, 0)
        seq_data, cond_data = batch

        cond_mask = datamodule.get_imgt_inpaint_mask(
            seq_data,
            ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"],
            batch_size,
        )

        with torch.no_grad():
            samples = sampler(
                model,
                cond_x=batch,
                cond_mask=cond_mask,
                fix_length=True,
                progress_bar=False,
            )

        # Extract the generated sequences corresponding to the ATOM records
        true_sequences = datamodule.data_to_sequences(seq_data, remove_gaps=False)
        pred_sequences = datamodule.data_to_sequences(samples, remove_gaps=False)

        for i in range(batch_size):
            wt_seq = true_sequences[i]
            pred_seq = pred_sequences[i]

            vh_region_atom_masks = {
                f"H-{region}": ~cond_data.vh.mask[i, idx].cpu().numpy()
                for region, idx in datamodule.vh_region_indices.items()
            }
            vl_region_atom_masks = {
                f"L-{region}": ~cond_data.vl.mask[i, idx].cpu().numpy()
                for region, idx in datamodule.vl_region_indices.items()
            }
            atom_masks = {**vh_region_atom_masks, **vl_region_atom_masks}

            for region, seq in pred_seq.items():
                mask = atom_masks[region]
                atom_seq = "".join([aa for j, aa in enumerate(seq) if mask[j]])
                inverse_fold_sequences[f"{region}_pred"].append(atom_seq)

            for region, seq in wt_seq.items():
                mask = atom_masks[region]
                atom_seq = "".join([aa for j, aa in enumerate(seq) if mask[j]])
                inverse_fold_sequences[f"{region}_wt"].append(atom_seq)

    # Save the inverse folded sequences to a CSV file
    inverse_fold_df = pd.DataFrame(inverse_fold_sequences)
    inverse_fold_df.to_csv(f"{cfg.out_dir}/inverse_folded_sequences.csv", index=False)

    logger.info(
        f"Run finished! Saved inverse folded sequences to {cfg.out_dir}/inverse_folded_sequences.csv."
    )


if __name__ == "__main__":
    main()
