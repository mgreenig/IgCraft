"""
Performs inverse folding with IgCraft for an input set of PDB files.
"""

import os
from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from igcraft.data.constants import AA1_INDEX
from igcraft.data.pdb import AntibodyPDBData
from igcraft.model import load_model, init_model
from igcraft.model.datamodule import PairedStructureDatamodule

# Register a new resolver for summation in the config
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))


@hydra.main(version_base=None, config_path="../config", config_name="graft_cdrs")
def main(cfg: DictConfig) -> None:
    """
    Runs CDR grafting for an input set of PDB files.
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
        for (vh_id, vl_id), chain_data in data.items():
            for _ in range(cfg.n_sequences):
                ids.append((pdb_id, vh_id, vl_id))
                dataset.append(chain_data)

    # Pass the list directly to the datamodule
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False
    )

    logger.info(f"Grafting CDRs for {len(dataset)} VH/VL pairs...")

    # Instantiate the sampler
    sampler = instantiate(cfg.sampler)

    # Get the batch sizes for the dataloader
    batch_sizes = [
        min(cfg.batch_size, len(dataset) - i)
        for i in range(0, len(dataset), cfg.batch_size)
    ]

    grafted_sequences = defaultdict(list)

    # Add IDs
    for pdb_id, vh_id, vl_id in ids:
        grafted_sequences[f"pdb_id"].append(pdb_id)
        grafted_sequences[f"vh_id"].append(vh_id)
        grafted_sequences[f"vl_id"].append(vl_id)

    # Sample sequences and add per-region
    for batch, batch_size in zip(dataloader, batch_sizes):
        batch = model.transfer_batch_to_device(batch, model.device, 0)
        seq_data, cond_data = batch
        vh, vl = seq_data

        # Condition on the sequence of the CDRs
        fwr_regions = ["fwr1", "fwr2", "fwr3", "fwr4"]
        cond_mask = datamodule.get_imgt_inpaint_mask(
            seq_data,
            fwr_regions,
            batch_size,
            reveal_pads=cfg.cdr_pad_length > 0
            or (cfg.use_cdr_structure and cfg.use_fwr_structure),
        )  # only reveal pads if we are padding the CDRs or exposing the whole structure
        vh_cond_mask, vl_cond_mask = cond_mask

        vh_imgt_idx = datamodule.vh_region_indices
        vl_imgt_idx = datamodule.vl_region_indices

        # Mask denoting padding for the CDRs
        vh_pad_mask = torch.zeros(
            vh.shape, device=vh_cond_mask.device, dtype=torch.bool
        )
        vl_pad_mask = torch.zeros(
            vl.shape, device=vl_cond_mask.device, dtype=torch.bool
        )

        # Expose padded elements from the framework regions if specified
        if cfg.cdr_pad_length > 0:
            for i, (vh_seq, vl_seq) in enumerate(zip(vh, vl)):
                vh_non_pad_mask = vh_seq != AA1_INDEX["-"]
                vl_non_pad_mask = vl_seq != AA1_INDEX["-"]
                for region_num, region in enumerate(fwr_regions):
                    vh_region_mask = torch.zeros(
                        vh_seq.shape, device=vh_seq.device, dtype=torch.bool
                    )
                    vh_region_mask[vh_imgt_idx[region]] = True
                    vl_region_mask = torch.zeros(
                        vl_seq.shape, device=vl_seq.device, dtype=torch.bool
                    )
                    vl_region_mask[vl_imgt_idx[region]] = True
                    vh_region_idx = torch.nonzero(
                        vh_region_mask & vh_non_pad_mask
                    ).squeeze(-1)
                    vl_region_idx = torch.nonzero(
                        vl_region_mask & vl_non_pad_mask
                    ).squeeze(-1)

                    # If not on the first region, perform left-padding
                    if region_num > 0:
                        vh_pad_mask[i, vh_region_idx[: cfg.cdr_pad_length]] = True
                        vl_pad_mask[i, vl_region_idx[: cfg.cdr_pad_length]] = True

                    # If not on the last region, perform right-padding
                    if region_num < len(fwr_regions) - 1:
                        vh_pad_mask[i, vh_region_idx[-cfg.cdr_pad_length :]] = True
                        vl_pad_mask[i, vl_region_idx[-cfg.cdr_pad_length :]] = True

            # Unmask the padded CDR regions
            vh_cond_mask[vh_pad_mask] = True
            vl_cond_mask[vl_pad_mask] = True

        cond_mask = (vh_cond_mask, vl_cond_mask)

        # Mask the framework and CDR regions from the structure encoder if specified
        if not cfg.use_fwr_structure:
            vh_fwr_mask = torch.zeros(
                (datamodule.vh_length,), device=model.device, dtype=torch.bool
            )
            vl_fwr_mask = torch.zeros(
                (datamodule.vl_length,), device=model.device, dtype=torch.bool
            )

            for region in fwr_regions:
                # Mask the framework regions from the structure encoder
                vh_fwr_mask[..., vh_imgt_idx[region]] = True
                vl_fwr_mask[..., vl_imgt_idx[region]] = True

            cond_data.vh.mask |= vh_fwr_mask[None]
            cond_data.vl.mask |= vl_fwr_mask[None]

        if not cfg.use_cdr_structure:
            vh_cdr_mask = torch.zeros(
                (datamodule.vh_length,), device=model.device, dtype=torch.bool
            )
            vl_cdr_mask = torch.zeros(
                (datamodule.vl_length,), device=model.device, dtype=torch.bool
            )

            for region in ["cdr1", "cdr2", "cdr3"]:
                # Mask the cdr regions from the structure encoder
                vh_cdr_mask[..., vh_imgt_idx[region]] = True
                vl_cdr_mask[..., vl_imgt_idx[region]] = True

            cond_data.vh.mask |= vh_cdr_mask[None]
            cond_data.vl.mask |= vl_cdr_mask[None]

        # If no structure is used, pass None to the model
        if not (cfg.use_cdr_structure or cfg.use_fwr_structure):
            batch = (seq_data, None)

        with torch.no_grad():
            samples = sampler(
                model,
                cond_x=batch,
                cond_mask=cond_mask,
                fix_length=False,
                progress_bar=False,
            )

        # Extract the generated sequences corresponding to the ATOM records
        true_sequences = datamodule.data_to_sequences(seq_data)
        pred_sequences = datamodule.data_to_sequences(samples)

    # Save the grafted sequences to a CSV file
    # Save the grafted sequences to CSV/FASTA files
    sequence_dict = defaultdict(list)

    for true_seq, pred_seq in zip(true_sequences, pred_sequences):
        for region, true_region_seq in true_seq.items():
            sequence_dict[f"{region}_wt"].append(true_region_seq)

            # If the structure was provided for the region, we are
            # constrained to match exactly the length of the WT sequence
            # (to be compatible with the input structure)
            if ("cdr" in region and cfg.use_cdr_structure) or (
                "fwr" in region and cfg.use_fwr_structure
            ):
                # Pad with the true amino acids if shorter
                if len(pred_seq[region]) < len(true_region_seq):
                    pred_seq[region] = (
                        pred_seq[region] + true_region_seq[len(pred_seq[region]) :]
                    )

                # Clip the sequence to the length of the WT sequence if longer
                if len(pred_seq[region]) > len(true_region_seq):
                    pred_seq[region] = pred_seq[region][: len(true_region_seq)]

            sequence_dict[f"{region}_pred"].append(pred_seq[region])

    # Save the grafted sequences to a CSV file
    grafted_sequence_df = pd.DataFrame(sequence_dict)
    grafted_sequence_df["grafted_vh"] = grafted_sequence_df[
        [
            "H-fwr1_pred",
            "H-cdr1_wt",
            "H-fwr2_pred",
            "H-cdr2_wt",
            "H-fwr3_pred",
            "H-cdr3_wt",
            "H-fwr4_pred",
        ]
    ].apply(lambda x: "".join(x), axis=1)
    grafted_sequence_df["grafted_vl"] = grafted_sequence_df[
        [
            "L-fwr1_pred",
            "L-cdr1_wt",
            "L-fwr2_pred",
            "L-cdr2_wt",
            "L-fwr3_pred",
            "L-cdr3_wt",
            "L-fwr4_pred",
        ]
    ].apply(lambda x: "".join(x), axis=1)
    grafted_sequence_df.to_csv(f"{cfg.out_dir}/grafted_sequences.csv", index=False)

    # Save the grafted sequences to a FASTA file
    records = []
    for (pdb_id, vh_id, vl_id), df in grafted_sequence_df.groupby(
        ["pdb_id", "vh_id", "vl_id"]
    ):
        for i, row in df.reset_index().iterrows():
            seq = Seq(f"{row['grafted_vh']}:{row['grafted_vl']}")
            seq_id = f"{pdb_id}_{vh_id}_{vl_id}_{i}"
            record = SeqRecord(seq, id=seq_id, description="")
            records.append(record)

    with open(f"{cfg.out_dir}/grafted_sequences.fasta", "w") as f:
        SeqIO.write(records, f, "fasta")

    logger.info(f"Run finished! Saved generated framework sequences to {cfg.out_dir}.")


if __name__ == "__main__":
    main()
