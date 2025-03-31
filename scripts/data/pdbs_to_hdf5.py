"""
Reads the SAbDab database and saves the data in a HDF5 file.
"""

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multiprocessing import cpu_count

import h5py
import pandas as pd
from loguru import logger

from igcraft.data.pdb import AntibodyPDBData


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir",
        type=str,
        help="Path to the directory containing antibody PDB files.",
    )
    parser.add_argument(
        "--summary",
        "-s",
        type=str,
        default=None,
        help="Path to the SAbDab summary TSV file. Can be None if no duplicate filtering is required.",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        type=str,
        default="structures.hdf5",
        help="Path to save the HDF5 file.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=-1,
        help="Number of cores to use. -1 means all available.",
    )
    args = parser.parse_args()
    return args


def main():
    """Identifies unique data sequences from the SAbDab database and saves the data to a HDF5 file."""
    args = parse_args()

    if args.summary is not None:
        summary = pd.read_csv(args.summary, sep="\t")
        summary["resolution"] = (
            summary["resolution"].str.extract("([0-9.]+)").astype(float)
        )
    else:
        summary = None

    if args.cores == -1:
        ncpu = cpu_count()
    else:
        ncpu = args.cores

    logger.info(f"Reading in PDB files from {args.dir}...")

    pdb_files = [
        Path(args.dir) / p
        for p in os.listdir(args.dir)
        if p.endswith(".pdb") or p.endswith(".pdb.gz") or p.endswith(".cif") or p.endswith(".cif.gz")
    ]
    with ProcessPoolExecutor(ncpu) as executor:
        antibody_data = executor.map(AntibodyPDBData.from_pdb, pdb_files)

    logger.info(f"Loaded {len(pdb_files)} PDB files.")

    # Filter out None values and those without numbering, convert to dict
    antibody_data = [
        data
        for data in antibody_data
        if data is not None
        and any([numbering is not None for numbering in data.numberings.values()])
    ]

    # If a summary file is provided, filter for duplicates
    if summary is not None:

        logger.info("Identifying duplicate structures...")

        antibody_data = {data.pdb_id: data for data in antibody_data}
        sequence_pdb_ids = defaultdict(list)
        for pdb_id, data in antibody_data.items():
            for chain, seq in data.sequences.items():

                # Skip non-antibody chains
                if not data.chain_types.get(chain):
                    continue

                if chain in data.chain_pairings:

                    # If the chain is paired, skip any non-H chains to avoid double counting
                    if data.chain_types[chain] != "H":
                        continue

                    pair_seq = data.sequences[data.chain_pairings[chain]]
                    sequence_pdb_ids[(seq, pair_seq)].append(pdb_id)
                else:
                    sequence_pdb_ids[seq].append(pdb_id)

        # For duplicate sequences, pick the PDB with the lowest resolution
        pdb_resolutions = {
            row["pdb"]: row["resolution"] for _, row in summary.iterrows()
        }
        keep_antibodies = []
        for pdb_ids in sequence_pdb_ids.values():
            best_pdb_id = min(
                pdb_ids, key=lambda pdb_id: pdb_resolutions.get(pdb_id, 1000)
            )
            keep_antibodies.append(antibody_data[best_pdb_id])

    else:
        keep_antibodies = antibody_data

    logger.info(f"Found {len(keep_antibodies)} antibody structures. Writing to HDF5...")

    # Write to HDF5
    with h5py.File(args.outfile, "w") as file:
        for data in keep_antibodies:
            data.to_hdf5(file)

    logger.info(f"Data saved to {args.outfile}.")


if __name__ == "__main__":
    main()
