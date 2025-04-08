"""
Runs antibody sequence numbering on an input FASTA file of paired antibody sequences,
returning a CSV file with the per-region sequences.

Usage:

```
python scripts/data/fasta_to_csv.py /path/to/my/fasta --cores 8 --outfile /path/to/output.csv
```
"""

import argparse
from collections import defaultdict
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from anarci import run_anarci
from Bio.SeqIO.FastaIO import SimpleFastaParser
from loguru import logger



def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta",
        type=str,
        help="Path to a FASTA file that will be converted into a CSV file.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=-1,
        help="Number of cores to use. -1 means all available.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="alignment.csv",
        help="Path to save the alignment results.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Number of sequences to process in each chunk.",
    )
    args = parser.parse_args()
    return args

def main():
    """
    Runs alignment on the data sequences and saves the per-region sequences into a CSV file,
    where different columns correspond to different regions (H-fwr1, H-cdr1, etc.).
    """
    args = parse_args()

    if args.cores == -1:
        ncpu = cpu_count()
    else:
        ncpu = args.cores

    logger.info(f"Running ANARCI alignment with {ncpu} cores.")
    with open(args.fasta) as file:
        input_sequences = list(SimpleFastaParser(file))
        vh_inputs = []
        vl_inputs = []
        for seq_id, sequence in input_sequences:
            if ":" not in sequence:
                logger.warning(f"Skipping sequence {seq_id} due to missing ':' separator between VH/VL chains.")
                continue

            vh, vl = sequence.split(":")
            vh_inputs.append((seq_id, vh))
            vl_inputs.append((seq_id, vl))

    # Residue numbers delineating the boundaries of the IMGT regions
    IMGT_REGION_BOUNDARIES = [27, 39, 56, 66, 105, 118]
    COLNAMES = [
        "fwr1_aa",
        "cdr1_aa",
        "fwr2_aa",
        "cdr2_aa",
        "fwr3_aa",
        "cdr3_aa",
        "fwr4_aa",
    ]

    vh_regions = defaultdict(list)
    vl_regions = defaultdict(list)

    def update_region_dict(numbered_sequences, region_dict):
        """Updates the input IMGT region sequence dictionary with the region sequences from the numbering."""
        for numberings in numbered_sequences:
            if numberings is None:
                continue

            # First alignment is the best hit according to E-value
            numbering, _, _ = numberings[0]
            numbers = np.array([num for (num, _), aa in numbering if aa != "-"])

            region_split_idx = np.searchsorted(numbers, IMGT_REGION_BOUNDARIES)
            numbered_seq = np.array([aa for (num, _), aa in numbering if aa != "-"])
            region_seqs = tuple(
                ["".join(seq) for seq in np.split(numbered_seq, region_split_idx)]
            )

            if len(region_seqs) != 7:
                continue

            for region, seq in zip(COLNAMES, region_seqs):
                region_dict[region].append(seq)

    for i in range(0, len(input_sequences), args.chunksize):
        vh_chunk = vh_inputs[i : i + args.chunksize]
        vh_sequences, vh_numbered_sequences, vh_alignment_details, _ = run_anarci(
            vh_chunk, scheme="imgt", ncpu=ncpu, output=False
        )

        vl_chunk = vl_inputs[i : i + args.chunksize]
        vl_sequences, vl_numbered_sequences, vl_alignment_details, _ = run_anarci(
            vl_chunk, scheme="imgt", ncpu=ncpu, output=False
        )

        logger.info(
            f"{(i + args.chunksize) * 100 / len(input_sequences):.2f}% of sequences aligned."
        )

        update_region_dict(vh_numbered_sequences, vh_regions)
        vh_regions["sequence_id"].extend([seq_id for seq_id, _ in vh_chunk])

        update_region_dict(vl_numbered_sequences, vl_regions)
        vl_regions["sequence_id"].extend([seq_id for seq_id, _ in vl_chunk])

    vh_df = pd.DataFrame(vh_regions)
    vl_df = pd.DataFrame(vl_regions)
    paired_df = vh_df.join(vl_df, on="sequence_id", lsuffix="_heavy", rsuffix="_light")
    paired_df.to_csv(args.outfile, index=False)
