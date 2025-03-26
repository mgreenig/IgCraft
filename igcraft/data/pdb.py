"""
Utilities for processing antibody PDB files.
"""

import gzip
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from Bio.Data.PDBData import protein_letters_3to1
from Bio import SeqIO
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import Residue
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from loguru import logger

from .antibody_utils import (
    anarci_number,
    find_chain_pairings,
    get_atom_mask,
    get_cropped_epitope,
    get_imgt_ptr,
    is_valid_residue,
)
from .constants import AA1_INDEX


def has_seqres(filepath: Path, gzipped: bool = False):
    """Checks if a PDB file has SEQRES records."""
    open_fn = gzip.open if gzipped else open
    with open_fn(filepath, "rt") as f:
        for line in f:
            if line.startswith("SEQRES"):
                return True
    return False


@dataclass
class AntibodyPDBData:
    """
    Data class to store data from an antibody PDB structure.

    :param pdb_id: The PDB ID.
    :param residues: A dictionary mapping chain IDs to a list of Residue objects.
    :param sequences: A dictionary mapping chain IDs to the AA sequence found in SEQRES (1 letter codes).
    :param atom_masks: A dictionary mapping chain IDs to a binary mask indicating which of the SEQRES
        residues are present in the ATOM records (i.e. which amino acids have structural data).
    :param numberings: A dictionary mapping chain IDs to a list of ((number, insertion code), AA) tuples.
    :param chain_types: A dictionary mapping chain IDs to the chain type (H/L/None for non-data).
    :param non_duplicate_chains: A set of chain IDs that are not duplicates.
    :param chain_pairings: A dictionary mapping each chain ID to the H/L paired chain ID. Chains with
        no pairing are not included.
    :param chain_species: A dictionary mapping antibody chain IDs to the species of the chain.
    """

    pdb_id: str
    residues: dict[str, list[Residue]]
    sequences: dict[str, str]
    atom_masks: dict[str, np.ndarray]
    numberings: dict[str, list[tuple[tuple[int, str], str]] | None]
    chain_types: dict[str, str | None]
    non_duplicate_chains: set[str | tuple[str, str]]
    chain_pairings: dict[str, str]
    chain_species: dict[str, str]

    @classmethod
    def from_pdb(cls, path: Path):
        """
        Loads the data from a SAbDab PDB file, returning an :code:`AntibodyPDBData` object.

        :param path: The path to the PDB file.
        :return: An data PDB data object.
        """
        try:
            if ".pdb" in str(path):
                parser = PDBParser(QUIET=True)
            elif ".cif" in str(path):
                parser = MMCIFParser(QUIET=True)
            else:
                raise ValueError("File must be a PDB or CIF file.")

            if path.suffix == ".gz":
                with gzip.open(path, "rt") as f:
                    structure = parser.get_structure(path.stem, f)

                    # Read the SEQRES directly if present, if not fill in via ATOM records
                    if has_seqres(path, gzipped=True):
                        seqres_records = list(SeqIO.parse(f, "pdb-seqres"))
                    else:
                        chain_seqs = {
                            chain.id: "".join(
                                protein_letters_3to1[res.get_resname()]
                                for res in chain
                                if is_aa(res, standard=True)
                            )
                            for chain in structure[0].get_chains()
                        }
                        seqres_records = [
                            SeqRecord(Seq(seq), annotations={"chain": chain})
                            for chain, seq in chain_seqs.items()
                        ]
            else:
                structure = parser.get_structure(path.stem, str(path))

                # Read the SEQRES directly if present, if not fill in via ATOM records
                if has_seqres(path):
                    seqres_records = list(SeqIO.parse(path, "pdb-seqres"))
                else:
                    chain_seqs = {
                        chain.id: "".join(
                            protein_letters_3to1[res.get_resname()]
                            for res in chain
                            if is_aa(res, standard=True)
                        )
                        for chain in structure[0].get_chains()
                    }
                    seqres_records = [
                        SeqRecord(Seq(seq), annotations={"chain": chain})
                        for chain, seq in chain_seqs.items()
                    ]

            # Use only the first model
            model = structure[0]

            residues = {}
            sequences = {
                record.annotations["chain"]: "".join(
                    [aa for aa in record.seq if aa in AA1_INDEX]
                )
                for record in seqres_records
            }
            atom_masks = {}
            numberings = {}
            chain_types = {}
            chain_species = {}
            non_duplicate_chains = set()
            non_duplicate_seqs = set()
            for chain in model.get_chains():

                chain_residues = [res for res in chain if is_valid_residue(res)]

                if len(chain_residues) == 0:
                    continue
                if chain.id not in sequences:
                    continue

                sequence = sequences[chain.id]
                mask = get_atom_mask(chain_residues, sequence)
                numbering, chain_type, species = anarci_number(sequence)

                # If the chain is not a data chain, save the sequence and continue
                if numbering is False:
                    residues[chain.id] = chain_residues
                    atom_masks[chain.id] = None  # No atom mask for epitope chains
                    numberings[chain.id] = None
                    chain_types[chain.id] = None
                    chain_species[chain.id] = None
                    continue

                # Identify the portion of the sequence that was alignable - save only that portion
                numbered_seq = "".join([aa for _, aa in numbering if aa != "-"])
                numbering_start_idx = sequence.find(numbered_seq)
                numbering_idx = slice(
                    numbering_start_idx, numbering_start_idx + len(numbered_seq)
                )
                if numbering_start_idx == -1:
                    continue

                numbered_mask = mask[numbering_idx]
                residues_start_idx = np.sum(mask[:numbering_start_idx])
                residues_end_idx = residues_start_idx + np.sum(numbered_mask)
                numbered_residues = chain_residues[residues_start_idx:residues_end_idx]

                # Only save the numbered region of the data
                sequences[chain.id] = numbered_seq
                residues[chain.id] = numbered_residues
                numberings[chain.id] = numbering
                chain_types[chain.id] = chain_type
                chain_species[chain.id] = species
                atom_masks[chain.id] = numbered_mask

                # To detect ScFV chains, try to number the remainder of the chain
                # (if it is at least 50 residues long)
                remaining_seq = sequence[numbering_start_idx + len(numbered_seq) :]
                remaining_residues = chain_residues[residues_end_idx:]
                if len(remaining_seq) > 50:
                    other_numbering, other_chain_type, other_species = anarci_number(
                        remaining_seq
                    )
                    if other_numbering is not False:

                        # Give the other "chain" a unique chain ID
                        other_chain_id = chain.id + "_"

                        other_numbered_seq = "".join(
                            [aa for _, aa in other_numbering if aa != "-"]
                        )
                        other_numbering_start_idx = remaining_seq.find(
                            other_numbered_seq
                        )
                        other_numbering_idx = slice(
                            other_numbering_start_idx,
                            other_numbering_start_idx + len(other_numbered_seq),
                        )

                        if other_numbering_start_idx == -1:
                            continue

                        # Get the atom mask for the remainder of the chain
                        remaining_mask = get_atom_mask(
                            remaining_residues, remaining_seq
                        )

                        other_numbered_mask = remaining_mask[other_numbering_idx]
                        other_residues_start_idx = np.sum(
                            remaining_mask[:other_numbering_start_idx]
                        )
                        other_residues_end_idx = other_numbering_start_idx + np.sum(
                            other_numbered_mask
                        )
                        other_numbered_residues = remaining_residues[
                            other_residues_start_idx:other_residues_end_idx
                        ]

                        sequences[other_chain_id] = other_numbered_seq
                        residues[other_chain_id] = other_numbered_residues
                        chain_types[other_chain_id] = other_chain_type
                        numberings[other_chain_id] = other_numbering
                        chain_species[other_chain_id] = other_species
                        atom_masks[other_chain_id] = other_numbered_mask

            # Find chain pairings using CA distances
            chain_pairings = find_chain_pairings(residues, chain_types)

            for chain, seq in sequences.items():
                if chain in chain_pairings:
                    if chain_types[chain] == "H":
                        H_seq = seq
                        L_seq = sequences[chain_pairings[chain]]
                        chain = (chain, chain_pairings[chain])
                    else:
                        H_seq = sequences[chain_pairings[chain]]
                        L_seq = seq
                        chain = (chain_pairings[chain], chain)

                    seq = f"{H_seq}-{L_seq}"

                if seq not in non_duplicate_seqs:
                    non_duplicate_chains.add(chain)
                    non_duplicate_seqs.add(seq)

            return cls(
                pdb_id=path.stem,
                residues=residues,
                sequences=sequences,
                atom_masks=atom_masks,
                numberings=numberings,
                chain_types=chain_types,
                non_duplicate_chains=non_duplicate_chains,
                chain_pairings=chain_pairings,
                chain_species=chain_species,
            )

        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return None

    def populate_hdf5_group(
        self, group: h5py.Group, residues: list[Residue], chain: str | None
    ):
        """
        Saves the data for a list of residues to a HDF5 group.

        :param group: The HDF5 group to save the data to.
        :param residues: The list of Residue objects for the chain.
        :param chain: The chain ID. Only needed if the residues correspond to an data chain.
        :return: None.
        """
        # If the chain is an data chain, save the sequence, atom mask, and IMGT numbering
        if chain in self.numberings:
            group["sequence"] = list(self.sequences[chain])
            group["species"] = self.chain_species[chain]
            group.create_dataset("imgt_ptr", data=get_imgt_ptr(self.numberings[chain]))

        # Otherwise just save the sequence from the atom records
        else:
            group["sequence"] = [
                protein_letters_3to1[res.get_resname()] for res in residues
            ]

        structure_group = group.create_group("structure")

        structure_group.create_dataset(
            "n", data=np.array([res["N"].get_coord() for res in residues])
        )
        structure_group.create_dataset(
            "ca", data=np.array([res["CA"].get_coord() for res in residues])
        )
        structure_group.create_dataset(
            "c", data=np.array([res["C"].get_coord() for res in residues])
        )

        if chain in self.numberings:
            structure_group.create_dataset("atom_mask", data=self.atom_masks[chain])

    def get_complex_residues(
        self, include_duplicates: bool = False
    ) -> dict[
        str | tuple[str, str], tuple[list[Residue], list[Residue], list[Residue]]
    ]:
        """
        Extracts a dictionary mapping antibody chain names to 3-tuples of residues
        for the VH, VL, and epitope regions. Chain names are either 2-tuples of the
        form (H, L) for paired VH/VL chains or single chain IDs for unpaired chains.

        :param include_duplicates: Whether to include duplicate chains.
        :return: A dictionary mapping chain names to 3-tuples of residues.
        """
        all_target_residues = [
            res
            for chain, residues in self.residues.items()
            if self.chain_types[chain] is None
            for res in residues
        ]

        # Get a dictionary containing paired VH/VL chains and a crop of the target containing the epitope
        processed_chains = set()
        complex_residues = {}
        for chain, residues in self.residues.items():
            if chain in processed_chains:
                continue

            if self.chain_types[chain]:
                if chain in self.chain_pairings:
                    pair_chain = self.chain_pairings[chain]
                    if self.chain_types[chain] == "H":
                        H_chain = chain
                        L_chain = pair_chain
                        H_residues = self.residues[chain]
                        L_residues = self.residues[pair_chain]
                    else:
                        H_chain = pair_chain
                        L_chain = chain
                        H_residues = self.residues[pair_chain]
                        L_residues = self.residues[chain]

                    if (
                        not include_duplicates
                        and (H_chain, L_chain) not in self.non_duplicate_chains
                    ):
                        continue

                    if all_target_residues:
                        epitope_residues = get_cropped_epitope(
                            H_residues + L_residues, all_target_residues
                        )
                    else:
                        epitope_residues = []

                    complex_residues[(H_chain, L_chain)] = (
                        H_residues,
                        L_residues,
                        epitope_residues,
                    )
                    processed_chains.add(pair_chain)
                    processed_chains.add(chain)
                else:
                    if (
                        not include_duplicates
                        and chain not in self.non_duplicate_chains
                    ):
                        continue

                    if all_target_residues:
                        epitope_residues = get_cropped_epitope(
                            residues, all_target_residues
                        )
                    else:
                        epitope_residues = []

                    if self.chain_types[chain] == "H":
                        complex_residues[chain] = (residues, [], epitope_residues)
                    else:
                        complex_residues[chain] = ([], residues, epitope_residues)

                    processed_chains.add(chain)

        return complex_residues

    def to_hdf5(self, group: h5py.Group):
        """
        Saves the data to a HDF5 group.

        :param group: The HDF5 group to save the data to.
        """
        complex_residues = self.get_complex_residues()

        # Write the data to the HDF5 file
        if self.pdb_id not in group:
            pdb_group = group.create_group(self.pdb_id)
        else:
            pdb_group = group[self.pdb_id]

        for chain, (vh_res, vl_res, epitope_res) in complex_residues.items():
            if isinstance(chain, tuple):
                h_chain, l_chain = chain
                chain = "-".join(chain)
            elif vh_res:
                h_chain = chain
                l_chain = None
            else:
                h_chain = None
                l_chain = chain

            if chain in pdb_group:
                continue

            chain_group = pdb_group.create_group(chain)

            if vh_res:
                vh_group = chain_group.create_group("vh")
                self.populate_hdf5_group(vh_group, vh_res, h_chain)
            if vl_res:
                vl_group = chain_group.create_group("vl")
                self.populate_hdf5_group(vl_group, vl_res, l_chain)
            if epitope_res:
                epitope_group = chain_group.create_group("epitope")
                self.populate_hdf5_group(epitope_group, epitope_res, None)
