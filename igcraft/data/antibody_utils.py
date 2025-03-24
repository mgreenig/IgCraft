"""Utility functions related to antibodies."""

import numpy as np
from anarci import chain_type_to_class, number, run_anarci
from Bio.Align import PairwiseAligner
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1
from Bio.PDB.Residue import Residue


def anarci_number(
    sequence: str,
) -> tuple[list[tuple[tuple[int, str], str]] | bool, str | bool, str | bool]:
    """
    Uses ANARCI to number an antibody sequence returning a 3-tuple consisting of the numbering,
    chain type, and species. Returns False for all values if the sequence could not be numbered.
    """
    _, numberings, alignment_details, _ = run_anarci(sequence)

    if numberings[0]:
        numbering = numberings[0][0][0]
        details = alignment_details[0][0]
        chain_type = chain_type_to_class[details["chain_type"]]
        species = details["species"]
    else:
        numbering = False
        chain_type = False
        species = False

    return numbering, chain_type, species


def get_imgt_ptr(numbering: list[tuple[tuple[int, str], str]]) -> np.array:
    """
    Using an IMGT numbering of a data sequence, returns a pointer array containing the
    start index of each IMGT region. The pointer has length 6 (denoting the boundaries between
    the 7 IMGT regions).

    :param numbering: A list of ((number, insertion code), AA) tuples.
    :return: A numpy array of shape (6,) containing the boundary indices between the IMGT regions.
    """
    non_gap_numbers = np.array([number for (number, _), aa in numbering if aa != "-"])
    ptr = np.searchsorted(non_gap_numbers, [27, 39, 55, 66, 105, 118])

    return ptr


def split_imgt_regions(sequence: str) -> list[str]:
    """
    Splits an input VH or VL sequence into its seven constituent IMGT regions: FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4.

    :param sequence: The VH or VL sequence to split.
    :return: A list of the seven IMGT region sequences.
    """
    numbering, _ = number(sequence)
    ptr = get_imgt_ptr(numbering)

    regions = []
    start = 0
    for end in ptr:
        regions.append(sequence[start:end])
        start = end

    regions.append(sequence[start:])

    return regions


def find_chain_pairings(
    residues: dict[str, list[Residue]],
    chain_types: dict[str, str | None],
    contact_distance: float = 8.0,
) -> dict[str, str]:
    """
    For an input dictionary of per-chain residues possibly containing multiple H/L
    chain pairs, returns a dictionary mapping each chain ID to its paired chain ID.

    :param residues: A dictionary mapping chain IDs to a list of Residue objects.
    :param chain_types: A dictionary mapping chain IDs to the chain type (H/L/None for non-data).
    :param contact_distance: The maximum distance between C-alpha atoms to consider a contact.
    :return: A dictionary mapping each chain ID to its paired chain ID.
    """
    # dictionary mapping each chain to its potential partners
    chain_complementarity_map = {"H": ("L", "K"), "L": ("H",), "K": ("H",)}

    chain_pairings = {}
    for chain, chain_residues in residues.items():

        if chain in chain_pairings:
            continue

        # If the chain type is not recognised, assume it is not an data chain
        chain_type = chain_types[chain]
        if chain_type not in chain_complementarity_map:
            continue

        chain_ca = np.array(
            [res["CA"].get_coord() for res in chain_residues if "CA" in res]
        )

        max_contacts = 0
        paired_chain = None
        for other_chain, other_chain_residues in residues.items():
            other_chain_type = chain_types[other_chain]

            # Only consider other chains that are complementary
            if other_chain_type in chain_complementarity_map[chain_type]:
                other_chain_ca = np.array(
                    [
                        res["CA"].get_coord()
                        for res in other_chain_residues
                        if "CA" in res
                    ]
                )
                contacts = (
                    np.linalg.norm(
                        chain_ca[:, None, :] - other_chain_ca[None, :, :], axis=-1
                    )
                    < contact_distance
                )
                num_contacts = np.triu(contacts).sum()
                if num_contacts > max_contacts:
                    max_contacts = num_contacts
                    paired_chain = other_chain

        if paired_chain is not None:
            chain_pairings[chain] = paired_chain
            chain_pairings[paired_chain] = chain  # assume the pairing is mutual

    return chain_pairings


def is_valid_residue(residue: Residue) -> bool:
    """Checks if a residue is one of the standard 20 amino acids and has N/CA/C backbone coordinates."""
    return (
        is_aa(residue.get_resname(), standard=True)
        and "N" in residue
        and "CA" in residue
        and "C" in residue
    )


def get_atom_mask(residues: list[Residue], sequence: str) -> np.ndarray:
    """
    Creates a binary mask indicating which of the SEQRES residues are present in the ATOM records
    and contain N/CA/C atoms using a pairwise sequence alignment.

    :param residues: A list of Residue objects from the ATOM records.
    :param sequence: The corresponding SEQRES sequence.
    :return: A binary mask of the same length as the sequence indicating which residues
        are present in the ATOM records.
    """
    atom_sequence = "".join(protein_letters_3to1[res.get_resname()] for res in residues)
    aligner = PairwiseAligner()
    alignments = aligner.align(atom_sequence, sequence)
    alignment = alignments[0]
    mask = np.array([aa != "-" for aa in alignment[0]])
    return mask


def get_cropped_epitope(
    ab_residues: list[Residue], target_residues: list[Residue], crop_size: int = 128
) -> list[Residue]:
    """
    For an input list of data residues, crops the residues in the target to a fixed size, based on
    their minimum distance to the data.

    :param ab_residues: A list of Residue objects for the data chain(s).
    :param target_residues: A list of Residue objects for the target protein(s).
    :param crop_size: The number of residues to crop to.
    :return: A subset of the target residues corresponding to the closest :code:`crop_size`
        residues to the data.
    """
    ab_coords = np.array([res["CA"].get_coord() for res in ab_residues if "CA" in res])
    target_coords = np.array(
        [res["CA"].get_coord() for res in target_residues if "CA" in res]
    )
    distances = np.linalg.norm(
        target_coords[:, None, :] - ab_coords[None, :, :], axis=-1
    )
    min_distances = distances.min(axis=1)
    crop_indices = np.argsort(min_distances)[:crop_size]

    return [target_residues[i] for i in crop_indices]
