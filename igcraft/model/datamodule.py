"""
Contains the datamodule class the that loads the paired heavy / light chain or nanobody sequences.
"""

from abc import abstractmethod
from os.path import split
from typing import Any, Generic, Sequence, TypeVar

import h5py
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from Bio.Data.PDBData import protein_letters_3to1
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningDataModule

from ..data.antibody_utils import get_imgt_ptr, split_imgt_regions
from ..data.constants import AA1_LETTERS, AA1_INDEX, IMGT_REGIONS
from ..data.pdb import AntibodyPDBData
from .config import (
    PairedSequenceDatamoduleConfig,
    PairedStructureDatamoduleConfig,
    UnpairedSequenceDatamoduleConfig,
)


def seq_batch_to_aa1(seq_batch: torch.Tensor, remove_gaps: bool = True) -> list[str]:
    """
    Converts a batch of sequences from integer encoding to one-letter amino acid codes.

    :param seq_batch: A tensor of shape (N, L) containing sequences encoded as integers 0-20.
    :param remove_gaps: Whether to remove gaps from the sequences.
    :return: A list of sequences as strings.
    """
    sequences = ["".join([AA1_LETTERS[i.item()] for i in seq]) for seq in seq_batch]

    if not remove_gaps:
        return sequences

    return [seq.replace("-", "") for seq in sequences]


def pad_structure_frames(
    frames: tuple[np.ndarray, np.ndarray], mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pads the rotation and translation frames of a protein structure using an input
    mask specifying which positions should be filled in the output arrays. The mask
    should have the same number of elements :code:`True` as there are frames.

    :param frames: A tuple containing the rotation and translation arrays.
    :param mask: A 1-D mask tensor specifying the positions to fill in the output array.
    :return: A tuple of the padded rotation and translation arrays, each of the same
        length as the input mask, with masked elements filled with the input frames
        and other elements set to zero.
    """

    rot, trans = frames
    pad_rots = np.zeros((mask.shape[0], 3, 3), dtype=rot.dtype)
    pad_trans = np.zeros((mask.shape[0], 3), dtype=trans.dtype)
    pad_rots[mask] = rot
    pad_trans[mask] = trans
    return pad_rots, pad_trans


class IMGTTokenizer:
    """
    A tokenizer that encodes amino acid sequences for the different IMGT
    regions of an data chain as integers.
    """

    def __init__(self, region_lengths: dict[str, int]):
        """
        :param region_lengths: The maximum lengths for each IMGT region of the input sequences.
        """
        if not all([region in region_lengths for region in IMGT_REGIONS]):
            missing_regions = [
                region for region in IMGT_REGIONS if region not in region_lengths
            ]
            raise ValueError(
                f"All IMGT regions should be found as keys in the region_lengths dictionary. "
                f"The following were not found: {missing_regions}."
            )

        self.region_lengths = region_lengths

    def encode_from_raw(self, sequence: str) -> torch.Tensor:
        """
        Encodes a raw amino acid sequence as integers, splitting it into IMGT regions first
        and then padding each region to its maximum length.

        :param sequence: The raw amino acid sequence.
        :return: A tensor containing the encoded sequence.
        """
        regions = split_imgt_regions(sequence)
        return self.encode(regions)

    def pad_atom_mask(self, mask: np.ndarray, imgt_ptr: np.ndarray) -> np.ndarray:
        """
        Pads an atom mask to the full sequence length by adding :code:`False`
        elements to the end of each IMGT region.

        :param mask: The 1-D atom mask to pad.
        :param imgt_ptr: The IMGT region pointer, giving the 6 boundaries between the 7 IMGT regions.
        :return: The padded atom mask of the full sequence length.
        """
        return np.concatenate(
            [
                np.pad(m, (0, self.region_lengths[region] - len(m)))
                for m, region in zip(np.split(mask, imgt_ptr), IMGT_REGIONS)
            ]
        )

    def encode(self, imgt_sequences: Sequence[str]) -> torch.Tensor:
        """
        Encodes a list of IMGT region sequences as integers.

        :param imgt_sequences: A sequence of amino acid sequences for the different IMGT regions.
        :return: A tensor containing the encoded sequences.
        """
        # pad the sequences per-region and then concatenate them
        full_sequence = []
        for i, region in enumerate(IMGT_REGIONS):
            region_seq = imgt_sequences[i] + "-" * (
                self.region_lengths[region] - len(imgt_sequences[i])
            )
            full_sequence.extend(region_seq)

        return torch.as_tensor(
            [AA1_INDEX.get(aa, AA1_INDEX["-"]) for aa in full_sequence]
        )


class UnpairedSequenceDataset(Dataset):
    """
    A dataset that reads unpaired data sequences from a CSV file and returns them as tensors encoding
    amino acids as integers 0-20 (20 for gaps).
    """

    def __init__(self, path: str, colnames: list[str], tokenizer: IMGTTokenizer):
        """
        :param path: The path to the CSV or parquet file containing the sequences.
        :param colnames: Seven column names for the sequences of different IMGT regions on the chain,
            in the following order: FWR1, CDR1, FWR2, CDR2, FWR3, CDR3, FWR4.
        """
        if path.endswith(".pqt"):
            self._data = pl.read_parquet(path)
        elif path.endswith(".csv"):
            self._data = pl.read_csv(path)
        else:
            raise ValueError("Input file must be a .csv or .pqt file.")

        if len(colnames) != 7:
            raise ValueError("Column names must have length 7, for the 7 IMGT regions.")

        self.colnames = colnames
        self.tokenizer = tokenizer

        self._data = self._data.select(self.colnames)

    def __len__(self):
        """The number of paired sequences in the dataset."""
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """
        :param idx: The index of the sequence pair to retrieve.
        :return: A tensor containing the encoded sequence and an empty dictionary (no conditioning data).
        """
        return self.tokenizer.encode(self._data.row(idx)), {}


class PairedSequenceDataset(Dataset):
    """
    A dataset that reads paired VH/VL data sequences from a CSV file and returns them as tensors encoding
    amino acids as integers 0-20 (20 for gaps).
    """

    def __init__(
        self,
        path: str,
        vh_colnames: list[str],
        vl_colnames: list[str],
        vh_tokenizer: IMGTTokenizer,
        vl_tokenizer: IMGTTokenizer,
    ):
        """
        :param path: The path to the CSV or parquet file containing the sequences.
        :param vh_colnames: Seven column names for the sequences of different IMGT regions on the VH, in the following order:
            FWR1, CDR1, FWR2, CDR2, FWR3, CDR3, FWR4.
        :param vl_colnames: Seven column names for th sequences of different IMGT regions on the VL, in the following order:
            FWR1, CDR1, FWR2, CDR2, FWR3, CDR3, FWR4.
        :param vh_tokenizer: The tokenizer to use for encoding the VH sequences.
        :param vl_tokenizer: The tokenizer to use for encoding the VL sequences.
        """
        if path.endswith(".pqt"):
            self._data = pl.read_parquet(path)
        elif path.endswith(".csv"):
            self._data = pl.read_csv(path)
        else:
            raise ValueError("Input file must be a .csv or .pqt file.")

        if len(vh_colnames) != 7:
            raise ValueError(
                "VH column names must have length 7, for the 7 IMGT regions."
            )

        if len(vl_colnames) != 7:
            raise ValueError(
                "VL column names must have length 7, for the 7 IMGT regions."
            )

        self.vh_colnames = vh_colnames
        self.vl_colnames = vl_colnames
        self.vh_tokenizer = vh_tokenizer
        self.vl_tokenizer = vl_tokenizer

        self._vh_data = self._data.select(self.vh_colnames)
        self._vl_data = self._data.select(self.vl_colnames)

    def __len__(self):
        """The number of paired sequences in the dataset."""
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], dict]:
        """
        :param idx: The index of the sequence pair to retrieve.
        :return: A tuple of the two encoded sequences, the first for the VH and the second for the VL,
            and an empty dictionary as a placeholder for conditioning data.
        """
        vh = self.vh_tokenizer.encode(self._vh_data.row(idx))
        vl = self.vl_tokenizer.encode(self._vl_data.row(idx))

        return (vh, vl), {}


def backbone_to_frames(
    n: np.ndarray, ca: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts the backbone coordinates of a protein to a set of frames, where each frame is represented
    as a rotation (giving the orientation of the backbone) and a translation (giving the position of the CA).
    Same method as used in AlphaFold2.

    :param n: N atom coordinates as an (N, 3) array.
    :param ca: CA atom coordinates as an (N, 3) array.
    :param c: C atom coordinates as an (N, 3) array.
    :return: A tuple of two tensors, the first containing a (N, 3, 3) array of rotation matrices
        and the second containing a (N, 3) array of translation vectors.
    """
    v1 = c - ca
    v2 = n - ca

    e1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    u2 = v2 - e1 * np.sum(v2 * e1, axis=-1, keepdims=True)
    e2 = u2 / np.linalg.norm(u2, axis=-1, keepdims=True)
    e3 = np.linalg.cross(e1, e2)

    rotation = np.stack([e1, e2, e3], axis=-1)
    translation = ca

    return rotation, translation


class PairedStructureDataset(Dataset):
    """
    A dataset that reads paired VH/VL data structures from a HDF5 file and returns the underlying
    HDF5 groups. The key structure of the input HDF5 should be:
        - pdb_id: The PDB ID of the structure.
            - chain_id: The concatenated chain names: <VH>-<VL>.
                - :code:`vh_key`: The VH chain structure.
                - :code:`vl_key`: The VL chain structure.
                - :code:`epitope_key`: The epitope structure. (Optional)

    The structure groups (VH, VL, epitope) are expected to contain the following datasets:
        - :code:`n`: The N coordinates of the structure.
        - :code:`ca`: The CA coordinates of the structure.
        - :code:`c`: The C coordinates of the structure.
        - :code:`sequence`: The amino acid sequence of the structure (from SEQRES).

    And the VH/VL groups must contain the following:
        - :code:`imgt_ptr`: A pointer array specifying the boundaries of the IMGT regions in the sequence.
        - :code:`atom_mask`: A boolean mask with length equal to that of the sequence
            specifying for which residues the N/CA/C coordinates are present.
    """

    def __init__(
        self,
        path: str,
        vh_tokenizer: IMGTTokenizer,
        vl_tokenizer: IMGTTokenizer,
        max_epitope_length: int,
        noise_structures: bool = False,
        vh_key: str = "vh",
        vl_key: str = "vl",
        epitope_key: str = "epitope",
    ):
        """
        :param path: The path to the HDF5 file containing the paired structures.
        :param vh_tokenizer: The tokenizer to use for encoding the VH sequences.
        :param vl_tokenizer: The tokenizer to use for encoding the VL sequences.
        :param max_epitope_length: The maximum length of the epitope sequence.
        :param noise_structures: Whether to add gaussian noise (sigma=0.02) to the structure coordinates.
        :param vh_key: The key for the VH chain structure in each VH/VL group of the HDF5 file.
        :param vl_key: The key for the VL chain structure in each VH/VL group of the HDF5 file.
        :param epitope_key: The key for the epitope structure in each VH/VL group of the HDF5 file.
        """
        self._path = path
        self.vh_tokenizer = vh_tokenizer
        self.vl_tokenizer = vl_tokenizer
        self.max_epitope_length = max_epitope_length
        self.vh_key = vh_key
        self.vl_key = vl_key
        self.epitope_key = epitope_key
        self.noise_structures = noise_structures

        # Extract the names of all the paired structures
        with h5py.File(self._path) as f:
            self._names = [f"/{pdb_id}/{chain}" for pdb_id in f for chain in f[pdb_id]]

    def __len__(self) -> int:
        """The number of paired structure examples."""
        return len(self._names)

    def _extract_ab_data(
        self, group: h5py.Group, tokenizer: IMGTTokenizer
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Extracts antibody data from an HDF5 group, returning a tuple of the encoded
        sequence and a dictionary of structure data.

        :param group: The HDF5 group containing the antibody data.
        :param tokenizer: The tokenizer to use for encoding the sequence.
        :return: A tuple of the encoded sequence and a dictionary of structure data.
        """
        seq = group["sequence"][:].astype(str)
        imgt_ptr = group["imgt_ptr"][:]

        region_seqs = ["".join(seq) for seq in np.split(seq, imgt_ptr)]
        seq_encoded = tokenizer.encode(region_seqs)

        structure = group["structure"]
        n, ca, c = structure["n"][:], structure["ca"][:], structure["c"][:]

        if self.noise_structures:
            n += np.random.normal(0, 0.02, n.shape)
            ca += np.random.normal(0, 0.02, ca.shape)
            c += np.random.normal(0, 0.02, c.shape)

        frames = backbone_to_frames(n, ca, c)
        atom_mask = tokenizer.pad_atom_mask(structure["atom_mask"][:], imgt_ptr)
        rots, trans = pad_structure_frames(frames, atom_mask)

        structure_data = {
            "frames": (torch.as_tensor(rots), torch.as_tensor(trans)),
            "mask": torch.as_tensor(
                ~atom_mask
            ),  # Invert the mask to make it a padding mask
        }

        return seq_encoded, structure_data

    def _extract_epitope_data(self, group: h5py.Group) -> dict[str, Any]:
        """
        Extracts epitope data from an HDF5 group, returning a dictionary of structure data.

        :param group: The HDF5 group containing the epitope data.
        :return: A dictionary of structure data.
        """
        # One-hot encode the epitope sequence
        epitope_seq = torch.as_tensor(
            [
                AA1_INDEX.get(aa, AA1_INDEX["-"])
                for aa in group["sequence"][:].astype(str)
            ]
        )
        epitope_features = F.one_hot(epitope_seq, num_classes=len(AA1_INDEX)).float()

        # Mask denoting which epitope residues are present
        valid_mask = np.concatenate(
            [
                np.ones(epitope_seq.shape[0], dtype=np.bool),
                np.zeros(
                    self.max_epitope_length - epitope_seq.shape[0],
                    dtype=np.bool,
                ),
            ],
            axis=0,
        )

        epitope_structure = group["structure"]
        n, ca, c = (
            epitope_structure["n"][:],
            epitope_structure["ca"][:],
            epitope_structure["c"][:],
        )

        if self.noise_structures:
            n += np.random.normal(0, 0.02, n.shape)
            ca += np.random.normal(0, 0.02, ca.shape)
            c += np.random.normal(0, 0.02, c.shape)

        epitope_frames = backbone_to_frames(n, ca, c)
        epitope_rots, epitope_trans = pad_structure_frames(epitope_frames, valid_mask)

        structure_data = {
            "frames": (
                torch.as_tensor(epitope_rots),
                torch.as_tensor(epitope_trans),
            ),
            "mask": torch.as_tensor(
                ~valid_mask
            ),  # Invert to convert into a padding mask
            "features": F.pad(
                epitope_features,
                (0, 0, 0, self.max_epitope_length - epitope_seq.shape[0]),
            ),
        }

        return structure_data

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
        """
        Extracts the VH, VL, and epitope sequences and structures for a given index.

        The returning dictionary has the following key/value structure
        - "vh": A dictionary containing the VH chain structure data.
            - "frames": A 2-tuple of tensors containing the rotation and translation frames.
            - "mask": A mask tensor denoting which tokens have structural data.
        - "vl": A dictionary containing the VL chain structure data.
            - "frames": A 2-tuple of tensors containing the rotation and translation frames.
            - "mask": A mask tensor denoting which tokens have structural data.
        - "epitope": A dictionary containing the epitope structure data.
            - "frames": A 2-tuple of tensors containing the rotation and translation frames.
            - "mask": A mask tensor denoting which tokens have structural data.
            - "features": A one-hot encoded tensor of the epitope sequence.

        :param idx: The index of the sample to extract.
        :return: A 2-tuple of data consisting of the tokenized VH/VL sequences
            and a dictionary with keys "vh", "vl", and "epitope", each containing
            a dictionary of structure data.
        """
        with h5py.File(self._path) as f:
            name = self._names[idx]
            vh = f[f"{name}/{self.vh_key}"]
            vl = f[f"{name}/{self.vl_key}"]

            vh_seq_encoded, vh_structure_data = self._extract_ab_data(
                vh, self.vh_tokenizer
            )
            vl_seq_encoded, vl_structure_data = self._extract_ab_data(
                vl, self.vl_tokenizer
            )

            structure_data = {"vh": vh_structure_data, "vl": vl_structure_data}

            if "epitope" in f[name]:
                epitope = f[f"{name}/{self.epitope_key}"]
                structure_data["epitope"] = self._extract_epitope_data(epitope)
            else:
                structure_data["epitope"] = {
                    "frames": (
                        torch.zeros((self.max_epitope_length, 3, 3)),
                        torch.zeros((self.max_epitope_length, 3)),
                    ),
                    "mask": torch.ones(self.max_epitope_length, dtype=torch.bool),
                    "features": torch.zeros((self.max_epitope_length, len(AA1_INDEX))),
                }

            return (vh_seq_encoded, vl_seq_encoded), structure_data


DatasetType = TypeVar("DatasetType", bound=Dataset)


class BaseDatamodule(LightningDataModule, Generic[DatasetType]):
    """A base class for the antibody paired/unpaired data modules."""

    @abstractmethod
    def get_dataset(self, path: str) -> DatasetType | None:
        """
        Returns the dataset for the given path.

        :param path: The path to the dataset.
        :return: The dataset.
        """
        pass

    @abstractmethod
    def get_imgt_inpaint_mask(
        self,
        sequences: Any,
        regions: str | list[str],
        batch_size: int,
        reveal_pads: bool = True,
    ) -> Any:
        """
        Returns an inpainting mask tensor for the given IMGT region(s), specifying
        which positions are to be conditioned on during sampling (i.e. all other regions).

        :param sequences: The sequence tensors to create the mask for.
        :param regions: The IMGT region(s) to create the mask for. Can be a single
            string or a list of regions.
        :param batch_size: The batch size for the mask.
        :param reveal_pads: Whether to unmask the length of the regions being inpainted by revealing
            pad tokens in the mask.
        :return: A boolean mask of shape :code:`(batch_size, seq_len)`
            set to :code:`False` for positions within the region and :code:`True`
            for positions outside the region.
        """
        pass

    @abstractmethod
    def data_to_sequences(
        self,
        data: Any,
        split_by_region: bool = True,
        remove_gaps: bool = True,
    ) -> list[dict[str, str] | str]:
        """
        Converts a data representation (typically a tensor) into a list of string amino acid sequences.

        :param data: The batched sequences to convert into string form.
        :param split_by_region: Whether to return the sequences as dictionaries with keys for each IMGT region.
        :param remove_gaps: Whether to remove gap residues from the sequences.
        :return: The string sequences, each as a single string or a region dictionary.
        """
        pass


class UnpairedSequenceDatamodule(BaseDatamodule[UnpairedSequenceDataset]):
    """
    A datamodule for unpaired data chain sequences.
    """

    def __init__(self, tokenizer: IMGTTokenizer, cfg: UnpairedSequenceDatamoduleConfig):
        """
        :param tokenizer: The tokenizer to use for encoding the sequences.
        :param cfg: Configuration object for the UnpairedSequenceDatamodule -
            see :class:`UnpairedSequenceDatamoduleConfig` for the required parameters.
        """

        super().__init__()

        self.tokenizer = tokenizer
        self.colnames = cfg.colnames

        if cfg.train_dataset is not None:
            self._train_dataset = self.get_dataset(cfg.train_dataset)
        else:
            self._train_dataset = None

        if cfg.val_dataset is not None:
            self._val_dataset = self.get_dataset(cfg.val_dataset)
        else:
            self._val_dataset = None

        self.num_workers = cfg.num_workers
        self.batch_size = cfg.batch_size

        # Save the indices of the IMGT numbered regions in the datamodule's output tensors
        self.region_indices = {}
        current_idx = 0
        for region, length in tokenizer.region_lengths.items():
            self.region_indices[region] = slice(current_idx, current_idx + length)
            current_idx += length

        self.total_length = current_idx

    def get_dataset(self, path: str) -> UnpairedSequenceDataset:
        """
        Returns the dataset for the given path.

        :param path: The path to the dataset.
        :return: The dataset.
        """
        return UnpairedSequenceDataset(path, self.colnames, self.tokenizer)

    def get_imgt_inpaint_mask(
        self,
        sequences: torch.Tensor,
        regions: str | list[str],
        batch_size: int,
        reveal_pads: bool = True,
    ) -> torch.Tensor:
        """
        Returns an inpainting mask tensor for the given IMGT region(s), specifying
        which positions are to be conditioned on during sampling (i.e. all other regions).

        :param sequences: The sequence tensor to create the mask for.
        :param regions: The IMGT region(s) to create the mask for.
            Can be a single region or a list of regions. Each region must
            be one of ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4').
        :param batch_size: The batch size for the mask.
        :param reveal_pads: Whether to fix the length of the regions being inpainted by revealing
            pad tokens in the mask.
        :return: A boolean mask of shape :code:`(batch_size, seq_len)`
            set to :code:`False` for positions within the region and :code:`True`
            for positions outside the region.
        """
        mask = torch.ones((batch_size, self.total_length), dtype=torch.bool)

        if isinstance(regions, str):
            regions = [regions]

        for region in regions:
            region = region.lower()
            if region not in self.region_indices:
                raise ValueError(
                    f"IMGT region {region} not found. Possible values are: {list(self.region_indices.keys())}"
                )

            mask[..., self.region_indices[region]] = False

        if reveal_pads:
            mask[..., sequences == AA1_INDEX["-"]] = True

        return mask

    def data_to_sequences(
        self,
        data: torch.Tensor,
        split_by_region: bool = True,
        remove_gaps: bool = True,
    ) -> list[dict[str, str] | str]:
        """
        Converts a batched tensor of integers [0 - 20] into 1-letter code amino acid sequences.

        :param data: The batched sequence tensor to convert into string form.
        :param split_by_region: Whether to return the sequences as dictionaries with keys for each IMGT region.
        :param remove_gaps: Whether to remove gap residues from the sequences.
        :return: The string sequences, each as a single string or a region dictionary.
        """
        if split_by_region:
            region_sequences = {
                region: seq_batch_to_aa1(data[..., idx], remove_gaps=remove_gaps)
                for region, idx in self.region_indices.items()
            }
            sequences = []
            for i in range(data.shape[0]):
                sequences.append(
                    {region: seq[i] for region, seq in region_sequences.items()}
                )
        else:
            sequences = seq_batch_to_aa1(data, remove_gaps=remove_gaps)

        return sequences

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.batch_size})"

    @property
    def train_dataset(self):
        """The training dataset."""
        return self._train_dataset

    @property
    def validation_dataset(self):
        """The validation dataset."""
        return self._val_dataset

    def setup(self, stage: str):
        """Performs the train/test/validation split according to the proportions passed to the constructor."""
        if stage == "fit" and (
            self._train_dataset is None or self._val_dataset is None
        ):
            raise ValueError(
                "No train or validation dataset provided, setup failed for stage='fit'."
            )

    def train_dataloader(self) -> DataLoader:
        """The train dataloader using the train dataset."""
        return DataLoader(
            self._train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        The validation dataloader using the validation dataset.
        """
        return DataLoader(
            self._val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )


class PairedSequenceDatamodule(BaseDatamodule[PairedSequenceDataset]):
    """
    A datamodule for paired VH/VL sequences.
    """

    def __init__(
        self,
        vh_tokenizer: IMGTTokenizer,
        vl_tokenizer: IMGTTokenizer,
        cfg: PairedSequenceDatamoduleConfig,
    ):
        """
        :param vh_tokenizer: The tokenizer to use for encoding the VH sequences.
        :param vl_tokenizer: The tokenizer to use for encoding the VL sequences.
        :param cfg: Configuration object for the PairedSequenceDatamodule -
            see :class:`PairedSequenceDatamoduleConfig` for the required parameters.
        """

        super().__init__()

        self.vh_tokenizer = vh_tokenizer
        self.vl_tokenizer = vl_tokenizer
        self.vh_colnames = cfg.vh_colnames
        self.vl_colnames = cfg.vl_colnames

        if cfg.train_dataset is not None:
            self._train_dataset = self.get_dataset(cfg.train_dataset)
        else:
            self._train_dataset = None

        if cfg.val_dataset is not None:
            self._val_dataset = self.get_dataset(cfg.val_dataset)
        else:
            self._val_dataset = None

        self.num_workers = cfg.num_workers
        self.batch_size = cfg.batch_size

        # Save the indices of the IMGT numbered regions in the datamodule's output tensors
        self.vh_region_indices = {}
        current_idx = 0
        for region, length in vh_tokenizer.region_lengths.items():
            self.vh_region_indices[region] = slice(current_idx, current_idx + length)
            current_idx += length

        self.vl_region_indices = {}
        current_idx = 0
        for region, length in vl_tokenizer.region_lengths.items():
            self.vl_region_indices[region] = slice(current_idx, current_idx + length)
            current_idx += length

        self.vh_length = sum(vh_tokenizer.region_lengths.values())
        self.vl_length = sum(vl_tokenizer.region_lengths.values())

    def get_dataset(self, path: str) -> PairedSequenceDataset:
        """
        Returns the dataset for the given path.

        :param path: The path to the dataset.
        :return: The dataset.
        """
        return PairedSequenceDataset(
            path,
            self.vh_colnames,
            self.vl_colnames,
            self.vh_tokenizer,
            self.vl_tokenizer,
        )

    def get_imgt_inpaint_mask(
        self,
        sequences: tuple[torch.Tensor, torch.Tensor],
        regions: str | list[str],
        batch_size: int,
        reveal_pads: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns an inpainting mask tensor for the given IMGT region(s), specifying
        which positions are to be conditioned on during sampling (i.e. all other regions).

        :param sequences: The VH/VL sequence tensors to create the mask for.
        :param regions: The IMGT region(s) to create the mask for. Can be a single region
            or a list of regions. Each region can be of the form ('fwr1', 'cdr1', etc.)
            in which a mask is generated for both the VH and VL chains, or can be of
            the form ('H-fwr1', 'L-fwr1', etc.) to mask a specific chain.
        :param batch_size: The batch size for the mask.
        :param reveal_pads: Whether to unmask the length of the regions being inpainted by revealing
            pad tokens in the mask.
        :return: Boolean masks of shape :code:`(batch_size, seq_len)` for the VH/VL chains
            set to :code:`False` for positions within the region and :code:`True`
            for positions outside the region.
        """
        vh_mask = torch.ones((batch_size, self.vh_length), dtype=torch.bool)
        vl_mask = torch.ones((batch_size, self.vl_length), dtype=torch.bool)

        if isinstance(regions, str):
            regions = [regions]

        for region in regions:
            if region[0] == "H":
                _, region = region.split("-")
                if region not in self.vh_region_indices:
                    raise ValueError(
                        f"IMGT region H-{region} not found. Possible values are: "
                        f"{[f'H-{region}' for region in self.vh_region_indices]}"
                    )

                vh_mask[..., self.vh_region_indices[region]] = False

            elif region[0] == "L":
                _, region = region.split("-")
                if region not in self.vl_region_indices:
                    raise ValueError(
                        f"IMGT region L-{region} not found. Possible values are: "
                        f"{[f'L-{region}' for region in self.vl_region_indices]}"
                    )

                vl_mask[..., self.vl_region_indices[region]] = False

            elif region in self.vh_region_indices and region in self.vl_region_indices:
                vh_mask[..., self.vh_region_indices[region]] = False
                vl_mask[..., self.vl_region_indices[region]] = False

            else:
                raise ValueError(
                    f"IMGT region {region} not found. Possible values are: {list(self.vh_region_indices)}"
                )

        if reveal_pads:
            vh_mask[..., sequences[0] == AA1_INDEX["-"]] = True
            vl_mask[..., sequences[1] == AA1_INDEX["-"]] = True

        return vh_mask, vl_mask

    def data_to_sequences(
        self,
        data: tuple[torch.Tensor, torch.Tensor],
        split_by_region: bool = True,
        remove_gaps: bool = True,
    ) -> list[dict[str, str] | str]:
        """
        Converts a pair of batched sequence tensors for the VH/VL chains into 1-letter code amino acid sequences.
        If :code:`split_by_region=True`, VH/VL regions are given a chain type prefix in the output dictionaries
        (e.g. :code:`H-cdr3`, :code:`L-fwr1`). If :code:`split_by_region=False`, the sequences are returned
        in the form <VH>:<VL>.

        :param data: The batched VH/VL sequence tensors to convert into string form.
        :param split_by_region: Whether to return the sequences as dictionaries with keys for each IMGT region.
        :param remove_gaps: Whether to remove gap residues from the sequences.
        :return: The string sequences, each as a single string or a region dictionary.
        """
        vh, vl = data
        if split_by_region:
            vh_region_sequences = {
                f"H-{region}": seq_batch_to_aa1(vh[..., idx], remove_gaps=remove_gaps)
                for region, idx in self.vh_region_indices.items()
            }
            vl_region_sequences = {
                f"L-{region}": seq_batch_to_aa1(vl[..., idx], remove_gaps=remove_gaps)
                for region, idx in self.vl_region_indices.items()
            }

            sequences = []
            for i in range(vh.shape[0]):
                sequences.append(
                    {
                        **{
                            region: seq[i]
                            for region, seq in vh_region_sequences.items()
                        },
                        **{
                            region: seq[i]
                            for region, seq in vl_region_sequences.items()
                        },
                    }
                )
        else:
            vh_sequences = seq_batch_to_aa1(vh, remove_gaps=remove_gaps)
            vl_sequences = seq_batch_to_aa1(vl, remove_gaps=remove_gaps)
            sequences = [
                f"{vh_seq}:{vl_seq}"
                for vh_seq, vl_seq in zip(vh_sequences, vl_sequences)
            ]

        return sequences

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.batch_size})"

    @property
    def train_dataset(self):
        """The training dataset."""
        return self._train_dataset

    @property
    def validation_dataset(self):
        """The validation dataset."""
        return self._val_dataset

    def setup(self, stage: str):
        """Performs the train/test/validation split according to the proportions passed to the constructor."""
        if stage == "fit" and (
            self._train_dataset is None or self._val_dataset is None
        ):
            raise ValueError(
                "No train or validation dataset provided, setup failed for stage='fit'."
            )

    def train_dataloader(self) -> DataLoader:
        """The train dataloader using the train dataset."""
        return DataLoader(
            self._train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        The validation dataloader using the validation dataset.
        """
        return DataLoader(
            self._val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )


class PairedStructureDatamodule(PairedSequenceDatamodule):
    """
    A datamodule for paired VH/VL structures.
    """

    def __init__(
        self,
        vh_tokenizer: IMGTTokenizer,
        vl_tokenizer: IMGTTokenizer,
        cfg: PairedStructureDatamoduleConfig,
    ):
        """
        :param vh_tokenizer: The tokenizer to use for encoding the VH sequences.
        :param vl_tokenizer: The tokenizer to use for encoding the VL sequences.
        :param cfg: Configuration object for the PairedStructureDatamodule -
            see :class:`PairedStructureDatamoduleConfig` for the required parameters.
        """
        super(PairedSequenceDatamodule, self).__init__()

        self.vh_tokenizer = vh_tokenizer
        self.vl_tokenizer = vl_tokenizer
        self.vh_colnames = cfg.vh_colnames
        self.vl_colnames = cfg.vl_colnames
        self.max_epitope_length = cfg.max_epitope_length
        self.vh_key = cfg.vh_key
        self.vl_key = cfg.vl_key
        self.epitope_key = cfg.epitope_key

        if cfg.train_dataset is not None:
            self._train_dataset = self.get_dataset(cfg.train_dataset)
            self._train_dataset.noise_structures = True
        else:
            self._train_dataset = None

        if cfg.val_dataset is not None:
            self._val_dataset = self.get_dataset(cfg.val_dataset)
        else:
            self._val_dataset = None

        self.num_workers = cfg.num_workers
        self.batch_size = cfg.batch_size

        # Save the indices of the IMGT numbered regions in the datamodule's output tensors
        self.vh_region_indices = {}
        current_idx = 0
        for region, length in vh_tokenizer.region_lengths.items():
            self.vh_region_indices[region] = slice(current_idx, current_idx + length)
            current_idx += length

        self.vl_region_indices = {}
        current_idx = 0
        for region, length in vl_tokenizer.region_lengths.items():
            self.vl_region_indices[region] = slice(current_idx, current_idx + length)
            current_idx += length

        self.vh_length = sum(vh_tokenizer.region_lengths.values())
        self.vl_length = sum(vl_tokenizer.region_lengths.values())

    def get_dataset(self, path: str) -> PairedSequenceDataset | PairedStructureDataset:
        """
        Returns the dataset for the given path. The path can either be a path to a
        CSV file containing paired VH/VL sequences or a path to a HDF5 file containing
        paired VH/VL structures.

        :param path: The path to the dataset.
        :return: The paired structure or sequence dataset.
        """
        if path.endswith(".csv") or path.endswith(".pqt"):
            dataset = PairedSequenceDataset(
                path,
                self.vh_colnames,
                self.vl_colnames,
                self.vh_tokenizer,
                self.vl_tokenizer,
            )
        else:
            dataset = PairedStructureDataset(
                path,
                self.vh_tokenizer,
                self.vl_tokenizer,
                self.max_epitope_length,
                vh_key=self.vh_key,
                vl_key=self.vl_key,
                epitope_key=self.epitope_key,
            )

        return dataset

    def pdb_to_data(
        self, pdb_data: AntibodyPDBData, keep_chains: set[tuple[str, str]] | None = None
    ) -> dict[
        tuple[str, str], tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any]]
    ]:
        """
        Extracts the VH, VL, and epitope sequences and structures from a PDB data object.

        :param pdb_data: A PDB data object containing the paired antibody structure data.
        :param keep_chains: A set of chains to keep in the data. If None, all chains are kept.
        :return: A dictionary  mapping H/L chain pairs to a tuple of tokenized sequences and structure data.
        """
        complex_residues = pdb_data.get_complex_residues(include_duplicates=True)

        # Keep all chains if not specified
        if keep_chains is None:
            keep_chains = set(complex_residues)

        data_by_chain = {}
        for chain, (vh_res, vl_res, epitope_res) in complex_residues.items():
            # Chain being a string indicates it is unpaired, skip
            if isinstance(chain, str):
                continue

            if chain not in keep_chains:
                continue

            vh_chain, vl_chain = chain
            antibody_chain_ids = {
                "vh": vh_chain,
                "vl": vl_chain,
            }
            tokenizers = {
                "vh": self.vh_tokenizer,
                "vl": self.vl_tokenizer,
            }
            chain_residues = {
                "vh": vh_res,
                "vl": vl_res,
                "epitope": epitope_res,
            }

            antibody_sequences = {}
            structure_data = {}
            for chain_type, residues in chain_residues.items():
                n = np.array([res["N"].get_coord() for res in residues])
                ca = np.array([res["CA"].get_coord() for res in residues])
                c = np.array([res["C"].get_coord() for res in residues])

                # Skip chains with no residues
                if len(ca) == 0:
                    structure_data[chain_type] = {
                        "frames": (
                            torch.zeros((self.max_epitope_length, 3, 3)),
                            torch.zeros((self.max_epitope_length, 3)),
                        ),
                        "mask": torch.ones(self.max_epitope_length, dtype=torch.bool),
                    }
                    if chain_type == "epitope":
                        structure_data[chain_type]["features"] = torch.zeros(
                            (self.max_epitope_length, len(AA1_INDEX))
                        )
                    continue

                # For antibody chains, tokenize the SEQRES sequence
                if chain_type in antibody_chain_ids:
                    chain_id = antibody_chain_ids[chain_type]
                    imgt_ptr = get_imgt_ptr(pdb_data.numberings[chain_id])
                    atom_mask = tokenizers[chain_type].pad_atom_mask(
                        pdb_data.atom_masks[chain_id], imgt_ptr
                    )
                    sequence = list(pdb_data.sequences[chain_id])
                    imgt_sequences = [
                        "".join(seq) for seq in np.split(sequence, imgt_ptr)
                    ]
                    antibody_sequences[chain_type] = tokenizers[chain_type].encode(
                        imgt_sequences
                    )
                    features = None

                # Otherwise use the ATOM records sequence
                else:
                    atom_mask = np.concatenate(
                        [
                            np.ones(len(ca), dtype=bool),
                            np.zeros(self.max_epitope_length - len(ca), dtype=bool),
                        ]
                    )
                    sequence = [
                        protein_letters_3to1[res.get_resname()] for res in residues
                    ]
                    seq_index = torch.as_tensor(
                        [AA1_INDEX.get(aa, AA1_INDEX["-"]) for aa in sequence]
                    )
                    features = F.pad(
                        F.one_hot(seq_index, num_classes=len(AA1_INDEX)).float(),
                        (0, 0, 0, self.max_epitope_length - len(ca)),
                    )

                frames = backbone_to_frames(n, ca, c)
                rots, trans = pad_structure_frames(frames, atom_mask)
                structure_data[chain_type] = {
                    "frames": (torch.as_tensor(rots), torch.as_tensor(trans)),
                    "mask": torch.as_tensor(~atom_mask),
                }

                if features is not None:
                    structure_data[chain_type]["features"] = features

            data = (antibody_sequences["vh"], antibody_sequences["vl"]), structure_data
            data_by_chain[chain] = data

        return data_by_chain
