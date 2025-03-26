from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StructureData:
    """
    Object storing the structural conditioning data for a sequence generative model.

    :param frames: A tuple of two tensors, the first containing a (..., N, 3, 3) array of rotation matrices
        and the second containing a (..., N, 3) array of translation vectors.
    :param mask: A tensor of shape (..., N) containing a mask denoting which input frames are padding.
    :param features: An optional tensor of shape (..., N, D) containing additional features associated with
        the structural conditioning data.
    """

    frames: tuple[torch.Tensor, torch.Tensor]
    mask: torch.Tensor
    features: torch.Tensor | None = None

    def __post_init__(self):
        """
        Checks the shape of the input frames, mask, and feature tensors, adding a batch dimension if necessary.
        """
        rots, trans = self.frames
        if len(rots.shape) <= 3:
            if len(rots.shape) == 3:
                rots = rots[None]
            else:
                raise ValueError("Rotation matrices should have shape (..., N, 3, 3).")

        if len(trans.shape) <= 2:
            if len(trans.shape) == 2:
                trans = trans[None]
            else:
                raise ValueError("Translation vectors should have shape (..., N, 3).")

        self.frames = (rots, trans)

        if len(self.mask.shape) <= 1:
            if len(self.mask.shape) == 1:
                self.mask = self.mask[None]
            else:
                raise ValueError("Mask should have shape (..., N).")

        if self.features is not None:
            if len(self.features.shape) <= 2:
                if len(self.features.shape) == 2:
                    self.features = self.features[None]
                else:
                    raise ValueError("Features should have shape (..., N, D).")

    def __getitem__(self, idx: int | torch.Tensor | slice) -> StructureData:
        """
        Indexes the structure data object, returning a new object with the indexed data.
        Indexing is only supported along the batch dimension.
        """
        rots, trans = self.frames
        idx_frames = (rots[idx], trans[idx])
        idx_mask = self.mask[idx]

        if self.features is not None:
            idx_features = self.features[idx]
        else:
            idx_features = None

        return StructureData(idx_frames, idx_mask, idx_features)

    def __setitem__(self, idx: int | torch.Tensor | slice, value: StructureData):
        """
        Sets the structure data object at the specified index.
        """
        rots, trans = self.frames
        new_rots, new_trans = value.frames
        rots[idx] = new_rots
        trans[idx] = new_trans

        self.frames = (rots, trans)
        self.mask[idx] = value.mask

        if self.features is not None and value.features is not None:
            self.features[idx] = value.features

    def to(self, device: torch.device) -> StructureData:
        """
        Moves the structural data to the specified device.
        """
        rots, trans = self.frames
        rots = rots.to(device)
        trans = trans.to(device)
        mask = self.mask.to(device)

        if self.features is not None:
            features = self.features.to(device)
        else:
            features = None

        return StructureData((rots, trans), mask, features)


@dataclass
class UnpairedStructureData:
    """
    Object storing structural data for an unpaired Fv antibody.

    :param ab: Structural data for the antibody chain.
    :param epitope: Structural data for the epitope. Should be 100% padding if not present.
    """

    ab: StructureData
    epitope: StructureData

    def __getitem__(self, idx: int | torch.Tensor | slice) -> UnpairedStructureData:
        """
        Indexes the unpaired structure data object, returning a new object with the indexed data.
        Indexing is only supported along the batch dimension.
        """
        return UnpairedStructureData(self.ab[idx], self.epitope[idx])

    def to(self, device: torch.device) -> UnpairedStructureData:
        """
        Moves the structural data to the specified device.
        """
        return UnpairedStructureData(
            self.ab.to(device),
            self.epitope.to(device) if self.epitope is not None else None,
        )


@dataclass
class PairedStructureData:
    """
    Object storing structural data for a full paired Fv antibody, potentially with an epitope.

    :param vh: Structural data for the VH chain.
    :param vl: Structural data for the VL chain.
    :param epitope: Structural data for the epitope. Should be 100% padding if not present.
    """

    vh: StructureData
    vl: StructureData
    epitope: StructureData

    def __getitem__(self, idx: int | torch.Tensor | slice) -> PairedStructureData:
        """
        Indexes the paired structure data object, returning a new object with the indexed data.
        Indexing is only supported along the batch dimension.
        """
        return PairedStructureData(self.vh[idx], self.vl[idx], self.epitope[idx])

    def to(self, device: torch.device) -> PairedStructureData:
        """
        Moves the structural data to the specified device.
        """
        return PairedStructureData(
            self.vh.to(device), self.vl.to(device), self.epitope.to(device)
        )
