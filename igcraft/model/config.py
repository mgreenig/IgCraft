"""Structured configs for CDRGraft model, network and datamodule."""

from omegaconf import DictConfig
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class CheckpointSettings:
    """
    An object describing the settings for a saved model, consisting of a config path and a checkpoint path.

    :param config_path: Optional path to the configuration file used to train the model.
    :param checkpoint_path: Optional path to the checkpoint file saved by :code:`pytorch_lightning`.
    :param reset_optimizer: Whether to reset the optimizer state when loading the checkpoint. This
        includes reset the global step, learning rate scheduler, and any other optimizer state.
    """

    config_path: str | Path | None = Field(default=None)
    checkpoint_path: str | Path | None = Field(default=None)
    reset_optimizer: bool = Field(default=True)

    @field_validator("config_path", "checkpoint_path")
    def _path_validator(cls, v):
        """
        Validates that the paths provided exist and converts them to Path objects.
        """
        if isinstance(v, str):
            v = Path(v)

            if not v.exists():
                raise FileNotFoundError(f"Path {v} does not exist.")

        return v


@dataclass(config={"arbitrary_types_allowed": True}, kw_only=True)
class SequenceTransformerConfig:
    """
    Configuration for the Transformer network.

    :param num_layers: The number of self-attention block layers in the Transformer network.
    :param embed_dim: The embedding dimension of single representation of the network.
    :param num_heads: The number of heads to use for multi-head attention in self-attention blocks.
    :param dropout_p: The dropout probability in attention layers.
    :param output_dim: The dimension of output from the network.
    :param use_entropy_encoding: Whether to use a fourier feature entropy encoding in the
        embedding layer of the network. If :code:`True`, the entropy of the input probabilities
        is encoded with fourier features, and if :code:`False`, the time :code:`t` is encoded.
    :param fourier_n_min: The :code:`n_min` value to use for the fourier feature encoding of the time or entropy.
        See https://arxiv.org/pdf/2107.00630 (page 16) for more details. Default value of -1 taken from
        https://www.biorxiv.org/content/10.1101/2024.09.24.614734v1.full.pdf.
    :param fourier_n_max: The :code:`n_max` value to use for the fourier feature encoding of the time or entropy.
        See https://arxiv.org/pdf/2107.00630 (page 16) for more details. Default value of 16 taken from
        https://www.biorxiv.org/content/10.1101/2024.09.24.614734v1.full.pdf.
    """

    num_layers: int
    embed_dim: int
    num_heads: int
    dropout_p: float
    output_dim: int
    use_entropy_encoding: bool = True
    fourier_n_min: int = -1
    fourier_n_max: int = 16


@dataclass(config={"arbitrary_types_allowed": True}, kw_only=True)
class PairedSequenceTransformerConfig:
    """
    Configuration for a paired VH/VL transformer network.

    :param embed_dim: The embedding dimension used in the cross attention networks.
    :param num_heads: The number of heads to use for multi-head attention in cross-attention blocks.
    :param dropout_p: The dropout probability in attention layers.
    :param gate_bias_value: The bias value for the gating operation in cross-attention blocks.
    :param freeze_heavy: Boolean to freeze weights in the heavy network.
    :param freeze_light: Boolean to freeze weights in the light network.
    :param freeze_all: Boolean to freeze all weights in the network, including interaction modules.
    """

    embed_dim: int
    num_heads: int
    dropout_p: float = 0.1
    gate_bias_value: float = 0.0
    freeze_heavy: bool = False
    freeze_light: bool = False
    freeze_all: bool = False


@dataclass(kw_only=True)
class DatamoduleConfig:
    """
    Basic configuration for a datamodule.

    :param train_dataset: The path to the train dataset as a CSV file.
    :param val_dataset: The path to the validation dataset as a CSV file.
    :param test_dataset: The path to the test dataset as a CSV file.
    :param batch_size: The batch size for training, validation, and testing.
    :param num_workers: The number of workers to use for data loading.
    """

    train_dataset: Path | str | None = Field(default=None)
    val_dataset: Path | str | None = Field(default=None)
    test_dataset: Path | str | None = Field(default=None)
    batch_size: int | tuple[int, int] | tuple[int, int, int]
    num_workers: int = Field(default=0, ge=0)


@dataclass(kw_only=True)
class UnpairedSequenceDatamoduleConfig(DatamoduleConfig):
    """
    Configuration for a data module for unpaired data sequences.

    :param train_dataset: The path to the train dataset as a CSV file.
    :param val_dataset: The path to the validation dataset as a CSV file.
    :param test_dataset: The path to the test dataset as a CSV file.
    :param batch_size: The batch size for training, validation, and testing.
    :param num_workers: The number of workers to use for data loading.
    :param colnames: The 7 column names in the dataset for the 7 IMGT regions of the sequences,
        in the order: FWR1, CDR1, FWR2, CDR2, FWR3, CDR3, FWR4.
    """

    colnames: list[str]


@dataclass(kw_only=True)
class PairedSequenceDatamoduleConfig(DatamoduleConfig):
    """
    Configuration for a datamodule for paired VH/VL sequences.

    :param train_dataset: The path to the train dataset.
    :param val_dataset: The path to the validation dataset.
    :param test_dataset: The path to the test dataset.
    :param batch_size: The batch size for training, validation, and testing.
    :param num_workers: The number of workers to use for data loading.
    :param vh_colnames: The 7 column names in the datasets for the 7 regions of the VH sequence,
        in the order: FWR1, CDR1, FWR2, CDR2, FWR3, CDR3, FWR4.
    :param vl_colnames: The 7 column names in the datasets for the 7 regions of the VL sequence,
        in the order: FWR1, CDR1, FWR2, CDR2, FWR3, CDR3, FWR4.
    """

    vh_colnames: list[str]
    vl_colnames: list[str]

    @field_validator("dataset", "splits", "sampling_weights")
    def _path_validator(cls, v):
        """
        Validates that the paths provided exist and converts them to Path objects.
        """
        if isinstance(v, str):
            v = Path(v)

            if not v.exists():
                raise FileNotFoundError(f"Path {v} does not exist.")

        return v

    def __post_init__(self):
        """
        Validates that there are 7 IMGT region column names specified by :code:`vh_colnames`
        and :code:`vl_colnames` (each).
        """

        if len(self.vh_colnames) != 7:
            raise ValueError(
                f"There should be 7 VH column names specified (for the 7 regions FWR1, "
                f"CDR1, FWR2, CDR2, FWR3, CDR3, FWR4). Got {len(self.vh_colnames)}: {self.vh_colnames}."
            )

        if len(self.vl_colnames) != 7:
            raise ValueError(
                f"There should be 7 VL column names specified (for the 7 regions FWR1, "
                f"CDR1, FWR2, CDR2, FWR3, CDR3, FWR4). Got {len(self.vl_colnames)}: {self.vl_colnames}."
            )


@dataclass(kw_only=True)
class PairedStructureDatamoduleConfig(PairedSequenceDatamoduleConfig):
    """
    Configuration for a datamodule for paired VH/VL structures.

    :param train_dataset: The path to the train dataset as a CSV file.
    :param val_dataset: The path to the validation dataset as a CSV file.
    :param test_dataset: The path to the test dataset as a CSV file.
    :param batch_size: The batch size for training, validation, and testing.
    :param num_workers: The number of workers to use for data loading.
    :param vh_colnames: The 7 column names in the CSV datasets for the 7 regions of the VH sequence,
        in the order: FWR1, CDR1, FWR2, CDR2, FWR3, CDR3, FWR4.
    :param vl_colnames: The 7 column names in the CSV datasets for the 7 regions of the VL sequence,
        in the order: FWR1, CDR1, FWR2, CDR2, FWR3, CDR3, FWR4.
    :param max_epitope_length: The maximum length of the epitope sequence.
    :param vh_key: Key for VH chains in the structure groups of the HDF5 dataset.
    :param vl_key: Key for VL chains in the structure groups of the HDF5 dataset.
    :param epitope_key: Key for epitopes in the structure groups of the HDF5 dataset.
    """

    max_epitope_length: int
    vh_key: str
    vl_key: str
    epitope_key: str


@dataclass(kw_only=True)
class SequenceModelConfig:
    """
    Configuration for a sequence generative model.

    :param max_lr: The maximum learning rate for the learning rate scheduler.
    :param min_lr: The minimum learning rate for the learning rate scheduler.
    :param use_lr_schedule: Whether to use a learning rate scheduler.
    :param warmup_steps: The number of warmup steps for the learning rate scheduler.
    :param transition_steps: The number of transition steps for the learning rate scheduler.
    :param ema_decay: The decay rate for the exponential moving average of the model weights.
    :param num_eval_time_bins: The number of time bins to use for evaluation. (default: 10)
    :param mask_fwr_rate: The rate at which framework regions in the structural
        conditioning data are masked. (default: 0.5)
    """

    max_lr: float
    min_lr: float
    use_lr_schedule: bool
    warmup_steps: int
    transition_steps: int
    ema_decay: float
    num_eval_time_bins: int = 10
    mask_fwr_rate: float = 0.5


@dataclass(kw_only=True)
class PairedSequenceModelConfig(SequenceModelConfig):
    """
    Configuration for a paired sequence generative model.

    :param max_lr: The maximum learning rate for the learning rate scheduler.
    :param min_lr: The minimum learning rate for the learning rate scheduler.
    :param use_lr_schedule: Whether to use a learning rate scheduler.
    :param warmup_steps: The number of warmup steps for the learning rate scheduler.
    :param transition_steps: The number of transition steps for the learning rate scheduler.
    :param ema_decay: The decay rate for the exponential moving average of the model weights.
    :param num_eval_time_bins: The number of time bins to use for evaluation. (default: 10)
    :param vh_ckpt_path: Checkpoint file for an unpaired VH :class:`SequenceGenerativeModel`. If provided,
        the VH backbone will be loaded from this checkpoint.
    :param vl_ckpt_path: Checkpoint file for an unpaired VL :class:`SequenceGenerativeModel`. If provided,
        the VL backbone will be loaded from this checkpoint.
    """

    vh_ckpt_path: str | Path | None = Field(default=None)
    vl_ckpt_path: str | Path | None = Field(default=None)
