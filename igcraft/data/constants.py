"""Protein/antibody data constants."""

# One letter amino acid codes mapped to indices
AA1_INDEX = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "-": 20,
}

AA1_LETTERS = {v: k for k, v in AA1_INDEX.items()}

# The IMGT regions of an data chain
IMGT_REGIONS = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4")
