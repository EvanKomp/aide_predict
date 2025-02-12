# aide_predict/io/bio_files.py
'''
* Author: Evan Komp
* Created: 5/22/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Some functions are copied from EVcoupling, as to avoid the additional required dependancy. All credit goes to the EVcouples team:

Hopf T. A., Green A. G., Schubert B., et al. The EVcouplings Python framework for coevolutionary sequence analysis. Bioinformatics 35, 1582–1584 (2019)
'''
from collections import OrderedDict

import numpy as np

from aide_predict.utils.common import wrap

def read_fasta(fileobj):
    """
    Generator function to read a FASTA-format file
    (includes aligned FASTA, A2M, A3M formats)

    Credit to EVcouplings

    Args:
    - fileobj: file object opened for reading

    Returns:
    - Tuple of (sequence_id, sequence) for each entry
    """
    current_sequence = ""
    current_id = None

    for line in fileobj:
        # Start reading new entry. If we already have
        # seen an entry before, return it first.
        if line.startswith(">"):
            if current_id is not None:
                yield current_id, current_sequence

            current_id = line.rstrip()[1:]
            current_sequence = ""

        elif not line.startswith(";"):
            current_sequence += line.rstrip()

    # Also do not forget last entry in file
    yield current_id, current_sequence

def write_fasta(sequences, fileobj, width=80):
    """
    Write a list of IDs/sequences to a FASTA-format file

    Credit to EVcouplings

    Args:
    - sequences: list of (sequence_id, sequence) tuples
    - fileobj: file object opened for writing
    - width: width of sequence lines in FASTA file

    Returns:
    - None
    """
    for (seq_id, seq) in sequences:
        fileobj.write(">{}\n".format(seq_id))
        fileobj.write(wrap(seq, width=width) + "\n")


def read_a3m(fileobj, inserts="first"):
    """
    Read an alignment in compressed a3m format and expand
    into a2m format.

    Credit to EVcouplings

    Args:
    - fileobj: file object opened for reading
    - inserts: how to handle insert gaps in alignment
        (either "first" or "delete")

    Returns:
    - OrderedDict of sequence_id -> sequence
    """
    seqs = OrderedDict()

    for i, (seq_id, seq) in enumerate(read_fasta(fileobj)):
        # remove any insert gaps that may still be in alignment
        # (just to be sure)
        seq = seq.replace(".", "")

        if inserts == "first":
            # define "spacing" of uppercase columns in
            # final alignment based on target sequence;
            # remaining columns will be filled with insert
            # gaps in the other sequences
            if i == 0:
                uppercase_cols = [
                    j for (j, c) in enumerate(seq)
                    if (c == c.upper() or c == "-")
                ]
                gap_template = np.array(["."] * len(seq))
                filled_seq = seq
            else:
                uppercase_chars = [
                    c for c in seq if c == c.upper() or c == "-"
                ]
                filled = np.copy(gap_template)
                filled[uppercase_cols] = uppercase_chars
                filled_seq = "".join(filled)

        elif inserts == "delete":
            # remove all lowercase letters and insert gaps .;
            # since each sequence must have same number of
            # uppercase letters or match gaps -, this gives
            # the final sequence in alignment
            seq = "".join([c for c in seq if c == c.upper() and c != "."])
        else:
            raise ValueError(
                "Invalid option for inserts: {}".format(inserts)
            )

        seqs[seq_id] = filled_seq

    return seqs