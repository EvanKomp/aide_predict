# aide_predict/utils/alignment_calls.py
'''
* Author: Evan Komp
* Created: 6/12/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper of EVCouplings alignment functions. All credit goes to the EVcouplings team:
Hopf T. A., Green A. G., Schubert B., et al. The EVcouplings Python framework for coevolutionary sequence analysis. Bioinformatics 35, 1582–1584 (2019)

'''
import tempfile
import subprocess
import os

from evcouplings.align.tools import *

from Bio import pairwise2
from Bio.Align import substitution_matrices

from typing import Union, Optional

import logging
logger = logging.getLogger(__name__)

def sw_global_pairwise(seq1: "ProteinSequence", seq2: "ProteinSequence", matrix: str = 'BLOSUM62', gap_open: float = -10, gap_extend: float = -0.5) -> tuple['ProteinSequence', 'ProteinSequence']:
    """
    Align two ProteinSequence objects using global alignment with a specified substitution matrix.

    Args:
        seq1 (ProteinSequence): The first protein sequence to align.
        seq2 (ProteinSequence): The second protein sequence to align.
        matrix (str, optional): The substitution matrix to use. Defaults to 'BLOSUM62'.
        gap_open (float, optional): The gap opening penalty. Defaults to -10.
        gap_extend (float, optional): The gap extension penalty. Defaults to -0.5.

    Returns:
        tuple[ProteinSequence, ProteinSequence]: A tuple containing the aligned sequences as ProteinSequence objects.
    """
    if seq1.has_gaps or seq2.has_gaps:
        raise ValueError("Input sequences should not contain gaps to be aligned")

    # Load the substitution matrix
    subst_matrix = substitution_matrices.load(matrix)

    # Perform the global alignment
    alignments = pairwise2.align.globalds(str(seq1), str(seq2), subst_matrix, gap_open, gap_extend)

    # Get the best alignment (first in the list)
    best_alignment = alignments[0]

    # Create new ProteinSequence objects with the aligned sequences
    from aide_predict.utils.data_structures import ProteinSequence
    aligned_seq1 = ProteinSequence(best_alignment.seqA, id=seq1.id, structure=seq1.structure)
    aligned_seq2 = ProteinSequence(best_alignment.seqB, id=seq2.id, structure=seq2.structure)

    return aligned_seq1, aligned_seq2


def mafft_align(sequences: "ProteinSequences",
                existing_alignment: Optional["ProteinSequences"] = None,
                realign: bool = False,
                output_fasta: Optional[str] = None) -> "ProteinSequences":
    """
    Perform multiple sequence alignment using MAFFT.

    Args:
        sequences (ProteinSequences): The sequences to align.
        existing_alignment (Optional[ProteinSequences]): An existing alignment to add sequences to.
        realign (bool): If True, realign all sequences from scratch. If False, add new sequences to existing alignment.
        output_fasta (Optional[str]): Path to save the alignment. If None, a temporary file is used.

    Returns:
        ProteinSequences: The aligned sequences, either in memory or on file depending on output_fasta.

    Raises:
        subprocess.CalledProcessError: If MAFFT execution fails.
        FileNotFoundError: If MAFFT is not installed or not in PATH.
    """
    # Create a temporary directory for input and output files
    from aide_predict.utils.data_structures import ProteinSequences, ProteinSequencesOnFile

    # check that the sequences are not already gap containing
    if sequences.has_gaps:
        raise ValueError("Input sequences should not contain gaps to be aligned")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare input file
        if isinstance(sequences, ProteinSequencesOnFile):
            input_fasta = sequences.file_path
        else:
            input_fasta = os.path.join(temp_dir, "input.fasta")
            sequences.to_fasta(input_fasta)

        # Prepare output file
        if output_fasta:
            output_file = output_fasta
        else:
            output_file = os.path.join(temp_dir, "output.fasta")

        # Prepare MAFFT command
        mafft_cmd = ["mafft"]


        # prepare existing alignment
        if existing_alignment is not None:
            if not existing_alignment.aligned:
                raise ValueError("Existing alignment must be aligned")
            if isinstance(existing_alignment, ProteinSequencesOnFile):
                existing_fasta = existing_alignment.file_path
            else:
                existing_fasta = os.path.join(temp_dir, "existing.fasta")
                existing_alignment.to_fasta(existing_fasta)
        
        if existing_alignment is not None and not realign:
            # Add to existing alignment
            mafft_cmd.extend(["--add", input_fasta, "--keeplength", existing_fasta])

        elif existing_alignment is not None and realign:
            # Realignment
            mafft_cmd.extend(["--add", input_fasta, existing_fasta])
        else:
            # New alignment or realignment
            mafft_cmd.extend([input_fasta])

        mafft_cmd.extend([">", output_file])

        # Run MAFFT
        try:
            subprocess.run(" ".join(mafft_cmd), shell=True, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"MAFFT alignment failed: {e.stderr.decode()}") from e
        except FileNotFoundError:
            raise FileNotFoundError("MAFFT is not installed or not in PATH")

        # Return aligned sequences
        if output_fasta:
            return ProteinSequencesOnFile(output_fasta)
        else:
            return ProteinSequences.from_fasta(output_file)