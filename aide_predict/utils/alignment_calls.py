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

from typing import Union

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




# def mafft(
#         sequences: Union[ProteinSequences, str],
#         existing_alignment: Union[ProteinSequences, str] = None,
#         realign: bool = False,
#         to_outfile: str = None,
#     ):
#     """
#     Wrapper for MAFFT alignment function.

#     Parameters
#     ----------
#     sequences : ProteinSequences, str
#         Sequences to align
#         if string, assumed to be path to fasta file
#     existing_alignment : ProteinSequences, str
#         Existing alignment align with current sequences
#         if string, assumed to be path to fasta file
#     realign : bool
#         If true and existing alignemnt provided, realigns all sequences from scratch
#         otherwise keeps existing alignment fixed and adds new sequences.
#     to_outfile : str
#         Path to write alignment to.

#     Returns
#     -------
#     ProteinSequences
#         Aligned sequences
#     or None if to_outfile is provided
#     """
#     with tempfile.TemporaryDirectory() as temp_dir:

#         if isinstance(sequences, str):
#             seq_file = sequences
#         elif isinstance(sequences, ProteinSequencesOnFile):
#             seq_file = sequences.fasta_file

#         elif isinstance(sequences, ProteinSequences):
#             seq_file = os.path.join(temp_dir, 'sequences.fasta')
#             sequences.to_fasta(seq_file)
#         else:
#             raise ValueError(f"Invalid type for sequences: {type(sequences)}")

#         if existing_alignment is not None:
#             if isinstance(existing_alignment, str):
#                 pass
#             elif isinstance(existing_alignment, ProteinSequencesOnFile):
#                 existing_alignment = existing_alignment.fasta_file
#             elif isinstance(existing_alignment, ProteinSequences):
#                 existing_alignment_file = os.path.join(temp_dir, 'existing_alignment.fasta')
#                 existing_alignment.to_fasta(existing_alignment_file)
#                 existing_alignment = existing_alignment_file
#             else:
#                 raise ValueError(f"Invalid type for existing_alignment: {type(existing_alignment)}")

#         if to_outfile is not None:
#             alignment_file = to_outfile
#         else:
#             alignment_file = os.path.join(temp_dir, 'alignment.fasta')


#         cmd = ['mafft']
#         if existing_alignment is not None:
#             cmd.extend(['--add', seq_file])
#             if not realign:
#                 cmd.append('--keeplength')
#             cmd.append(existing_alignment)
#         else:
#             cmd.append(seq_file)
        
#         cmd.extend(['>', alignment_file])
#         logger.debug(f"Running MAFFT: {' '.join(cmd)}")
#         subprocess.run(' '.join(cmd), shell=True)

#         return ProteinSequences.from_fasta(alignment_file)

