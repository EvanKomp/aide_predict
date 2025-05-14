# aide_predict/bespoke_models/hmm.py
'''
* Author: Evan Komp
* Created: 6/11/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper of HMMs into an sklearn transformer for use in the AIDE pipeline. Uses HMMsearch against
the HMM

Here are the docs for HMMSearch:

Usage: hmmsearch [options] <hmmfile> <seqdb>

Basic options:
  -h : show brief help on version and usage

Options directing output:
  -o <f>           : direct output to file <f>, not stdout
  -A <f>           : save multiple alignment of all hits to file <f>
  --tblout <f>     : save parseable table of per-sequence hits to file <f>
  --domtblout <f>  : save parseable table of per-domain hits to file <f>
  --pfamtblout <f> : save table of hits and domains to file, in Pfam format <f>
  --acc            : prefer accessions over names in output
  --noali          : don't output alignments, so output is smaller
  --notextw        : unlimit ASCII text output line width
  --textw <n>      : set max width of ASCII text output lines  [120]  (n>=120)

Options controlling reporting thresholds:
  -E <x>     : report sequences <= this E-value threshold in output  [10.0]  (x>0)
  -T <x>     : report sequences >= this score threshold in output
  --domE <x> : report domains <= this E-value threshold in output  [10.0]  (x>0)
  --domT <x> : report domains >= this score cutoff in output

Options controlling inclusion (significance) thresholds:
  --incE <x>    : consider sequences <= this E-value threshold as significant
  --incT <x>    : consider sequences >= this score threshold as significant
  --incdomE <x> : consider domains <= this E-value threshold as significant
  --incdomT <x> : consider domains >= this score threshold as significant

Options controlling model-specific thresholding:
  --cut_ga : use profile's GA gathering cutoffs to set all thresholding
  --cut_nc : use profile's NC noise cutoffs to set all thresholding
  --cut_tc : use profile's TC trusted cutoffs to set all thresholding

Options controlling acceleration heuristics:
  --max    : Turn all heuristic filters off (less speed, more power)
  --F1 <x> : Stage 1 (MSV) threshold: promote hits w/ P <= F1  [0.02]
  --F2 <x> : Stage 2 (Vit) threshold: promote hits w/ P <= F2  [1e-3]
  --F3 <x> : Stage 3 (Fwd) threshold: promote hits w/ P <= F3  [1e-5]
  --nobias : turn off composition bias filter

Other expert options:
  --nonull2     : turn off biased composition score corrections
  -Z <x>        : set # of comparisons done, for E-value calculation
  --domZ <x>    : set # of significant seqs, for domain E-value calculation
  --seed <n>    : set RNG seed to <n> (if 0: one-time arbitrary seed)  [42]
  --tformat <s> : assert target <seqfile> is in format <s>: no autodetection
  --cpu <n>     : number of parallel CPU workers to use for multithreads  [2]

Some of these need to be user parameterizable, and some need to be fixed.
'''
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from aide_predict.bespoke_models.base import ProteinModelWrapper, RequiresMSAForFitMixin, CanRegressMixin
from aide_predict.utils.data_structures import ProteinSequences

from typing import Optional, Union

import logging
logger = logging.getLogger(__name__)


class HMMWrapper(CanRegressMixin, RequiresMSAForFitMixin, ProteinModelWrapper):
    """
    Wrapper for Hidden Markov Models (HMMs) using HMMsearch to score sequences.

    This wrapper builds an HMM from an input alignment and uses HMMsearch to get scores for new sequences.
    Bit scores are used to compare to the HMM as opposed to E values. Tune the threshold parameter
    accordingly.

    Attributes:
        threshold (float): Threshold for HMMsearch.
        metadata_folder (str): Folder to store metadata.
        wt (Optional[ProteinSequence]): Wild-type sequence.

    """
    def __init__(self, threshold: float = 100, metadata_folder: Optional[str] = None, wt: Optional[Union[str, 'ProteinSequence']] = None):
        """
        Initialize the HMMWrapper.

        Args:
            threshold (float): Threshold for HMMsearch. Defaults to 100.
            metadata_folder (Optional[str]): Folder to store metadata. Defaults to None.
            wt (Optional[Union[str, 'ProteinSequence']]): Wild-type sequence. Defaults to None.
        """
        self.threshold = threshold
        super().__init__(metadata_folder=metadata_folder, wt=wt)

    def _more_tags(self):
        return {'stateless': True,
                'preserves_dtype': [],
                }

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'HMMWrapper':
        """
        Fit the HMM model using the input alignment.

        Args:
            X (ProteinSequences): Alignment or sequences to build HMM from.
            y (Optional[np.ndarray]): Not used, present for API consistency.

        Returns:
            HMMWrapper: The fitted model.

        Raises:
            ValueError: If the input sequences are not aligned.
        """
        if not X.aligned:
            raise ValueError("Input sequences must be aligned for HMM building.")

        alignment_path = os.path.join(self.metadata_folder, 'alignment.a2m')
        hmm_path = os.path.join(self.metadata_folder, 'alignment.hmm')

        X.to_fasta(alignment_path)
        
        if os.path.exists(hmm_path):
            logger.debug("HMM model already exists, skipping fit.")
        else:
            cmd = f"hmmbuild {hmm_path} {alignment_path}"
            logger.info(f"Building HMM: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            
            if not os.path.exists(hmm_path):
                raise RuntimeError(f"Failed to create HMM file at {hmm_path}")

        self.fitted_ = True
        return self
    
    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Get HMM scores for input sequences.

        Args:
            X (ProteinSequences): Sequences to score.

        Returns:
            np.ndarray: Array of HMM scores for each input sequence.

        Raises:
            RuntimeError: If HMMsearch fails to run or produce output.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            seq_file = os.path.join(tmpdirname, 'seqs.fasta')
            out_tbl = os.path.join(tmpdirname, 'out.tbl')
            hmm_path = os.path.join(self.metadata_folder, 'alignment.hmm')

            X.to_fasta(seq_file)

            cmd = (f"hmmsearch --tblout {out_tbl} -T {self.threshold} --domT {self.threshold} "
                   f"--incT {self.threshold} --incdomT {self.threshold} {hmm_path} {seq_file}")
            
            logger.info(f"Running hmmsearch: {cmd}")
            process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            logger.info(process.stdout)
            logger.error(process.stderr)

            if not os.path.exists(out_tbl) or os.path.getsize(out_tbl) == 0:
                raise RuntimeError("HMMsearch failed to produce output")

            data = pd.read_csv(out_tbl, delim_whitespace=True, comment='#', header=None)

        scores = np.zeros((len(X), 1))
        for i, seq in enumerate(X):
          seq_id = seq.id if seq.id is not None else str(hash(seq))
          score = data[data[0] == seq_id][5].values
          scores[i] = float(score[0]) if len(score) > 0 else np.nan

        return scores
    



