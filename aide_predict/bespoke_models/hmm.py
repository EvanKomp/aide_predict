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

from sklearn.utils import check_array
from sklearn.base import TransformerMixin, RegressorMixin
import numpy as np
import pandas as pd

from aide_predict.bespoke_models.base import ProteinModelWrapperRequiresMSA
from aide_predict.io.bio_files import write_fasta
from aide_predict.utils.common import process_amino_acid_sequences

import logging
logger = logging.getLogger(__name__)


class HMMWrapper(TransformerMixin, RegressorMixin, ProteinModelWrapperRequiresMSA):
    """Wrapper for HMMs.
    """
    _requires_wt_during_inference = False
    _per_position_capable = False
    _requires_msa = True
    _requires_fixed_length = False

    def __init__(self, metadata_folder: str=None, threshold=100):
        self.threshold = threshold
        super().__init__(metadata_folder=metadata_folder)

    def _more_tags(self):
        return {'stateless': True,
                'preserves_dtype': [],
                }

    def fit(self, X, y=None):
        """Fit the model.
        
        Params:
        - X: ignored. Model is fit based on alignment in metadata_folder."""
        if os.path.exists(os.path.join(self.metadata_folder, 'alignment.hmm')):
            logger.debug("Model already exists, skipping fit.")
        else:
            # run hmmbuild
            cmd = f"hmmbuild {os.path.join(self.metadata_folder, 'alignment.hmm')}" \
                  f" {os.path.join(self.metadata_folder, 'alignment.a2m')}"
            
            logger.info(f"Building hmm: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            assert os.path.exists(os.path.join(self.metadata_folder, 'alignment.hmm'))
        self.fitted_ = True
        return self
    
    def _transform(self, X):
        """Get HMM scores for sequences.

        Params:
        - X: np.ndarray of sequence amino acid strings
        """

        # create temp directory to call hmmsearch
        with tempfile.TemporaryDirectory() as tmpdirname:
            # write the sequences to a file
            seq_file = os.path.join(tmpdirname, 'seqs.fasta')
            out_tbl = os.path.join(tmpdirname, 'out.tbl')
            with open(seq_file, 'w') as f:
                write_fasta([('seq_'+str(i), seq) for i, seq in enumerate(X)], f)
            cmd = f"hmmsearch --tblout {out_tbl} -T {self.threshold} --domT {self.threshold}" \
                  f" --incT {self.threshold} --incdomT {self.threshold}" \
                  f" {os.path.join(self.metadata_folder, 'alignment.hmm')} {seq_file}"
            logger.info(f"Running hmmsearch: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

            # load and read the tblout to get scores
            data = []
            with open(out_tbl, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    data.append(line.split())
        data = pd.DataFrame(data)
        # get the scores
        # they need to be mapped back to the correct order
        scores = np.zeros(len(X))
        for i, seq in enumerate(X):
            try:
                scores[i] = float(data[data[0] == 'seq_'+str(i)][5])
            except ValueError:
                scores[i] = 0.0
        return scores
    
    def _predict(self, X):
        return self._transform(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)



