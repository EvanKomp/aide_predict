# aide_predict/utils/jackhmmer.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Python wrapper for running Jackhmmer. These MSAs are needed as input for a number of covariance strategies.
'''
import os
from dataclasses import dataclass

import subprocess

from typing import List, Optional

import logging
logger = logging.getLogger(__name__)

@dataclass
class JackhmmerArgs:
    """Defaults are EVcouplings defaults for running Jackhmmer."""
    seqdb: str
    cpus: int = 1
    iterations: int = 3
    evalue: float = 0.0001
    zvalue: Optional[int] = None
    popen: float = 0.02
    pextend: float = 0.4
    mx: str = 'BLOSUM62'
    executable: str = 'jackhmmer'


class Jackhmmer:
    """Wrapper for running Jackhmmer
    
    The executable has the signature:
    `jackhmmer [options] <seqfile> <seqdb>`

    We will 

    Printout from jackhmmer:
    # jackhmmer :: iteratively search a protein sequence against a protein database
    # HMMER 3.4 (Aug 2023); http://hmmer.org/
    # Copyright (C) 2023 Howard Hughes Medical Institute.
    # Freely distributed under the BSD open source license.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Usage: jackhmmer [-options] <seqfile> <seqdb>

    Basic options:
    -h     : show brief help on version and usage
    -N <n> : set maximum number of iterations to <n>  [5]  (n>0)

    Options directing output:
    -o <f>          : direct output to file <f>, not stdout
    -A <f>          : save multiple alignment of hits to file <f>
    --tblout <f>    : save parseable table of per-sequence hits to file <f>
    --domtblout <f> : save parseable table of per-domain hits to file <f>
    --chkhmm <f>    : save HMM checkpoints to files <f>-<iteration>.hmm
    --chkali <f>    : save alignment checkpoints to files <f>-<iteration>.sto
    --acc           : prefer accessions over names in output
    --noali         : don't output alignments, so output is smaller
    --notextw       : unlimit ASCII text output line width
    --textw <n>     : set max width of ASCII text output lines  [120]  (n>=120)

    Options controlling scoring system in first iteration:
    --popen <x>   : gap open probability
    --pextend <x> : gap extend probability
    --mx <s>      : substitution score matrix choice (of some built-in matrices)
    --mxfile <f>  : read substitution score matrix from file <f>

    Options controlling reporting thresholds:
    -E <x>     : report sequences <= this E-value threshold in output  [10.0]  (x>0)
    -T <x>     : report sequences >= this score threshold in output
    --domE <x> : report domains <= this E-value threshold in output  [10.0]  (x>0)
    --domT <x> : report domains >= this score cutoff in output

    Options controlling significance thresholds for inclusion in next round:
    --incE <x>    : consider sequences <= this E-value threshold as significant
    --incT <x>    : consider sequences >= this score threshold as significant
    --incdomE <x> : consider domains <= this E-value threshold as significant
    --incdomT <x> : consider domains >= this score threshold as significant

    Options controlling acceleration heuristics:
    --max    : Turn all heuristic filters off (less speed, more power)
    --F1 <x> : Stage 1 (MSV) threshold: promote hits w/ P <= F1  [0.02]
    --F2 <x> : Stage 2 (Vit) threshold: promote hits w/ P <= F2  [1e-3]
    --F3 <x> : Stage 3 (Fwd) threshold: promote hits w/ P <= F3  [1e-5]
    --nobias : turn off composition bias filter

    Options controlling model construction after first iteration:
    --fragthresh <x> : if L <= x*alen, tag sequence as a fragment  [0.5]  (0<=x<=1)

    Options controlling relative weights in models after first iteration:
    --wpb     : Henikoff position-based weights  [default]
    --wgsc    : Gerstein/Sonnhammer/Chothia tree weights
    --wblosum : Henikoff simple filter weights
    --wnone   : don't do any relative weighting; set all to 1
    --wid <x> : for --wblosum: set identity cutoff  [0.62]  (0<=x<=1)

    Options controlling effective seq number in models after first iteration:
    --eent       : adjust eff seq # to achieve relative entropy target
    --eentexp    : adjust eff seq # to reach rel. ent. target using exp scaling
    --eclust     : eff seq # is # of single linkage clusters
    --enone      : no effective seq # weighting: just use nseq
    --eset <x>   : set eff seq # for all models to <x>
    --ere <x>    : for --eent[exp]: set minimum rel entropy/position to <x>
    --esigma <x> : for --eent[exp]: set sigma param to <x>
    --eid <x>    : for --eclust: set fractional identity cutoff to <x>

    Options controlling prior strategy in models after first iteration:
    --pnone    : don't use any prior; parameters are frequencies
    --plaplace : use a Laplace +1 prior

    Options controlling E value calibration:
    --EmL <n> : length of sequences for MSV Gumbel mu fit  [200]  (n>0)
    --EmN <n> : number of sequences for MSV Gumbel mu fit  [200]  (n>0)
    --EvL <n> : length of sequences for Viterbi Gumbel mu fit  [200]  (n>0)
    --EvN <n> : number of sequences for Viterbi Gumbel mu fit  [200]  (n>0)
    --EfL <n> : length of sequences for Forward exp tail tau fit  [100]  (n>0)
    --EfN <n> : number of sequences for Forward exp tail tau fit  [200]  (n>0)
    --Eft <x> : tail mass for Forward exponential tail tau fit  [0.04]  (0<x<1)

    Other expert options:
    --nonull2     : turn off biased composition score corrections
    -Z <x>        : set # of comparisons done, for E-value calculation
    --domZ <x>    : set # of significant seqs, for domain E-value calculation
    --seed <n>    : set RNG seed to <n> (if 0: one-time arbitrary seed)  [42]
    --qformat <s> : assert query <seqfile> is in format <s>: no autodetection
    --tformat <s> : assert target <seqdb> is in format <s>>: no autodetection
    --cpu <n>     : number of parallel CPU workers to use for multithreads  [2]
    """
    def __init__(self, args: JackhmmerArgs):
        self.args = args

    def run(self, seqfile: str, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, 'jackhmmer.out')
        alignment_file = os.path.join(output_dir, 'jackhmmer.sto')

        cmd = [
            self.args.executable,
            '-N', str(self.args.iterations),
            '-E', str(self.args.evalue),
            '--incE', str(self.args.evalue),
            '--incdomE', str(self.args.evalue),
            '--cpu', str(self.args.cpus),
            '-o', output_file,
            '-A', alignment_file,
            '--popen', str(self.args.popen),
            '--pextend', str(self.args.pextend),
            '--mx', self.args.mx,
            '--noali',
            seqfile,
            self.args.seqdb,
        ]

        logger.info(f"Running jackhmmer with command `{' '.join(cmd)}`")

        subprocess.run(cmd)
        
