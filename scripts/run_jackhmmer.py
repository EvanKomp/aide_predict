# scripts/run_jackhmmer.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

This runs jackhmmer on the WT protein.
TODO: OR IT DOESN'T, and just creates an empty folder. We need to dynamically parse the params and determine of the
downstream zero shot model if present, requires and MSA. This metadata will be stored in the importable
portion of the code. 

Optional environmental variable JACKHMMER_PATH if it is not in the PATH.

Here we wrap around the pipeline from EVcouplings since they already wrote it and the MSAs produced
from their processing steps are used within the community.


Notes:

EXPECTED KWARGS for the Evcouplings pipeline:
- prefix: str
    The prefix file name that all output files will be based on.
- sequence_id: the id of the sequence
- sequence_file: the file containing the sequence eg. fasta with one sequence
- sequence_download_url : we can just pass None
- region: the region of the sequence to use, None for full sequence
- first_index: the first index of the sequence, leave as None for 1
- use_bitscores: whether to use bitscores or evalues
- domain_threshold: the threshold for domain detection
- sequence_threshold: the threshold for sequence detection
- database: the database to search against
- iterations: the number of iterations to run
- cpu: cpu count
- nobias: whether to use the bias correction
- reuse_alignment: whether to reuse the alignment, leave as false for running from scratch
- checkpoints_hmm : leave as false
- checkpoint_ali: leave as false
- jackhmmer: the path to the jackhmmer executable
- extract_annotation: sure why not
- seqid_filter : seqid thresholf to filter down highly similar sequences from msa
- hhfilter: the path to the hhfilter executable
- minimum_sequence_coverage: the minimum sequence coverage to be included. We will leave this as 0.0 because it will
    be filtered in the MSAProcessing class
- minimum_column_coverage: the minimum column coverage to be included. We will leave this as 0.0 because it will
    be filtered in the MSAProcessing class
- compute_num_effective_seqs: True so we can log it
- theta: identity threshold for clustering and computing weights. We can ignore this as we will compute weights later.
'''
import os

import dvc.api

from evcouplings.align.protocol import  standard as evcouplings_pipeline
from aide_prediction.io.bio_files import read_fasta

import logging
logging.basicConfig(level=logging.INFO, filemode='w', filename='./logs/run_jackhmmer.log')
logger = logging.getLogger(__name__)

PARAMS = dvc.api.params_show()['jackhmmer']
EXECDIR = os.getcwd()

if __name__ == '__main__':
    # check if jachmmer path is in env
    if 'JACKHMMER_PATH' in os.environ:
        jackhmmer = os.environ['JACKHMMER_PATH']
    else:
        jackhmmer = 'jackhmmer'

    if 'HHFILTER_PATH' in os.environ:
        hhfilter = os.environ['HHFILTER_PATH']
    else:
        hhfilter = 'hhfilter'

    # get the sqequence ID
    iterator = read_fasta(os.path.join(EXECDIR, 'data', 'wt.fasta'))
    sequence_id, _ = next(iterator)

    # prepare the arguments
    kwargs = {
        'prefix': os.path.join(EXECDIR, 'data', 'jackhmmer', 'msa_pipe'),
        'sequence_id': sequence_id,
        'sequence_file': os.path.join(EXECDIR, 'data', 'wt.fasta'),
        'sequence_download_url': None,
        'region': None,
        'first_index': None,
        'use_bitscores': PARAMS['use_bitscores'],
        'domain_threshold': PARAMS['domain_threshold'],
        'sequence_threshold': PARAMS['sequence_threshold'],
        'database': PARAMS['seqdb'],
        'iterations': PARAMS['iterations'],
        'cpu': PARAMS['cpus'],
        'nobias': True,
        'reuse_alignment': False,
        'checkpoints_hmm': False,
        'checkpoint_ali': False,
        'jackhmmer': jackhmmer,
        'extract_annotation': True,
        'seqid_filter': PARAMS['sequence_identity_filter'],
        'hhfilter': hhfilter,
        'minimum_sequence_coverage': 0.0,
        'minimum_column_coverage': 0.0,
        'compute_num_effective_seqs': False,
        'theta': 1.0
    }

    # check if we need to run jackhmmer at all
    # TODO
    if True:
        outcfg = evcouplings_pipeline.run(**kwargs)
        # outconfig is also saved to file.
    else:
        os.makedirs(os.path.join(EXECDIR, 'data', 'jackhmmer'))


