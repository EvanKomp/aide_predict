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
import time
import shutil
import json

import dvc.api
import numpy as np

from aide_predict.utils.common import convert_dvc_params
# from aide_predict.utils.msa import place_target_seq_at_top_of_msa

from aide_predict.io.bio_files import read_fasta

import logging
logging.basicConfig(level=logging.INFO, filemode='w', filename='./logs/run_msa.log')
logger = logging.getLogger(__name__)

PARAMS = convert_dvc_params(dvc.api.params_show())
EXECDIR = os.getcwd()

if __name__ == '__main__':
    t0 = time.time()
    # create outpit directory
    outdir = os.path.join(EXECDIR, 'data', 'run_msa')
    os.makedirs(outdir, exist_ok=True)

    # skip everything if the user specified MSA was not needed.
    if not PARAMS.use_msa:
        logger.info('MSA not needed for this model.')
        exit(0)
    # get the sqequence ID
    with open(os.path.join(EXECDIR, 'data', 'wt.fa'), 'r') as f:
        try:
            iterator = read_fasta(f)
            sequence_id, _ = next(iterator)
        except StopIteration:
            sequence_id = None

    # determine what type of MSA we will do
    if PARAMS.msa_creation.msa_mode == 'jackhmmer':
        from evcouplings.align.protocol import  standard as evcouplings_pipeline

        # check if jachmmer path is in env
        if 'JACKHMMER_PATH' in os.environ:
            jackhmmer = os.environ['JACKHMMER_PATH']
        else:
            jackhmmer = 'jackhmmer'

        if 'HHFILTER_PATH' in os.environ:
            hhfilter = os.environ['HHFILTER_PATH']
        else:
            hhfilter = 'hhfilter'


        # prepare the arguments
        kwargs = {
            'prefix': os.path.join(outdir, 'jck'),
            'sequence_id': sequence_id,
            'sequence_file': os.path.join(EXECDIR, 'data', 'wt.fa'),
            'sequence_download_url': None,
            'region': None,
            'first_index': None,
            'use_bitscores': PARAMS.msa_creation.jackhmmer.use_bitscores,
            'domain_threshold': PARAMS.msa_creation.jackhmmer.domain_threshold,
            'sequence_threshold': PARAMS.msa_creation.jackhmmer.sequence_threshold,
            'database': 'db_'+PARAMS.msa_creation.jackhmmer.seqdb,
            'iterations': PARAMS.msa_creation.jackhmmer.iterations,
            'cpu': PARAMS.msa_creation.jackhmmer.cpus,
            'nobias': True,
            'reuse_alignment': False,
            'checkpoints_hmm': False,
            'checkpoints_ali': False,
            'jackhmmer': jackhmmer,
            'extract_annotation': True,
            'seqid_filter': PARAMS.msa_creation.jackhmmer.sequence_identity_filter,
            'hhfilter': hhfilter,
            'minimum_sequence_coverage': PARAMS.msa_creation.jackhmmer.minimum_sequence_coverage,
            'minimum_column_coverage': PARAMS.msa_creation.jackhmmer.minimum_column_coverage,
            'compute_num_effective_seqs': False,
            'theta': PARAMS.msa_creation.jackhmmer.theta
        }
        # add the user's database locations
        for db_name, db_loc in PARAMS.sequence_databases.__dict__.items():
            kwargs['db_'+db_name] = db_loc
            
        logger.info(f"Running evcouplings jackhmmer pipeline with params {kwargs}")

        outcfg = evcouplings_pipeline(**kwargs)
        logger.info(f"jackhmmer complete. evcouplings output config: {outcfg}")
        # outconfig is also saved to file.

        # move the resulting a2m file to the output directory
        shutil.move(outcfg['alignment_file'], os.path.join(outdir, 'alignment.a2m'))


    elif PARAMS.msa_creation.msa_mode == 'starting_sequences':
        # here we simply need to align the sequences
        # or use existing alignment
        # at data/starting_sequences.fa/a2m
        # potentially also align actives in data/experimental_data.csv
        if PARAMS.msa_creation.starting_sequences.prealigned:
            # rename to alignment.a2m
            shutil.copy(os.path.join(EXECDIR, 'data', 'starting_sequences.a2m'), os.path.join(outdir, 'alignment.a2m'))
        else:
            raise NotImplementedError('Not yet implemented.')
        
        if PARAMS.msa_creation.starting_sequences.add_training_sequences:
            # need to add the additional sequences to the alignment
            raise NotImplementedError('Not yet implemented.')

    
    else:
        raise ValueError(f"Unknown MSA mode: {PARAMS.msa_creation.msa_mode}")
    
    # ensure that the resulting alignment has the WT sequence at the top
    # if sequence_id:
    #     place_target_seq_at_top_of_msa(msa_file=os.path.join(outdir, 'alignment.a2m'), target_seq_id=sequence_id)

    # get the count of sequences in the MSA
    with open(os.path.join(outdir, 'alignment.a2m'), 'r') as f:
        sequences = read_fasta(f)
        sequences = list(sequences)
        num_seqs = len(sequences)
        sequences = [list(s) for _, s in sequences]
        sequence_array = np.array(sequences)
        num_gaps = np.sum(sequence_array == '-') + np.sum(sequence_array == '.')
        gap_frac = num_gaps / (sequence_array.size)
    metrics = {
        'msa_num_seqs': num_seqs,
        'msa_gap_frac': gap_frac,
    }
    with open(os.path.join(EXECDIR, 'data', 'metrics', 'run_msa.json'), 'w') as f:
        json.dump(metrics, f)


    
    t1 = time.time()
    t = t1 - t0
    logger.info(f"Successfully terminated, time elapsed: {t/60} mins.")


