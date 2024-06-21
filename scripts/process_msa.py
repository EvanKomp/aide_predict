# scripts/prepare_msa.py
'''
* Author: Evan Komp
* Created: 5/9/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Cleanup and compute MSA weights from MSA built on WT protein.
This is handled by the MSAProcessing class, Adapted from the Mark's lab (see aide_predict/utils/msa.py for references)

Note that we can also add additional weights manually via IDs. 

'''
import os
import dvc.api
import json

from aide_predict.utils.msa import MSAProcessing, MSAProcessingArgs
from aide_predict.io.bio_files import read_fasta

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='./logs/prepare_msa.log')

PARAMS = dvc.api.params_show()['msaprocessing']

def main():
    # first prepare directories
    EXECDIR = os.getcwd()
    if not os.path.exists(os.path.join(EXECDIR, 'data', 'process_msa')):
        os.makedirs(os.path.join(EXECDIR, 'data', 'process_msa'))

    # prepare the arguments
    args = MSAProcessingArgs(
        theta=PARAMS['theta'],
        use_weights=PARAMS['use_weights'],
        preprocess_MSA=PARAMS['preprocess'],
        threshold_focus_cols_frac_gaps=PARAMS['threshold_focus_cols_frac_gaps'],
        threshold_sequence_frac_gaps=PARAMS['threshold_sequence_frac_gaps'],
        remove_sequences_with_indeterminate_AA_in_focus_cols=PARAMS['remove_sequences_with_indeterminate_AA_in_focus_cols'],
    )
    msa = MSAProcessing(args)

    # get the sqequence ID
    with open(os.path.join(EXECDIR, 'data', 'wt.fa'), 'r') as f:
        try:
            iterator = read_fasta(f)
            wt_id, _ = next(iterator)
        except StopIteration:
            wt_id = None
    logger.info(f'wt_id: {wt_id}')

    # process the MSA
    msa.process(
        MSA_location=os.path.join(EXECDIR, 'data', 'run_msa', 'alignment.a2m'),
        weights_location=os.path.join(EXECDIR, 'data', 'process_msa', 'weights.npy'),
        focus_seq_id=wt_id,
        additional_weights=None,
        new_a2m_location=os.path.join(EXECDIR, 'data', 'process_msa', 'alignment.a2m'),
    )
    metrics = {
        'msa_Neff': msa.Neff,
        'msa_num_seqs': msa.num_sequences,
        'msa_Neff_norm': msa.Neff / len(msa.seq_name_to_sequence[wt_id]),
    }
    with open(os.path.join(EXECDIR, 'data', 'metrics', 'process_msa.json'), 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()