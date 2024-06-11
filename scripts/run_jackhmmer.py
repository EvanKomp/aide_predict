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
'''
import os

import dvc.api

from aide_predict.utils.jackhmmer import Jackhmmer, JackhmmerArgs
from aide_predict.utils.msa import convert_sto_a2m

import logging
logging.basicConfig(level=logging.INFO, filemode='w', filename='./logs/run_jackhmmer.log')
logger = logging.getLogger(__name__)

PARAMS = dvc.api.params_show()['jackhmmer']
EXECDIR = os.getcwd()

if __name__ == '__main__':
    # check if jachmmer path is in env
    if 'JACKHMMER_PATH' in os.environ:
        executable = os.environ['JACKHMMER_PATH']
    else:
        executable = 'jackhmmer'

    # prepare the arguments
    args = JackhmmerArgs(
        seqdb=PARAMS['seqdb'],
        cpus=PARAMS['cpus'],
        iterations=PARAMS['iterations'],
        use_bitscores=PARAMS['use_bitscores'],
        tvalue=PARAMS['tvalue'],
        evalue=PARAMS['evalue'],
        popen=PARAMS['popen'],
        pextend=PARAMS['pextend'],
        mx=PARAMS['mx'],
        executable=executable
    )

    # check if we need to run jackhmmer at all
    # TODO
    if True:
        jackhmmer = Jackhmmer(args)
        jackhmmer.run(os.path.join(EXECDIR, 'data', 'wt.fasta'), os.path.join(EXECDIR, 'data', 'jackhmmer'))

        # convert to a2m and delete massive sto file
        convert_sto_a2m(
            os.path.join(EXECDIR, 'data', 'jackhmmer', 'jackhmmer.sto'),
            os.path.join(EXECDIR, 'data', 'jackhmmer', 'jackhmmer.a2m')
        )
    else:
        os.makedirs(os.path.join(EXECDIR, 'data', 'jackhmmer'))


