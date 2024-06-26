# aide_predict/bespoke_models/tranception.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Wrapper of the Tranception model by Notin et al. See https://github.com/OATML-Markslab/Tranception/tree/main

This wrapper is designed around the author's main evaluation entrypoint script: https://github.com/OATML-Markslab/Tranception/blob/main/score_tranception_proteingym.py
For now, and options not exposed to that script are not exposed here.

Original script signature:

parser = argparse.ArgumentParser(description='Tranception scoring')
parser.add_argument('--checkpoint', type=str, help='Path of Tranception model checkpoint')
parser.add_argument('--model_framework', default='pytorch', type=str, help='Underlying framework [pytorch|JAX]')
parser.add_argument('--batch_size_inference', default=20, type=int, help='Batch size for inference')

#We may pass in all required information about the DMS via the provided reference files, or specify all relevant fields manually
parser.add_argument('--DMS_reference_file_path', default=None, type=str, help='Path to reference file with list of DMS to score')
parser.add_argument('--DMS_index', default=0, type=int, help='Index of DMS assay in reference file')
#Fields to be passed manually if reference file is not used
parser.add_argument('--target_seq', default=None, type=str, help='Full wild type sequence that is mutated in the DMS asssay')
parser.add_argument('--DMS_file_name', default=None, type=str, help='Name of DMS assay file')
parser.add_argument('--MSA_filename', default=None, type=str, help='Name of MSA (eg., a2m) file constructed on the wild type sequence')
parser.add_argument('--MSA_weight_file_name', default=None, type=str, help='Weight of sequences in the MSA (optional)')
parser.add_argument('--MSA_start', default=None, type=int, help='Sequence position that the MSA starts at (1-indexing)')
parser.add_argument('--MSA_end', default=None, type=int, help='Sequence position that the MSA ends at (1-indexing)')

parser.add_argument('--DMS_data_folder', type=str, help='Path to folder that contains all DMS assay datasets')
parser.add_argument('--output_scores_folder', default='./', type=str, help='Name of folder to write model scores to')
parser.add_argument('--deactivate_scoring_mirror', action='store_true', help='Whether to deactivate sequence scoring from both directions (Left->Right and Right->Left)')
parser.add_argument('--indel_mode', action='store_true', help='Flag to be used when scoring insertions and deletions. Otherwise assumes substitutions')
parser.add_argument('--scoring_window', default="optimal", type=str, help='Sequence window selection mode (when sequence length longer than model context size)')
parser.add_argument('--num_workers', default=10, type=int, help='Number of workers for model scoring data loader')
parser.add_argument('--inference_time_retrieval', action='store_true', help='Whether to perform inference-time retrieval')
parser.add_argument('--retrieval_inference_weight', default=0.6, type=float, help='Coefficient (alpha) used when aggregating autoregressive transformer and retrieval')
parser.add_argument('--MSA_folder', default='.', type=str, help='Path to MSA for neighborhood scoring')
parser.add_argument('--MSA_weights_folder', default=None, type=str, help='Path to MSA weights for neighborhood scoring')
parser.add_argument('--clustal_omega_location', default=None, type=str, help='Path to Clustal Omega (only needed with scoring indels with retrieval)')

Notes: 
- DMS_data is expected to be a csv with columns mutant, and optionally DMS_score, DMS_score_bin
- The tokenizer file tranception uses is a file in their repo, and is not part of the data files installed
with the package. We could copy it to our package, but for postarity instead retrieve it at runtime

'''
import os
import io
import json
from dataclasses import dataclass
import urllib

import pandas as pd

from aide_predict.bespoke_models.base import ModelWrapperRequiresMSA

# EXPOSURE OF TRANCEPTION
# MEANT TO MIMIC ORIGINAL SCRIPT
# DO NOT USE DIRECTLY
@dataclass
class _TranceptionArgs:
    """Defaults taken from the original script.
    """
    checkpoint: str # path to tranception model checkpoint
    model_framework: str = 'pytorch' # this should never change but including to match original script
    batch_size_inference: int = 20
    # these reference a protein gym setup where there is a mapping to existing datasets. We have our own data and do not need this
    DMS_reference_file_path: str = None # this should never change 
    DMS_index: int = 0 # this should never change
    # use these to specify data
    DMS_data_folder: str = None
    target_seq: str = None
    DMS_file_name: str = None
    MSA_filename: str = None
    MSA_folder: str = None
    MSA_weights_folder: str = None
    MSA_weight_file_name: str = None
    MSA_start: int = None
    MSA_end: int = None
    # output
    output_scores_folder: str = './'
    deactivate_scoring_mirror: bool = False
    indel_mode: bool = False
    scoring_window: str = "optimal"
    num_workers: int = 10
    inference_time_retrieval: bool = True
    retrieval_inference_weight: float = 0.6
    clustal_omega_location: str = None

class _Tranception:
    def __init__(self, args: _TranceptionArgs):
        # try to catch os errors before it goes into the bowels of tranception
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Tranception checkpoint not found at {args.checkpoint}")
        if not os.path.exists(os.path.join(args.DMS_data_folder, args.DMS_file_name)):
            raise FileNotFoundError(f"DMS assay file not found at {os.path.join(args.DMS_data_folder, args.DMS_file_name)}")
        if not os.path.exists(os.path.join(args.MSA_folder, args.MSA_filename)):
            raise FileNotFoundError(f"MSA file not found at {os.path.join(args.MSA_folder, args.MSA_filename)}")
        if args.MSA_weight_file_name and not os.path.exists(os.path.join(args.MSA_weights_folder, args.MSA_weight_file_name)):
            raise FileNotFoundError(f"MSA weight file not found at {os.path.join(args.MSA_weights_folder, args.MSA_weight_file_name)}")

        # check the expected fileformats
        # start with DMS data
        dms_data = pd.read_csv(os.path.join(args.DMS_data_folder, args.DMS_file_name))
        if 'mutant' not in dms_data.columns:
            raise ValueError("DMS data file must have a column named 'mutant'")

        self.args = args
        self._prepare_tranception_format_config()
    
    def _prepare_tranception_format_config(self):
        try:
            from transformers import PreTrainedTokenizerFast
            from tranception import config, model_pytorch
            import torch
            import tranception
        except ImportError:
            raise ImportError("Tranception is not installed. Please see the module's install instructions`")
        args = self.args

        
        tokenizer_json_url = 'https://raw.githubusercontent.com/OATML-Markslab/Tranception/main/tranception/utils/tokenizers/Basic_tokenizer'
        # this is in json format
        with urllib.request.urlopen(tokenizer_json_url) as url:
            tokenizer_json = json.loads(url.read().decode())
        # now make an in memory file for pretrined tokenizer to read
        tokenizer_file = io.StringIO()
        json.dump(tokenizer_json, tokenizer_file)

        # TAKEN FROM TRANCEPTION SCRIPT
        # tokenizer modified to load from url as above
        model_name = args.checkpoint.split("/")[-1]

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file,
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )

        target_seq=args.target_seq
        DMS_file_name=args.DMS_file_name
        DMS_id = DMS_file_name.split(".")[0]
        if args.inference_time_retrieval:
            MSA_data_file = args.MSA_folder + os.sep + args.MSA_filename if args.MSA_folder is not None else None
            MSA_weight_file_name = args.MSA_weights_folder + os.sep + args.MSA_weight_file_name if args.MSA_weights_folder is not None else None
            MSA_start = args.MSA_start - 1 # MSA_start based on 1-indexing
            MSA_end = args.MSA_end

        config = json.load(open(args.checkpoint+os.sep+'config.json'))
        config = tranception.config.TranceptionConfig(**config)
        config.attention_mode="tranception"
        config.position_embedding="grouped_alibi"
        config.tokenizer = tokenizer
        config.scoring_window = args.scoring_window

        if args.inference_time_retrieval:
            config.retrieval_aggregation_mode = "aggregate_indel" if args.indel_mode else "aggregate_substitution"
            config.MSA_filename=MSA_data_file
            config.full_protein_length=len(target_seq)
            config.MSA_weight_file_name=MSA_weight_file_name
            config.retrieval_inference_weight=args.retrieval_inference_weight
            config.MSA_start = MSA_start
            config.MSA_end = MSA_end
            if args.indel_mode:
                config.clustal_omega_location = args.clustal_omega_location
        else:
            config.retrieval_aggregation_mode = None

        if args.model_framework=="pytorch":
            model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.checkpoint,config=config)
            if torch.cuda.is_available():
                model.cuda()
        model.eval()

        if not os.path.isdir(args.output_scores_folder):
            os.mkdir(args.output_scores_folder)
        retrieval_type = '_retrieval_' + str(args.retrieval_inference_weight) if args.inference_time_retrieval else '_no_retrieval'
        mutation_type = '_indels' if args.indel_mode else '_substitutions'
        mirror_type = '_no_mirror' if args.deactivate_scoring_mirror else ''
        scoring_filename = args.output_scores_folder + os.sep + model_name + retrieval_type + mirror_type + mutation_type
        if not os.path.isdir(scoring_filename):
            os.mkdir(scoring_filename)
        scoring_filename += os.sep + DMS_id + '.csv'
        
        DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name, low_memory=False)
        self._dms_data = DMS_data
        self.model = model

    def run(self):
        """Actually call tranception.
        
        """
        all_scores = self.model.score_mutants(
            DMS_data = self._dms_data,
            target_seq = self.args.target_seq,
            scoring_mirror = not self.args.deactivate_scoring_mirror,
            batch_size_inference = self.args.batch_size_inference,
            num_workers = self.args.num_workers,
            output_folder = self.args.output_scores_folder
        )
        return all_scores
    
###############
# Public Api

class Tranception(ModelWrapperRequiresMSA):

    # We have a lot more metadata to check now.
    def check_metadata(self):
        """Ensures that eveything this model class needs is in the metadata folder.
        
        Tranception requires:
        - the input MSA, 
        """
        