# aide_predict/utils/msa.py
'''
* MSAProcessing class Refactored from Frazer et al. 
@article{Frazer2021DiseaseVP,
  title={Disease variant prediction with deep generative models of evolutionary data.},
  author={Jonathan Frazer and Pascal Notin and Mafalda Dias and Aidan Gomez and Joseph K Min and Kelly P. Brock and Yarin Gal and Debora S. Marks},
  journal={Nature},
  year={2021}
}

* Author: Evan Komp
* Created: 5/8/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Peocessing of MSAs for preparation of input data for the zero-shot model.
Note that The MSAProcessing class IS A REFACTORING  of the MSA processing class from The marks Lab https://github.com/OATML-Markslab/EVE/blob/master/utils/data_utils.py
Credit is given to them for the original implementation and the methodology of sequence weighting.
Here, we make it more pythonic and readbale, as well as an order of magnitude speed up.

In addition to refactoring, we add some additional functionality:
- Manual sequence weighting. The motivation here is sequences coming from environmental classes that are more important for
   your design task
- A focus seq need not be present, in which case all columns are considered focus columns and contribute to weight computation
- One Hot encoding is reworked to use sklearn's OneHotEncoder instead of a loop of loops, with about an order of magnitude speedup
- Weight computation leverages numpy array indexing instead of a loop, and if torch is available
  and advanced hardware is present, GPU is used.
      Tested on 10000 protein sequences sequences of length 55:
        - original: 8.9 seconds
        - cpu array operations: 1.2 seconds (7.4x speedup)
        - gpu array operations: 0.2 seconds (44.5x speedup)
- other minor speedups with array operations
'''
import os
from dataclasses import dataclass
from collections import defaultdict
import subprocess
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from sklearn.preprocessing import OneHotEncoder

from aide_predict.io.bio_files import read_fasta, write_fasta

import logging
logger = logging.getLogger(__name__)


@dataclass
class MSAProcessingArgs:
    """Defaults are EVcouplings defaults for running Jackhmmer.
    Parameters:
    - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01   
    - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that;
        otherwise compute weights from scratch and store them at weights_location
    - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
    - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
        - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
        - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
    - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
        - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
        - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
    - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
    - weight_computation_batch_size: (int) Number of sequences to compute weights per batch.
      N X N array must be stored in cpu or gpu memory, this parameter limits it to avoid memory errors.
      10000 is a good default value.
    """
    theta: float = 0.2
    use_weights: bool = True
    preprocess_MSA: bool = True
    threshold_sequence_frac_gaps: float = 0.5
    threshold_focus_cols_frac_gaps: float = 0.3
    remove_sequences_with_indeterminate_AA_in_focus_cols: bool = True
    weight_computation_batch_size: int = 10000

class MSAProcessing:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    def __init__(self, args: MSAProcessingArgs):
        self.args = args

    def process(self, MSA_location: str, weights_location: str, focus_seq_id: str = None, additional_weights: dict = None, new_a2m_location: str = None):
        """
        Parameters:
        - MSA_location: (path) Location of the MSA data. Constraints on input MSA format: 
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corresponding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - focus_seq_id: (str or None) The sequence id of the focus sequence. If None, all columns in the MSA are considered focus columns.
        - weights_location: (path) Location to load from/save to the sequence weights
        - additional_weights: (dict) Additional weights to apply to sequences. keys should be sequence ids, values should be weights
           the additional weights are scaler multiplied by the computed weights, thus if you have some sequences that are more important
        - new_a2m_location: (path) If preprocess is turned on, saves the new MSA (eg. with some sequences dropped) to this location)
        """
        if new_a2m_location is None and self.args.preprocess_MSA:
            raise ValueError("Preprocessing MSA is on, but no new_a2m_location provided")

        self.new_a2m_location = new_a2m_location    
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.focus_seq_id = focus_seq_id
        self._read_alignment()

        # check the additional weights ids
        if additional_weights is not None:
            for seq_name in additional_weights.keys():
                if '>'+seq_name not in self.seq_name_to_sequence:
                    raise ValueError(f"Sequence {seq_name} not found in MSA")

        self._cut_bad_seqs_and_columns()
        self._encode_sequences()
        self._compute_weights(additional_weights)
        

    def _read_alignment(self):
        """Load the MSA from file into python objects."""
        self._aa_dict = dict(zip(self.alphabet, range(len(self.alphabet))))
        
        # read the ids and sequences
        self.seq_name_to_sequence = defaultdict(str)
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            found_focus_seq = False
            for i, line in enumerate(msa_data):
                line = line.rstrip()

                if line.startswith(">"):
                    name = line
                    if i == 0:
                        if self.focus_seq_id is None or name == '>'+self.focus_seq_id:
                            found_focus_seq = True
                            self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line
        if self.focus_seq_id is not None and not found_focus_seq:
            raise ValueError(f"Focus sequence {self.focus_seq_id} not found in MSA")
        logger.info(f"Loaded MSA with {len(self.seq_name_to_sequence)} sequences, target sequence: {self.focus_seq_name}")

    def _cut_bad_seqs_and_columns(self):
        """Pre-process the MSA to remove bad sequences and columns."""
        logger.info(f"Original width of MSA: {len(self.seq_name_to_sequence[self.focus_seq_name])}")
        if self.args.preprocess_MSA:
            msa_df = pd.DataFrame.from_dict(self.seq_name_to_sequence, orient='index', columns=['sequence'])
            # Data clean up - make all gaps "-" and upper case
            sequences = np.array([list(seq) for seq in msa_df.sequence])
            sequences[sequences == '.'] = '-'
            sequences = np.char.upper(sequences)
            msa_df.sequence = [''.join(seq) for seq in sequences]

            # Remove columns that would be gaps in the wild type
            if self.focus_seq_id is not None:
                non_gap_wt_cols = [aa!='-' for aa in msa_df.sequence[self.focus_seq_name]]
                sequences = np.array([list(seq) for seq in msa_df.sequence])
                sequences = sequences[:, non_gap_wt_cols]
                msa_df['sequence'] = [''.join(seq) for seq in sequences]

                if self.args.threshold_sequence_frac_gaps < 0 or self.args.threshold_sequence_frac_gaps > 1:
                    raise ValueError("Invalid fragment filtering parameter")
                if self.args.threshold_focus_cols_frac_gaps < 0 or self.args.threshold_focus_cols_frac_gaps > 1:
                    raise ValueError("Invalid focus position filtering parameter")
            logger.info(f"Removed gap columns in target sequence: remaining width {len(msa_df.sequence[self.focus_seq_name])}")
            
            msa_array = np.array([list(seq) for seq in msa_df.sequence])
            gaps_array = msa_array == '-'

            # Identify fragments with too many gaps
            seq_gaps_frac = gaps_array.mean(axis=1)
            seq_below_threshold = seq_gaps_frac <= self.args.threshold_sequence_frac_gaps
            logger.info("Proportion of sequences dropped due to fraction of gaps: "+str(round(float(1 - seq_below_threshold.sum()/seq_below_threshold.shape)*100,2))+"%")
            # Identify focus columns
            columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
            index_cols_below_threshold = columns_gaps_frac <= self.args.threshold_focus_cols_frac_gaps
            logger.info("Proportion of non-focus columns removed: "+str(round(float(1 - index_cols_below_threshold.sum()/index_cols_below_threshold.shape)*100,2))+"%")
            # Lower case non focus cols and filter fragment sequences
            msa_array[:,~index_cols_below_threshold] = np.char.lower(msa_array[:,~index_cols_below_threshold])
            msa_df.sequence = [''.join(seq) for seq in msa_array]
            msa_df = msa_df[seq_below_threshold]
            # Overwrite seq_name_to_sequence with clean version
            self.seq_name_to_sequence = defaultdict(str)
            self.seq_name_to_sequence.update(msa_df.sequence.to_dict())
            
            # save as a new a2m file
            if self.new_a2m_location is not None:
                with open(self.new_a2m_location, 'w') as f:
                    f.write(f"{self.focus_seq_name}\n")
                    f.write(msa_df.sequence[self.focus_seq_name]+'\n')
                    for seq_name, sequence in self.seq_name_to_sequence.items():
                        f.write(seq_name+'\n')
                        f.write(sequence+'\n')

    
    def _encode_sequences(self):
        """Encode the sequences into one-hot format."""
        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        if self.focus_seq_id is None:
            self.focus_cols = list(range(len(self.focus_seq)))
        else:
            self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s!='-'] 
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Connect local sequence index with uniprot index (index shift inferred from 1st row of MSA)
        try:
            focus_loc = self.focus_seq_name.split("/")[-1]
            split = focus_loc.split("-")
            start = split[0]
            stop = split[1]
        except IndexError:
            start,stop = 1, len(self.focus_seq)

        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col+int(start):self.focus_seq[idx_col] for idx_col in self.focus_cols} 
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col+int(start):idx_col for idx_col in self.focus_cols} 

        # Move all letters to CAPS; keeps focus columns only
        for seq_name,sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".","-")
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.args.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet)+['-'])
            sequence_array = np.array([list(seq) for seq in self.seq_name_to_sequence.values()])
            seq_name_array = np.array(list(self.seq_name_to_sequence.keys()))
            valid_characters = np.isin(sequence_array, list(self.alphabet)+['-'])
            valid_rows = np.all(valid_characters, axis=1)
            invalid_seq_names = seq_name_array.reshape(-1)[~valid_rows.reshape(-1)]

            for seq_name in invalid_seq_names:
                del self.seq_name_to_sequence[seq_name]

        # Encode the sequences
        logger.info("Encoding sequences")
        alphabet = list(self.alphabet) + ['-']
        seq_array = np.array([list(seq) for seq in self.seq_name_to_sequence.values()])
        ohe = OneHotEncoder(categories=[list(alphabet)]*seq_array.shape[1], sparse_output=False)
        self.one_hot_encoding = ohe.fit_transform(seq_array)
        # this is shape (N_seq, seq_len*alphabet_size), reshape
        self.one_hot_encoding = self.one_hot_encoding.reshape((self.one_hot_encoding.shape[0], len(self.focus_cols), len(alphabet)))
        # remove the gap column
        self.one_hot_encoding = self.one_hot_encoding[:, :, :-1]
        logger.info("One hot encoding complete")

    def _compute_weights(self, additional_weights: dict):
        if self.args.use_weights:
            try:
                self.weights = np.load(file=self.weights_location)
                logger.info("Loaded sequence weights from disk")
            except:
                logger.info("Computing sequence weights")
                list_seq = self.one_hot_encoding
                list_seq = list_seq.reshape((list_seq.shape[0], list_seq.shape[1] * list_seq.shape[2]))
                
                weights = []
                for i in tqdm(range(0, len(list_seq), self.args.weight_computation_batch_size), desc="Computing weights"):
                    weights.append(_compute_weight(list_seq[i:i+self.args.weight_computation_batch_size], self.args.theta))
                self.weights = np.concatenate(weights)

        else:
            # If not using weights, use an isotropic weight matrix
            logger.info("Not weighting sequence data by MSA quality")
            self.weights = np.ones(self.one_hot_encoding.shape[0])

        self.Neff = np.sum(self.weights)
        self.num_sequences = self.one_hot_encoding.shape[0]
        if self.args.use_weights:
            logger.info(f"Neff (effictive sequence count) = {self.Neff}")
        else:
            logger.info(f"Not using sequence weights, N = {self.num_sequences}")

        if additional_weights is not None:
            logger.info("Applying manual sequence weights as a factor")
            for seq_id, weight in additional_weights.items():
                # map the weight to the sequence id...
                index = list(self.seq_name_to_seq_name.keys()).index('>'+seq_id)
                self.weights[index] *= weight
        if self.args.use_weights:
            np.save(file=self.weights_location, arr=self.weights)

        logger.info(f"Data Shape = {self.one_hot_encoding.shape}")
        self.seq_name_to_weight = dict(zip(self.seq_name_to_sequence.keys(), self.weights))


def _compute_weight(list_seq, theta):
    """Array operation to compute the weight of a sequence.
    
    Will use GPU if available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        elif torch.backends.mps.is_available():
            DEVICE = torch.device('mps')
        else:
            DEVICE = torch.device('cpu')

        list_seq = torch.from_numpy(list_seq.astype(np.float32)).to(DEVICE)
        logger.info(f"Using {DEVICE} for sequence weighting")

        # Compute the dot product of each row with itself
        seq_dot_seq = torch.einsum('ij,ij->i', list_seq, list_seq)
        torch_active = True
    except ImportError:
        torch_active = False
        logger.info("Using numpy for sequence weighting")
        seq_dot_seq = np.einsum('ij,ij->i', list_seq, list_seq)

    # Compute the dot product of list_seq with each row
    list_seq_dot_seq = list_seq @ list_seq.T

    # Divide the two results element-wise
    denom = list_seq_dot_seq / seq_dot_seq[:, None]

    # Compute the weights
    if torch_active:
        denom = denom.cpu().numpy()

    weights = np.sum(denom > 1 - theta, axis=1)
    # if the denom is 0, skip it
    new_weights = np.zeros(len(weights))
    new_weights[weights > 0] = 1 / weights[weights > 0]
    return new_weights

def place_target_seq_at_top_of_msa(msa_file: str, target_seq_id: str):
    """Place the target sequence at the top of the MSA.
    
    Also checks for if the top sequence has a position range, and if not,
    adds it as the whole sequences
    """
    with open(msa_file, 'r') as f:
        sequences = dict(list(read_fasta(f)))
    if target_seq_id not in sequences:
        raise ValueError(f"Target sequence {target_seq_id} not found in MSA")
    # make sure target seq is at the top of the dict
    target_seq = sequences[target_seq_id]
    del sequences[target_seq_id]

    # check if the target sequence has a position range
    try:
        focus_loc = target_seq_id.split("/")[-1]
        split = focus_loc.split("-")
        start = split[0]
        stop = split[1]
        assert int(start) > 0
        assert int(stop) > 0
    except (IndexError, AssertionError):
        start = 1
        stop = len(target_seq)
        target_seq_id = f"{target_seq_id}/{start}-{stop}"

    sequences = [(target_seq_id, target_seq)] + [(k, v) for k, v in sequences.items()]
    with open(msa_file, 'w') as f:
        write_fasta(sequences, f)