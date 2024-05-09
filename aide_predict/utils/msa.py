# aide_predict/utils/msa.py
'''
* Refactored from Frazer et al. 
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
Note that THIS IS A REFACTORING  of the MSA processing class from The marks Lab https://github.com/OATML-Markslab/EVE/blob/master/utils/data_utils.py
Credit is given to them for the original implementation and the methodology of sequence weighting.
Here, we make it more pythonic and readbale.

In addition to refactoring, we add some additional functionality:
- Manual sequence weighting. The motivation here is sequences coming from environmental classes that are more important for
   your design task
'''
import os
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

def prepare_sto_file(sto_file)


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
    """
    theta: float = 0.2
    use_weights: bool = True
    preprocess_MSA: bool = True
    threshold_sequence_frac_gaps: float = 0.5
    threshold_focus_cols_frac_gaps: float = 0.3
    remove_sequences_with_indeterminate_AA_in_focus_cols: bool = True

class MSAProcessing:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    def __init__(self, args: MSAProcessingArgs):
        self.args = args

    def process(self, MSA_location: str, weights_location: str, focus_seq_name: str, additional_weights: dict = None):
        """
        Parameters:
        - MSA_location: (path) Location of the MSA data. Constraints on input MSA format: 
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corresponding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - weights_location: (path) Location to load from/save to the sequence weights
        - additional_weights: (dict) Additional weights to apply to sequences. keys should be sequence ids, values should be weights
           the additional weights are scaler multiplied by the computed weights, thus if you have some sequences that are more important
        """
        # check the additional weights ids
        for seq_name in additional_weights.keys():
            if seq_name not in self.seq_name_to_sequence:
                raise ValueError(f"Sequence {seq_name} not found in MSA")
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.focus_seq_name = focus_seq_name
        self._read_alignment()
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
                    if name == self.focus_seq_name:
                        found_focus_seq = True
                else:
                    self.seq_name_to_sequence[name] += line
        if not found_focus_seq:
            raise ValueError(f"Focus sequence {self.focus_seq_name} not found in MSA")
        logger.info(f"Loaded MSA with {len(self.seq_name_to_sequence)} sequences, target sequence: {self.focus_seq_name}")

    def _cut_bad_seqs_and_columns(self):
        """Pre-process the MSA to remove bad sequences and columns."""
        logger.inf("Original width of MSA: ", len(self.seq_name_to_sequence[self.focus_seq_name]))
        if self.args.preprocess_MSA:
            msa_df = pd.DataFrame.from_dict(self.seq_name_to_sequence, orient='index', columns=['sequence'])
            # Data clean up - make all gaps "-" and upper case
            msa_df.sequence = msa_df.sequence.apply(lambda x: x.replace(".","-")).apply(lambda x: ''.join([aa.upper() for aa in x]))

            # Remove columns that would be gaps in the wild type
            non_gap_wt_cols = [aa!='-' for aa in msa_df.sequence[self.focus_seq_name]]
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa for aa,non_gap_ind in zip(x, non_gap_wt_cols) if non_gap_ind]))
            if self.args.threshold_sequence_frac_gaps < 0 or self.args.threshold_sequence_frac_gaps > 1:
                raise ValueError("Invalid fragment filtering parameter")
            if self.args.threshold_focus_cols_frac_gaps < 0 or self.args.threshold_focus_cols_frac_gaps > 1:
                raise ValueError("Invalid focus position filtering parameter")
            logger.info(f"Removed gap columns in target sequence: remaining width {len(msa_df.sequence[self.focus_seq_name])}")
            
            msa_array = np.array([list(seq) for seq in msa_df.sequence])
            gaps_array = np.array(list(map(lambda seq: [aa=='-' for aa in seq], msa_array)))

            # Identify fragments with too many gaps
            seq_gaps_frac = gaps_array.mean(axis=1)
            seq_below_threshold = seq_gaps_frac <= self.args.threshold_sequence_frac_gaps
            print("Proportion of sequences dropped due to fraction of gaps: "+str(round(float(1 - seq_below_threshold.sum()/seq_below_threshold.shape)*100,2))+"%")
            # Identify focus columns
            columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
            index_cols_below_threshold = columns_gaps_frac <= self.args.threshold_focus_cols_frac_gaps
            print("Proportion of non-focus columns removed: "+str(round(float(1 - index_cols_below_threshold.sum()/index_cols_below_threshold.shape)*100,2))+"%")
            # Lower case non focus cols and filter fragment sequences
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in zip(x, index_cols_below_threshold)]))
            msa_df = msa_df[seq_below_threshold]
            # Overwrite seq_name_to_sequence with clean version
            self.seq_name_to_sequence = defaultdict(str)
            for seq_idx in range(len(msa_df['sequence'])):
                self.seq_name_to_sequence[msa_df.index[seq_idx]] = msa_df.sequence[seq_idx]
    
    def _encode_sequences(self):
        """Encode the sequences into one-hot format."""
        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s!='-'] 
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Connect local sequence index with uniprot index (index shift inferred from 1st row of MSA)
        try:
            focus_loc = self.focus_seq_name.split("/")[-1]
            start,stop = focus_loc.split("-")
        except IndexError:
            start,stop = 0, len(self.focus_seq)

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
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name,sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                del self.seq_name_to_sequence[seq_name]

        # Encode the sequences
        logger.info("Encoding sequences")
        self.one_hot_encoding = np.zeros((len(self.seq_name_to_sequence.keys()),len(self.focus_cols),len(self.alphabet)))
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            for j,letter in enumerate(sequence):
                if letter in self.aa_dict: 
                    k = self.aa_dict[letter]
                    self.one_hot_encoding[i,j,k] = 1.0
        
    def _compute_weights(self, additional_weights: dict):
        if self.args.use_weights:
            try:
                self.weights = np.load(file=self.weights_location)
                logger.info("Loaded sequence weights from disk")
            except:
                logger.info("Computing sequence weights")
                list_seq = self.one_hot_encoding
                list_seq = list_seq.reshape((list_seq.shape[0], list_seq.shape[1] * list_seq.shape[2]))
                def compute_weight(seq):
                    number_non_empty_positions = np.dot(seq,seq)
                    if number_non_empty_positions>0:
                        denom = np.dot(list_seq,seq) / np.dot(seq,seq) 
                        denom = np.sum(denom > 1 - self.args.theta) 
                        return 1/denom
                    else:
                        return 0.0
                self.weights = np.array(list(map(compute_weight,list_seq)))
                np.save(file=self.weights_location, arr=self.weights)

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
                index = list(self.seq_name_to_sequence.keys()).index(seq_id)
                self.weights[index] *= weight

        logger.info(f"Data Shape = {self.one_hot_encoding.shape}")
        self.seq_name_to_weight = dict(zip(self.seq_name_to_sequence.keys(), self.weights))

