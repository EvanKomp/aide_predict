# aide_predict/bespoke_models/predictors/saprot.py
'''
* Author: Evan Komp
* Created: 7/16/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper around SaProt model. Please see here and all credit to the oroginal authors for their method and model:
https://www.biorxiv.org/content/10.1101/2023.10.01.560349v2

'''
import os
import subprocess
import shutil
import json
import numpy as np
from typing import List, Optional, Union
import warnings

import numpy as np
from tqdm import tqdm

from aide_predict.bespoke_models.base import RequiresFixedLengthMixin, RequiresStructureMixin, CacheMixin, ExpectsNoFitMixin
from aide_predict.bespoke_models.predictors.pretrained_transformers import LikelihoodTransformerBase, MarginalMethod
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence, ProteinStructure
from aide_predict.utils.common import MessageBool

foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"

try:
    import torch
    from transformers import EsmTokenizer, EsmForMaskedLM
    # check which for foldseek to see if the executable is available
    fs = shutil.which("foldseek")
    if fs is None:
        raise ImportError("FoldSeek executable not found.")

    AVAILABLE = MessageBool(True, "SaProt model is available.")
except ImportError:
    AVAILABLE = MessageBool(False, "SaProt model is not available. Please install the transformers library and make sure foldseek is in path.")

import logging
logger = logging.getLogger(__name__)

def get_structure_tokens(structure: ProteinStructure, foldseek_path: str, process_id: int = 0, plddt_threshold: float = 70., return_seq_tokens: bool = False) -> str:
    """
    Extract structure tokens from a ProteinStructure using FoldSeek.
    
    Args:
        structure (ProteinStructure): The protein structure to process.
        foldseek_path (str): Path to the FoldSeek executable.
        process_id (int): Process ID for temporary files. Used for parallel processing.
        plddt_threshold (float): Threshold for pLDDT scores. Regions below this are masked.
    
    Returns:
        str: A string of structure tokens.
    """
    tmp_save_path = f"get_struc_seq_{process_id}.tsv"
    cmd = f"{foldseek_path} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {structure.pdb_file} {tmp_save_path}"
    subprocess.run(cmd, shell=True, check=True)

    with open(tmp_save_path, "r") as r:
        line = r.readline()
        _, seq, struc_seq = line.split("\t")[:3]

    # Mask low pLDDT regions if pLDDT file is available
    if structure.plddt_file:
        with open(structure.plddt_file, "r") as r:
            plddts = np.array(json.load(r)["plddt"])
        
        # Mask regions with pLDDT < threshold
        indices = np.where(plddts < plddt_threshold)[0]
        np_seq = np.array(list(struc_seq))
        np_seq[indices] = "#"
        struc_seq = "".join(np_seq)

    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    if not return_seq_tokens:
        return struc_seq
    else:
        return struc_seq, seq


class SaProtLikelihoodWrapper(RequiresStructureMixin, LikelihoodTransformerBase):
    _available = AVAILABLE

    def __init__(
        self,
        metadata_folder: str = None,
        model_checkpoint: str = 'westlake-repl/SaProt_650M_AF2',
        marginal_method: MarginalMethod = MarginalMethod.WILDTYPE,
        positions: list = None,
        pool: bool = True,
        flatten: bool = True,
        wt: str = None,
        batch_size: int = 2,
        device: str = 'cpu',
        foldseek_path: str = 'foldseek'
    ):
        super().__init__(
            metadata_folder=metadata_folder,
            marginal_method=marginal_method,
            positions=positions,
            pool=pool,
            flatten=flatten,
            wt=wt,
            batch_size=batch_size,
            device=device
        )
        self.model_checkpoint = model_checkpoint
        self.foldseek_path = foldseek_path
        self._aa_to_index = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self._vectorized_aa_to_index = np.vectorize(lambda x: self._aa_to_index.get(x, -1))
        logger.debug(f"SaProt model initialized with {self.__dict__}")


    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'SaProtLikelihoodWrapper':
        self.fitted_ = True
        return self
    
    def _load_model(self) -> None:
        self.model_ = EsmForMaskedLM.from_pretrained(self.model_checkpoint).to(self.device)
        self.tokenizer_ = EsmTokenizer.from_pretrained(self.model_checkpoint)
    
    def _cleanup_model(self) -> None:
        del self.model_
        del self.tokenizer_

    def _tokenize(self, sequences: List[str], on_device: bool=True) -> "torch.Tensor":
        if on_device:
            return self.tokenizer_(sequences, add_special_tokens=True, return_tensors='pt', padding=True).to(self.device)
        else:
            return self.tokenizer_(sequences, add_special_tokens=False, return_tensors='np', padding=True)

    def _prepare_input_sequence(self, sequence: ProteinSequence) -> str:
        """Convert a protein sequence into a string ready for tokenization.
        
        NOTE: if the input sequence has no structure, WT structure will be used, and if that is not available, no structure tokens will be used.
        """

        if sequence.structure is None:
            
            if self.wt is not None and self.wt.structure:
                struc_tokens = get_structure_tokens(self.wt.structure, self.foldseek_path)
                return ''.join([a + b.lower() for a, b in zip(str(sequence).upper(), struc_tokens)])
            else:
                return ''.join([aa + '#' for aa in str(sequence).upper()])
        
        else:
            struc_tokens = get_structure_tokens(sequence.structure, self.foldseek_path)
            return ''.join([a + b.lower() for a, b in zip(str(sequence).upper(), struc_tokens)])

    def _compress_saprot_probs(self, probs: np.ndarray) -> np.ndarray:
        """
        Compress SaProt log probabilities to standard amino acid probabilities.

        Args:
            lprobs (np.ndarray): probabilities from SaProt model, shape (L, 446).

        Returns:
            np.ndarray: Compressed probabilities for standard amino acids, shape (L, 20).
        """
        
        # Reshape the array to separate amino acids and their structural variations
        # 5 is the offset for special tokens, 21 is the number of variations per AA (20 structural + 1 mask)
        reshaped_probs = probs[:, 5:-21].reshape(probs.shape[0], 20, 21)
        
        # Sum probabilities across structural variations for each amino acid
        compressed_probs = reshaped_probs.sum(axis=2)
        
        # Normalize the probabilities
        compressed_probs /= compressed_probs.sum(axis=1, keepdims=True)

        compressed_log_probs = np.log(compressed_probs)
        
        return compressed_log_probs

    def _compute_log_likelihoods(self, X: ProteinSequences, mask_positions: List[List[int]] = None) -> List[np.ndarray]:
        prepared_sequences = []
        for i, seq in enumerate(X):

            prepared_seq = self._prepare_input_sequence(seq)
            
            if mask_positions and i < len(mask_positions):
                seq_chars = list(prepared_seq)
                for pos in mask_positions[i]:
                    seq_chars[pos*2] = "#"
                prepared_seq = ''.join(seq_chars)
            
            prepared_sequences.append(prepared_seq)
        
        with torch.no_grad():
            tokenized = self._tokenize(prepared_sequences, on_device=True)
            logits = self.model_(**tokenized).logits
            probs = torch.softmax(logits, dim=-1)

            results = []
            for j in range(len(tokenized.input_ids)):
                probs_j = probs[j][tokenized.attention_mask[j].bool()].cpu().numpy()
                compressed_log_probs = self._compress_saprot_probs(probs_j[1:-1])  # Remove start and end tokens
                results.append(compressed_log_probs)

        return results

    def _index_log_probs(self, log_probs: np.ndarray, sequences: ProteinSequences) -> np.ndarray:
        """
        Index log probabilities by the observed amino acids in the sequences.

        Args:
            log_probs (np.ndarray): Log probabilities for each sequence and position, shape (n_sequences, seq_length, 20).
            sequences (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: Indexed log probabilities, shape (n_sequences, seq_length).
        """
        # Convert sequences to a 2D numpy array of characters
        seq_array = np.array([list(str(seq).upper()) for seq in sequences])
        
        # Convert amino acids to indices
        aa_indices = self._vectorized_aa_to_index(seq_array)
        
        # Create row and column indices
        rows = np.arange(log_probs.shape[0])
        rows_expanded = np.expand_dims(rows, axis=0)
        
        # Index the log probabilities
        return log_probs[rows_expanded, aa_indices]


    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return super().get_feature_names_out(input_features)