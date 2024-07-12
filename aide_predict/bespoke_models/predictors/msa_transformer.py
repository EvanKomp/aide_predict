# aide_predict/bespoke_models/predictors/msa_transformer.py
'''
* Author: Evan Komp
* Created: 7/8/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import warnings
from typing import List, Union, Optional

import numpy as np
import tqdm


from aide_predict.bespoke_models.base import RequiresMSAMixin, RequiresFixedLengthMixin
from aide_predict.bespoke_models.predictors.pretrained_transformers import LikelihoodTransformerBase, MarginalMethod
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool

try:
    import esm
    import torch
    from esm import pretrained
    AVAILABLE = MessageBool(True, "MSA Transformer is available.")
except ImportError:
    AVAILABLE = MessageBool(False, "MSA Transformer requires fair-esm, which is not installed.")

class MSATransformerLikelihoodWrapper(RequiresMSAMixin, RequiresFixedLengthMixin, LikelihoodTransformerBase):
    """
    A wrapper for the MSA Transformer model to compute log likelihoods for protein sequences.

    This class uses the MSA Transformer model to calculate log likelihoods for protein sequences
    based on multiple sequence alignments (MSAs). It supports various marginal likelihood
    calculation methods and can handle masked positions.

    Attributes:
        _available (MessageBool): Indicates whether the MSA Transformer model is available.
    """

    _available = AVAILABLE

    def __init__(self, metadata_folder: str,
                 marginal_method: MarginalMethod = MarginalMethod.WILDTYPE,
                 positions: Optional[List[int]] = None,
                 flatten: bool = False,
                 pool: bool = True,
                 batch_size: int = 32,
                 device: str = 'cpu',
                 wt: Optional[Union[str, ProteinSequence]] = None):
        """
        Initialize the MSATransformerLikelihoodWrapper.

        Args:
            metadata_folder (str): Folder to store metadata.
            marginal_method (MarginalMethod): Method to compute marginal likelihoods.
            positions (Optional[List[int]]): Specific positions to consider.
            flatten (bool): Whether to flatten the output.
            pool (bool): Whether to pool the likelihoods across positions.
            batch_size (int): Number of sequences to process in each batch.
            device (str): Device to use for computations ('cpu' or 'cuda').
            wt (Optional[Union[str, ProteinSequence]]): Wild type sequence.
        """
        super().__init__(metadata_folder=metadata_folder,
                         marginal_method=marginal_method,
                         positions=positions,
                         flatten=flatten,
                         pool=pool,
                         batch_size=batch_size,
                         device=device,
                         wt=wt)

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'MSATransformerLikelihoodWrapper':
        """
        Fit the MSA Transformer model to the protein sequences.

        This method loads the pre-trained model and stores the MSA used for fitting.

        Args:
            X (ProteinSequences): The input protein sequences (MSA).
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            MSATransformerLikelihoodWrapper: The fitted wrapper.
        """
        self.model_, self.alphabet_ = pretrained.esm_msa1b_t12_100M_UR50S()
        self.model_ = self.model_.to(self.device)
        self.msa_length_: int = X.width
        self.original_msa_: ProteinSequences = X
        self.batch_converter_ = self.alphabet_.get_batch_converter()
        return self

    def _prepare_msa_batch(self, msa: ProteinSequences, query_sequence: ProteinSequence, mask_positions: List[int] = []) -> torch.Tensor:
        """
        Prepare a batch of the MSA including the query sequence.

        Args:
            msa (ProteinSequences): The MSA to be prepared.
            query_sequence (ProteinSequence): The query sequence to be added to the MSA.
            mask_positions (List[int]): Positions to mask in the query sequence.

        Returns:
            torch.Tensor: The prepared MSA batch tokens.
        """
        batch_converter = self.alphabet_.get_batch_converter()
        data = [(str(hash(s)), str(s)) for s in msa] + [(str(hash(query_sequence)), str(query_sequence))]
        _, _, batch_tokens = batch_converter(data)
        for i in mask_positions:
            batch_tokens[0, -1, i+1] = self.alphabet_.mask_idx
        return batch_tokens

    def _compute_log_likelihoods(self, X: ProteinSequences, mask_positions: Optional[List[List[int]]] = None) -> List[np.ndarray]:
        """
        Compute log likelihoods for the input sequences using the MSA Transformer model.

        Args:
            X (ProteinSequences): The input protein sequences.
            mask_positions (Optional[List[List[int]]]): Positions to mask for each sequence.

        Returns:
            List[np.ndarray]: The log likelihoods for each sequence.
        """
        all_log_likelihoods = []

        for i, sequence in enumerate(X):
            sequence_logits = []
            batch_sizes = []
            bar = tqdm.tqdm(total=len(self.original_msa_) // (self.batch_size - 1), desc="MSA batches")
            for msa_sequences in self.original_msa_.iter_batches(self.batch_size - 1):
                mask_pos = mask_positions[i] if mask_positions else []
                batch_tokens = self._prepare_msa_batch(msa_sequences, sequence, mask_positions=mask_pos)
                batch_tokens = batch_tokens.to(self.device)

                with torch.no_grad():
                    results = self.model_(batch_tokens, repr_layers=[], return_contacts=False)
                
                logits = results["logits"]
                query_logits = logits[0, -1, 1:, :]  # Remove start token
                
                sequence_logits.append(query_logits.cpu().numpy())
                batch_sizes.append(batch_tokens.shape[1] - 1)
                bar.update(1)

            # Compute weighted average of logits
            avg_logits = np.average(sequence_logits, axis=0, weights=batch_sizes)
            
            # Convert back to tensor and apply log_softmax
            avg_logits_tensor = torch.tensor(avg_logits)
            log_probs = torch.log_softmax(avg_logits_tensor, dim=-1)
            
            all_log_likelihoods.append(log_probs.cpu().numpy())

        return all_log_likelihoods

    def _index_log_probs(self, log_probs: np.ndarray, sequences: ProteinSequences) -> np.ndarray:
        """
        Index log probabilities by the observed amino acids in the sequences.

        Args:
            log_probs (np.ndarray): Log probabilities for each sequence and position.
            sequences (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: Indexed log probabilities.

        Raises:
            AssertionError: If log_probs and sequences have incompatible lengths.
        """
        _, _, tokens = self.batch_converter_([(str(hash(s)), str(s)) for s in sequences])
        tokens = tokens.squeeze(0)[:, 1:]  # Remove start token
        
        batch_size, seq_len = tokens.shape
        assert seq_len == log_probs.shape[0], "Log probs and sequences must have the same length."
        rows = np.arange(log_probs.shape[0])
        rows_expanded = np.expand_dims(rows, axis=0)
        log_likelihoods = log_probs[rows_expanded, tokens]
        return log_likelihoods
