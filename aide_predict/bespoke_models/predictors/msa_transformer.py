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
from esm import pretrained

from aide_predict.bespoke_models.base import ProteinModelWrapper, PositionSpecificMixin, RequiresMSAMixin, RequiresFixedLengthMixin, CanRegressMixin, RequiresWTDuringInferenceMixin
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool

try:
    import esm
    import torch
    from esm.pretrained import esm_msa1b_t12_100M_UR50S
    AVAILABLE = MessageBool(True, "MSA Transformer is available.")
except ImportError:
    AVAILABLE = MessageBool(False, "MSA Transformer requires fair-esm, which is not installed.")

class MSATransformerLikelihoodWrapper(PositionSpecificMixin, RequiresMSAMixin, RequiresFixedLengthMixin, CanRegressMixin, RequiresWTDuringInferenceMixin, ProteinModelWrapper):
    """
    A wrapper for the MSA Transformer model to compute log likelihoods for protein sequences.
    
    This class uses the MSA Transformer model to calculate log likelihoods for protein sequences.
    It supports mutant, wildtype, and masked marginal likelihood calculations for fixed-length sequences.

    Attributes:
        marginal_method (str): The method to use for calculating marginal likelihoods.
        positions (Optional[List[int]]): Specific positions to compute likelihoods for. If None, all positions are used.
        pool (bool): Whether to pool the likelihoods across positions.
        flatten (bool): Whether to flatten the output array.
        batch_size (int): The batch size for processing sequences.
        device (str): The device to use for computations ('cuda' or 'cpu').
    """

    _available = AVAILABLE

    def __init__(self, metadata_folder: str, 
                 marginal_method: str = 'wildtype_marginal',
                 positions: Optional[List[int]] = None, 
                 flatten: bool = False,
                 pool: bool = True,
                 batch_size: int = 32,
                 device: str = 'cpu',
                 wt: Optional[Union[str, ProteinSequence]] = None):
        """
        Initialize the MSATransformerLikelihoodWrapper.

        Args:
            metadata_folder (str): The folder where metadata is stored.
            marginal_method (str): The method to use for calculating marginal likelihoods.
            positions (Optional[List[int]]): Specific positions to compute likelihoods for. If None, all positions are used.
            flatten (bool): Whether to flatten the output array.
            pool (bool): Whether to pool the likelihoods across positions.
            batch_size (int): The batch size that will be given as input to the model. Ideally this is the size of the MSA.
            device (str): The device to use for computations ('cuda' or 'cpu').
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence, if any.
        """
        super().__init__(metadata_folder=metadata_folder, wt=wt, positions=positions, pool=pool, flatten=flatten)
        self.marginal_method = marginal_method
        self.batch_size = batch_size
        self.device = device

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'MSATransformerLikelihoodWrapper':
        """
        Fit the MSA Transformer model to the protein sequences.

        This method loads the pre-trained model and stores the MSA used for fitting.

        Args:
            X (ProteinSequences): The input protein sequences (MSA).
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            MSATransformerLikelihoodWrapper: The fitted wrapper.

        Raises:
            ValueError: If the input sequences are not aligned or of fixed length.
        """
        self.model_, self.alphabet_ = pretrained.esm_msa1b_t12_100M_UR50S()
        self.model_ = self.model_.to(self.device)
        self.msa_length_ = X.width
        self.original_msa_ = X
        self.batch_converter_ = self.alphabet_.get_batch_converter()
        return self

    def _prepare_msa_batch(self, msa: ProteinSequences, query_sequence: ProteinSequence, mask_positions: List[int]=[]):
        """
        Prepare a batch of the MSA including the query sequence.

        Args:
            msa (ProteinSequences): The MSA to be prepared.
            query_sequence (ProteinSequence): The query sequence to be added to the MSA.

        Returns:
            protein_sequences: The prepared MSA batch.
        """
        batch_converter = self.alphabet_.get_batch_converter()
        data = [(str(hash(s)), str(s)) for s in msa] + [(str(hash(query_sequence)), str(query_sequence))]
        _, _, batch_tokens = batch_converter(data)
        for i in mask_positions:
            batch_tokens[0, -1, i+1] = self.alphabet_.mask_idx
        return batch_tokens


    def _compute_log_likelihoods(self, X: List[str], mask_positions=[]) -> np.ndarray:
        """
        Compute log likelihoods for the input sequences using the MSA Transformer model.

        Args:
            X (List[str]): The input protein sequences.

        Returns:
            np.ndarray: The log likelihoods for each sequence.
        """
        batch_converter = self.batch_converter_

        all_log_likelihoods = []

        for sequence in tqdm.tqdm(X, desc="Sequences"):
            sequence_log_likelihoods = []
            n_batches = len(self.original_msa_) // (self.batch_size-1)
            batch_sizes = []
            with tqdm.tqdm(total=n_batches, desc="MSA batches") as pbar:
                for msa_sequences in self.original_msa_.iter_batches(self.batch_size - 1):
                    batch_tokens = self._prepare_msa_batch(msa_sequences, sequence, mask_positions=mask_positions)

                    batch_tokens = batch_tokens.to(self.device)

                    with torch.no_grad():
                        results = self.model_(batch_tokens, repr_layers=[], return_contacts=False)
                    
                    logits = results["logits"]
                    
                    # Extract logits for the query sequence (last in the batch)
                    query_logits = logits[0, -1, 1:, :]  # Remove start token
                    
                    # Compute log probabilities
                    log_probs = torch.log_softmax(query_logits, dim=-1)
                                    
                    sequence_log_likelihoods.append(log_probs.cpu().numpy())
                    pbar.update(1)
                    batch_sizes.append(batch_tokens.shape[1] - 1)

            # Average log likelihoods across all batches
            avg_log_likelihood = np.average(sequence_log_likelihoods, axis=0, weights=batch_sizes)
            # of shape (seq_length, n_tokens)
            all_log_likelihoods.append(np.expand_dims(avg_log_likelihood, axis=0))

        # we can stack because all sequences are the same length
        return np.vstack(all_log_likelihoods)
    
    def _index_log_probs(self, log_probs: np.ndarray, sequences: ProteinSequences) -> np.ndarray:
        """
        Index log probabilities by the observed amino acids in the sequences.

        Args:
            log_probs (np.ndarray): Log probabilities for each sequence and position.
            sequences (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: Indexed log probabilities.
        """
        # Tokenize the sequences
        _, _, tokens = self.batch_converter_([(str(hash(s)), str(s)) for s in sequences])
        tokens = tokens.squeeze(0)[:, 1:]  # Remove start token
        
        batch_size, seq_len = tokens.shape
        assert seq_len == log_probs.shape[0], "Log probs and sequences must have the same length."
        rows = np.arange(log_probs.shape[0])
        rows_expanded = np.expand_dims(rows, axis=0)
        log_likelihoods = log_probs[rows_expanded, tokens]
        return log_likelihoods

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the protein sequences into log likelihoods.

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: The log likelihoods for the sequences.

        Raises:
            ValueError: If the input sequences are not of the same length as the original MSA.
        """
        if X.width != self.msa_length_:
            raise ValueError(f"Input sequences must have the same length as the original MSA ({self.msa_length_}).")

        if self.marginal_method == 'mutant_marginal':
            log_probs = self._compute_log_likelihoods(X)
            # this should be of shape (n_sequences, seq_length, n_tokens)
            log_likelihoods = np.vstack([self._index_log_probs(l, [X[i]]) for i, l in enumerate(log_probs)])
            # this should be of shape (n_sequences, seq_length)
            if self.wt is not None:
                # index the log probs by the wt tokens.
                wt_likelihoods = np.vstack([self._index_log_probs(l, ProteinSequences([self.wt])) for l in log_probs])
                log_likelihoods = log_likelihoods - wt_likelihoods
        elif self.marginal_method == 'wildtype_marginal':
            if self.wt is None:
                raise ValueError("Wildtype sequence must be provided for wildtype marginal likelihoods.")
            
            wt_log_probs = self._compute_log_likelihoods([self.wt])[0]
            log_likelihoods = self._index_log_probs(wt_log_probs, X)
            wt_likelihoods = self._index_log_probs(wt_log_probs, ProteinSequences([self.wt]))
            log_likelihoods = log_likelihoods - wt_likelihoods
            
        elif self.marginal_method == 'masked_marginal':
            if self.wt is None:
                raise ValueError("Wildtype sequence must be provided for masked marginal likelihoods.")
            mask_positions = self.positions if self.positions is not None else X.mutated_positions

            output_probs = np.zeros((len(X), len(mask_positions)))

            for i, pos in enumerate(mask_positions):
                masked_log_probs = self._compute_log_likelihoods([self.wt], mask_positions=[pos])[0]
                mutant_likelihoods = self._index_log_probs(masked_log_probs, X)
                wt_likelihoods = self._index_log_probs(masked_log_probs, ProteinSequences([self.wt]))
                output_probs[:, i] = (mutant_likelihoods - wt_likelihoods)[:, pos]

            # expand back out to full sequence
            log_likelihoods = np.zeros((len(X), X.width))
            log_likelihoods[:, mask_positions] = output_probs
        else:
            raise ValueError(f"Unknown marginal method: {self.marginal_method}")

        if self.positions is not None:
            log_likelihoods = log_likelihoods[:, self.positions]
            if self.pool:
                log_likelihoods = np.mean(log_likelihoods, axis=1)

        elif self.pool:
            # here user did not specify positions, so we only want to pool over changed positions
            masks = [x.mutated_positions(self.wt) for x in X]
            changed_log_likelihoods = [log_likelihoods[i, mask] for i, mask in enumerate(masks)]
            log_likelihoods = np.array([np.mean(x) for x in changed_log_likelihoods]).reshape(-1,1)

        return log_likelihoods

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (Optional[List[str]]): Ignored. Present for API consistency.

        Returns:
            List[str]: Output feature names.
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model has not been fitted yet. Call fit() before using this method.")
        
        positions = self.positions if self.positions is not None else range(self.msa_length_)
        
        if self.pool:
            return ["MSA_log_likelihood"]
        elif self.flatten:
            return [f"pos{p}_log_likelihood" for p in positions]
        else:
            raise ValueError("Cannot return feature names for non-flattened, non-pooled output.")