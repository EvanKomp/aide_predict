# aide_predict/bespoke_models/embedders/esm2.py
'''
* Author: Evan Komp
* Created: 7/5/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

ESM2 language model self supervised embeddings.
'''

import warnings
from typing import List, Union, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from aide_predict.bespoke_models.base import ProteinModelWrapper, PositionSpecificMixin, CanHandleAlignedSequencesMixin
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence

try:
    import transformers
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

class ESM2Embedding(PositionSpecificMixin, CanHandleAlignedSequencesMixin, ProteinModelWrapper):
    """
    A protein sequence embedder that uses the ESM2 model to generate embeddings.
    
    This class wraps the ESM2 model to provide embeddings for protein sequences.
    It can handle both aligned and unaligned sequences and allows for retrieving
    embeddings from a specific layer of the model.

    Attributes:
        model_checkpoint (str): The name of the ESM2 model checkpoint to use.
        layer (int): The layer from which to extract embeddings (-1 for last layer).
        positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
        pool (bool): Whether to pool the encoded vectors across positions.
        flatten (bool): Whether to flatten the output array.
        batch_size (int): The batch size for processing sequences.
        device (str): The device to use for computations ('cuda' or 'cpu').
    """

    _available = AVAILABLE

    def __init__(self, metadata_folder: str, 
                 model_checkpoint: str = 'facebook/esm2_t6_8M_UR50D',
                 layer: int = -1,
                 positions: Optional[List[int]] = None, 
                 flatten: bool = False,
                 pool: bool = False,
                 batch_size: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 wt: Optional[Union[str, ProteinSequence]] = None):
        """
        Initialize the ESM2Embedding.

        Args:
            metadata_folder (str): The folder where metadata is stored.
            model_checkpoint (str): The name of the ESM2 model checkpoint to use.
            layer (int): The layer from which to extract embeddings (-1 for last layer).
            positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
            flatten (bool): Whether to flatten the output array.
            batch_size (int): The batch size for processing sequences.
            device (str): The device to use for computations ('cuda' or 'cpu').
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence, if any.

        Notes: WT is set to None to avoid normalization. For an embedder this is effectively a feature scaler which you
        should do manually if you want
        """
        super().__init__(metadata_folder=metadata_folder, wt=None, positions=positions, pool=pool, flatten=flatten)
        self.model_checkpoint = model_checkpoint
        self.layer = layer
        self.batch_size = batch_size
        self.device = device

    def _prepare_sequences(self, X: ProteinSequences) -> ProteinSequences:
        """
        Prepare the protein sequences for ESM tokenization

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            List[str]: The tokenized protein sequences.
        """
        return [' '.join(seq) for seq in X]

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'ESM2Embedding':
        """
        Fit the ESM2 embedder to the protein sequences.

        This method doesn't actually fit anything but loads the pre-trained model.

        Args:
            X (ProteinSequences): The input protein sequences.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            ESM2Embedding: The fitted embedder.
        """
        self.model_ = EsmModel.from_pretrained(self.model_checkpoint).to(self.device)
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_checkpoint)
        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the protein sequences into ESM2 embeddings.

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: The ESM2 embeddings for the sequences.
        """
        all_embeddings = []
        if not X.fixed_length and self.positions is not None and not X.aligned:
            raise ValueError("Cannot specify positions for variable length sequences unless aligned, where positions are interpreted as aligned positions")
        if not X.fixed_length and not self.pool and not X.aligned:
            raise ValueError("Cannot return position-specific embeddings for variable length sequences.")
        
        mapping = None
        if X.has_gaps:
            # here we need to store a mapping such that if positions were specified we can map back to
            # the aligned positions
            mapping = X.get_alignment_mapping()
            X = X.with_no_gaps()
            # raise if positions were not passed - here behavior is uncertain
            if self.positions is None and not self.pool:
                raise ValueError("Cannot return position-specific embeddings for sequences with gaps unless positions are specified or pooling is on.")

        base_index = 0
        for batch in X.iter_batches(self.batch_size):
            batch_sequences = self._prepare_sequences(batch)
            inputs = self.tokenizer_(batch_sequences, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model_(**inputs, output_hidden_states=True)
            
            # Get embeddings from the specified layer
            embeddings = outputs.hidden_states[self.layer]
            # these include masked out tokens
            mask = inputs['attention_mask']
            embeddings = [embeddings[i, mask[i].bool(), :].cpu().numpy() for i in range(embeddings.shape[0])]
            # these should each be of shape (seq_len, hidden_size)
            
            # Remove special tokens (assuming first and last tokens are special)
            embeddings = [emb[1:-1] for emb in embeddings]
            
            if self.positions is not None and mapping is None:
                # here we have fixed length so we can just use positions
                embeddings = [emb[self.positions] for emb in embeddings]
            elif self.positions is not None and mapping is not None:
                # here we have variable length sequences that were input as an aligned set,
                # the user asked for positions in the alignment
                aligned_embeddings = []
                for i, emb in enumerate(embeddings):
                    seq_mapping = mapping[base_index + i]
                    aligned_emb = np.zeros((len(self.positions), emb.shape[1]))
                    for j, pos in enumerate(self.positions):
                        if pos in seq_mapping:
                            aligned_pos = seq_mapping.index(pos)
                            aligned_emb[j] = emb[aligned_pos]
                        # If pos is not in seq_mapping, it remains a zero vector
                    aligned_embeddings.append(aligned_emb)
                embeddings = aligned_embeddings
            else:
                # Here positions were not specified and either have fixed length or pooling
                # is on
                pass
            
            if self.pool:
                embeddings = [emb.mean(axis=0) for emb in embeddings]
            
            # add 0th dimension
            embeddings = [np.expand_dims(emb, 0) for emb in embeddings]
            all_embeddings.extend(embeddings)

            base_index += len(batch)
        
        # stack along 0 dimension
        return np.vstack(all_embeddings)

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
        
        embedding_dim = self.model_.config.hidden_size
        positions = self.positions if self.positions is not None else range(self.model_.config.max_position_embeddings - 2)  # -2 for special tokens
        
        if self.pool:
            return [f"ESM2_emb{i}" for i in range(embedding_dim)]
        elif self.flatten:
            return [f"pos{p}_emb{i}" for p in positions for i in range(embedding_dim)]
        else:
            raise ValueError("Cannot return feature names for non-flattened non-pooled embeddings.")