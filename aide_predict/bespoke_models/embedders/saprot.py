# aide_predict/bespoke_models/embedders/saprot.py
'''
* Author: Evan Komp
* Created: 7/16/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import warnings
from typing import List, Union, Optional

import numpy as np
import torch
from tqdm import tqdm

from aide_predict.bespoke_models.base import ProteinModelWrapper, PositionSpecificMixin, CacheMixin
from aide_predict.bespoke_models import model_device_context
from aide_predict.bespoke_models.predictors.saprot import get_structure_tokens
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool

try:
    from transformers import EsmTokenizer, EsmModel
    AVAILABLE = MessageBool(True, "SaProt model is available.")
except ImportError:
    AVAILABLE = MessageBool(False, "SaProt model is not available. Please install the transformers library.")

from aide_predict.bespoke_models.predictors.saprot import get_structure_tokens

import logging
logger = logging.getLogger(__name__)

class SaProtEmbedding(CacheMixin, PositionSpecificMixin, ProteinModelWrapper):
    """
    A protein sequence embedder that uses the SaProt model to generate embeddings.
    
    This class wraps the SaProt model to provide embeddings for protein sequences.
    It can handle both aligned and unaligned sequences and allows for retrieving
    embeddings from a specific layer of the model.

    Attributes:
        model_checkpoint (str): The name of the SaProt model checkpoint to use.
        layer (int): The layer from which to extract embeddings (-1 for last layer).
        positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
        pool (bool): Whether to pool the encoded vectors across positions.
        flatten (bool): Whether to flatten the output array.
        batch_size (int): The batch size for processing sequences.
        device (str): The device to use for computations ('cuda' or 'cpu').
        foldseek_path (str): Path to the FoldSeek executable.
    """

    _available = AVAILABLE

    def __init__(self, metadata_folder: str=None, 
                 model_checkpoint: str = 'westlake-repl/SaProt_650M_AF2',
                 layer: int = -1,
                 positions: Optional[List[int]] = None, 
                 flatten: bool = False,
                 pool: bool = False,
                 batch_size: int = 32,
                 device: str = 'cpu',
                 foldseek_path: str = 'foldseek',
                 wt: Optional[Union[str, ProteinSequence]] = None,
                 **kwargs):
        """
        Initialize the SaProtEmbedding.

        Args:
            metadata_folder (str): The folder where metadata is stored.
            model_checkpoint (str): The name of the SaProt model checkpoint to use.
            layer (int): The layer from which to extract embeddings (-1 for last layer).
            positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
            flatten (bool): Whether to flatten the output array.
            pool (bool): Whether to pool the encoded vectors across positions.
            batch_size (int): The batch size for processing sequences.
            device (str): The device to use for computations ('cuda' or 'cpu').
            foldseek_path (str): Path to the FoldSeek executable.
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence, if any.
        """
        super().__init__(metadata_folder=metadata_folder, wt=wt, positions=positions, pool=pool, flatten=flatten, **kwargs)
        self.model_checkpoint = model_checkpoint
        self.layer = layer
        self.batch_size = batch_size
        self.device = device
        self.foldseek_path = foldseek_path

    def _prepare_input_sequence(self, sequence: ProteinSequence) -> str:
        """Convert a protein sequence into a string ready for tokenization."""
        if sequence.structure is None:
            if self.wt is not None and self.wt.structure:
                struc_tokens = get_structure_tokens(self.wt.structure, self.foldseek_path)
                return ''.join([a + b.lower() for a, b in zip(str(sequence).upper(), struc_tokens)])
            else:
                return ''.join([aa + '#' for aa in str(sequence).upper()])
        else:
            struc_tokens = get_structure_tokens(sequence.structure, self.foldseek_path)
            return ''.join([a + b.lower() for a, b in zip(str(sequence).upper(), struc_tokens)])

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'SaProtEmbedding':
        """
        Fit the SaProt embedder to the protein sequences.

        This method doesn't actually fit anything but loads the pre-trained model.

        Args:
            X (ProteinSequences): The input protein sequences.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            SaProtEmbedding: The fitted embedder.
        """
        self.fitted_ = True
        return self
    
    def _load_model(self) -> None:
        """Load and the model and other objects into memory on device such that they can be accessed in
        `_compute_log_likelihoods` and `_index_log_probs`.

        Required abstract class from `LikelihoodTransformerBase`.
        """
        self.model_ = EsmModel.from_pretrained(self.model_checkpoint).to(self.device)
        self.tokenizer_ = EsmTokenizer.from_pretrained(self.model_checkpoint)
    
    def _cleanup_model(self) -> None:
        """
        Clean up the model and other objects loaded into memory in `_load_model`.

        Required abstract class from `LikelihoodTransformerBase`.
        """
        del self.model_
        del self.tokenizer_

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the protein sequences into SaProt embeddings.

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: The SaProt embeddings for the sequences.
        """
        with model_device_context(self, self._load_model, self._cleanup_model, self.device):
            all_embeddings = []
            
            bar = tqdm(total=len(X), desc="Computing SaProt embeddings", unit="sequence")
            for batch in X.iter_batches(self.batch_size):
                batch_sequences = [self._prepare_input_sequence(seq) for seq in batch]
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
                
                if self.positions is not None:
                    embeddings = [emb[self.positions] for emb in embeddings]
                
                if self.pool:
                    embeddings = [emb.mean(axis=0) for emb in embeddings]
                
                # add 0th dimension
                embeddings = [np.expand_dims(emb, 0) for emb in embeddings]
                all_embeddings.extend(embeddings)

                bar.update(len(batch))
            
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
            return [f"SaProt_emb{i}" for i in range(embedding_dim)]
        elif self.flatten:
            return [f"pos{p}_emb{i}" for p in positions for i in range(embedding_dim)]
        else:
            return [f"pos{p}" for p in positions]