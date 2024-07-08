# aide_predict/bespoke_models/embedders/msa_transformer.py
'''
* Author: Evan Komp
* Created: 7/8/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import warnings
from typing import List, Union, Optional

import numpy as np
import torch
from esm import pretrained

from aide_predict.bespoke_models.base import ProteinModelWrapper, PositionSpecificMixin, RequiresMSAMixin, RequiresFixedLengthMixin
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool

try:
    import esm
    from esm.pretrained import esm_msa1b_t12_100M_UR50S
    AVAILABLE = MessageBool(True, "MSA Transformer is available.")
except ImportError:
    AVAILABLE = MessageBool(False, "MSA Transformer requires fair-esm, which is not installed.")

class MSATransformerEmbedding(PositionSpecificMixin, RequiresMSAMixin, RequiresFixedLengthMixin, ProteinModelWrapper):
    """
    A protein sequence embedder that uses the MSA Transformer model to generate embeddings.
    
    This class wraps the MSA Transformer model to provide embeddings for protein sequences.
    It requires fixed-length sequences and an MSA for fitting. At prediction time,
    it can handle sequences of the same length as the MSA used for fitting.

    Attributes:
        layer (int): The layer from which to extract embeddings (-1 for last layer).
        positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
        pool (bool): Whether to pool the encoded vectors across positions.
        flatten (bool): Whether to flatten the output array.
        batch_size (int): The batch size for processing sequences.
        device (str): The device to use for computations ('cuda' or 'cpu').
    """

    _available = AVAILABLE

    def __init__(self, metadata_folder: str, 
                 layer: int = -1,
                 positions: Optional[List[int]] = None, 
                 flatten: bool = False,
                 pool: bool = False,
                 batch_size: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 wt: Optional[Union[str, ProteinSequence]] = None):
        """
        Initialize the MSATransformerEmbedding.

        Args:
            metadata_folder (str): The folder where metadata is stored.
            layer (int): The layer from which to extract embeddings (-1 for last layer).
            positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
            flatten (bool): Whether to flatten the output array.
            pool (bool): Whether to pool the encoded vectors across positions.
            batch_size (int): The batch size that will be given as input to the model. Ideally this is the size of the MSA.
            device (str): The device to use for computations ('cuda' or 'cpu').
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence, if any.
        """
        super().__init__(metadata_folder=metadata_folder, wt=None, positions=positions, pool=pool, flatten=flatten)
        self.layer = layer
        self.batch_size = batch_size
        self.device = device


    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'MSATransformerEmbedding':
        """
        Fit the MSA Transformer embedder to the protein sequences.

        This method loads the pre-trained model and stores the MSA used for fitting.

        Args:
            X (ProteinSequences): The input protein sequences (MSA).
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            MSATransformerEmbedding: The fitted embedder.

        Raises:
            ValueError: If the input sequences are not aligned or of fixed length.
        """

        self.model_, self.alphabet_ = pretrained.esm_msa1b_t12_100M_UR50S()
        self.model_ = self.model_.to(self.device)
        self.msa_length_ = X.width
        self.original_msa_ = X
        return self
    
    def _prepare_msa_batch(self, msa: ProteinSequences, query_sequence: ProteinSequence) -> ProteinSequences:
        """
        Prepare a batch of the MSA including the query sequence.

        Args:
            msa (ProteinSequences): The MSA to be prepared.
            query_sequence (ProteinSequence): The query sequence to be added to the MSA.

        Returns:
            protein_sequences: The prepared MSA batch.
        """
        msa.append(query_sequence)
        return msa

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the protein sequences into MSA Transformer embeddings.

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: The MSA Transformer embeddings for the sequences.

        Raises:
            ValueError: If the input sequences are not of the same length as the original MSA.
        """
        if X.width != self.msa_length_:
            raise ValueError(f"Input sequences must have the same length as the original MSA ({self.msa_length_}).")
        
        batch_converter = self.alphabet_.get_batch_converter()

        all_embeddings = []

        for sequence in X:
            sequence_embeddings = []
            for msa_sequences in self.original_msa_.iter_batches(self.batch_size - 1):
                sequences_batch = self._prepare_msa_batch(msa_sequences, sequence)

                batch_tokens = batch_converter([(str(hash(s)), s) for s in sequences_batch])[2]
                batch_tokens = batch_tokens.to(self.device)

                if self.layer == -1:
                    self.layer = self.model_.num_layers - 1

                with torch.no_grad():
                    results = self.model_(batch_tokens, repr_layers=[self.layer], return_contacts=False)
                
                embeddings = results["representations"][self.layer]
                
                # Extract embedding for the query sequence (last in the batch)
                # remove the start token
                query_embedding = embeddings[0, -1, 1:, :].cpu().numpy()
                sequence_embeddings.append(query_embedding)

            # Average embeddings across all batches
            avg_embedding = np.mean(sequence_embeddings, axis=0)

            if self.positions is not None:
                avg_embedding = avg_embedding[self.positions]
            
            if self.pool:
                avg_embedding = avg_embedding.mean(axis=0)
            
            all_embeddings.append(np.expand_dims(avg_embedding, axis=0))

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
        
        embedding_dim = self.model_.args.embed_dim
        positions = self.positions if self.positions is not None else range(self.msa_length_)
        
        if self.pool:
            return [f"MSA_emb{i}" for i in range(embedding_dim)]
        elif self.flatten:
            return [f"pos{p}_emb{i}" for p in positions for i in range(embedding_dim)]
        else:
            return [f"pos{p}" for p in positions]