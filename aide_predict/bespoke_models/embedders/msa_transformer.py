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
import tqdm


from aide_predict.bespoke_models import model_device_context
from aide_predict.bespoke_models.base import ProteinModelWrapper, PositionSpecificMixin, RequiresMSAPerSequenceMixin, CacheMixin, CanHandleAlignedSequencesMixin
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool

try:
    import esm
    from esm.pretrained import esm_msa1b_t12_100M_UR50S
    from esm import pretrained
    import torch
    AVAILABLE = MessageBool(True, "MSA Transformer is available.")
except ImportError:
    AVAILABLE = MessageBool(False, "MSA Transformer requires fair-esm, which is not installed.")

import logging
logger = logging.getLogger(__name__)

class MSATransformerEmbedding(CacheMixin, PositionSpecificMixin, CanHandleAlignedSequencesMixin, RequiresMSAPerSequenceMixin, ProteinModelWrapper):
    """
    A protein sequence embedder that uses the MSA Transformer model to generate embeddings.
    
    This class wraps the MSA Transformer model to provide embeddings for protein sequences.
    It requires that each sequence has its own MSA. It can handle both aligned and unaligned 
    sequences and allows for retrieving embeddings from a specific layer of the model.

    Attributes:
        layer (int): The layer from which to extract embeddings (-1 for last layer).
        positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
        pool (bool): Whether to pool the encoded vectors across positions.
        flatten (bool): Whether to flatten the output array.
        batch_size (int): The batch size for processing sequences.
        n_msa_seqs (int): The number of sequences to sample from each MSA.
        device (str): The device to use for computations ('cuda' or 'cpu').
    """

    _available = AVAILABLE

    def __init__(self, metadata_folder: str=None, 
                 layer: int = -1,
                 positions: Optional[List[int]] = None, 
                 flatten: bool = False,
                 pool: bool = False,
                 batch_size: int = 32,
                 n_msa_seqs: int = 360,
                 device: str = 'cpu',
                 use_cache: bool = True,
                 wt: Optional[Union[str, ProteinSequence]] = None):
        """
        Initialize the MSATransformerEmbedding.

        Args:
            metadata_folder (str): The folder where metadata is stored.
            layer (int): The layer from which to extract embeddings (-1 for last layer).
            positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
            flatten (bool): Whether to flatten the output array.
            pool (bool): Whether to pool the encoded vectors across positions.
            batch_size (int): The batch size for processing MSA batches.
            n_msa_seqs (int): The number of sequences to use from the MSA, sampled from the weight vector.
            device (str): The device to use for computations ('cuda' or 'cpu').
            use_cache (bool): Whether to cache results to avoid redundant computations.
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence, if any.
        """
        super().__init__(metadata_folder=metadata_folder, wt=wt, positions=positions, pool=pool, flatten=flatten, use_cache=use_cache)
        self.layer = layer
        self.batch_size = batch_size
        self.device = device
        self.n_msa_seqs = n_msa_seqs
        self._msa_cache = {}

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'MSATransformerEmbedding':
        """
        Fit the MSA Transformer embedder.

        This method initializes the model and prepares for embedding generation.
        No actual training occurs as the model is pre-trained.

        Args:
            X (ProteinSequences): The input protein sequences, each with its own MSA.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            MSATransformerEmbedding: The fitted embedder.
        """
        self._msa_cache = {}
        with model_device_context(self, self._load_model, self._cleanup_model, self.device):
            self.embedding_dim_ = self.model_.args.embed_dim
        self.fitted_ = True
        return self
    
    def _load_model(self) -> None:
        """
        Load the model and related components into memory.

        This method loads the MSA Transformer model, alphabet, and batch converter.
        """
        self.model_, self.alphabet_ = pretrained.esm_msa1b_t12_100M_UR50S()
        self.model_ = self.model_.to(self.device)
        self.batch_converter_ = self.alphabet_.get_batch_converter()
    
    def _cleanup_model(self) -> None:
        """
        Clean up the model and other objects loaded into memory.

        This method frees memory by deleting model-related objects.
        """
        del self.model_
        del self.alphabet_
        del self.batch_converter_

    def _get_sampled_msa(self, msa: ProteinSequences) -> ProteinSequences:
        """
        Get a sampled MSA from the cache or create a new one.

        Args:
            msa (ProteinSequences): The original MSA.

        Returns:
            ProteinSequences: A sampled subset of the MSA.
        """
        msa_hash = hash(msa)
        if msa_hash not in self._msa_cache:
            # Sample with a consistent seed based on the hash to ensure reproducibility
            seed = abs(msa_hash) % (2**32)  # Ensure we have a positive seed within uint32 range
            np.random.seed(seed)
            sampled_msa = msa.sample(min(self.n_msa_seqs, len(msa)), replace=False, keep_first=True)
            self._msa_cache[msa_hash] = sampled_msa
        return self._msa_cache[msa_hash]

    def _prepare_msa_batch(self, msa: ProteinSequences, query_sequence: ProteinSequence) -> List[tuple]:
        """
        Prepare a batch of the MSA including the query sequence for the MSA Transformer.

        Args:
            msa (ProteinSequences): The MSA batch to be prepared.
            query_sequence (ProteinSequence): The query sequence to include.

        Returns:
            List[tuple]: Batch data in the format expected by the MSA Transformer.
        """
        # Ensure the query sequence is at the last position
        batch_data = [(str(hash(s)), str(s).upper().replace('.', '-')) for s in msa]
        batch_data.append((str(hash(query_sequence)), str(query_sequence).upper().replace('.', '-')))
        return batch_data

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the protein sequences into MSA Transformer embeddings.

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: The MSA Transformer embeddings for the sequences.

        Raises:
            ValueError: If any sequence doesn't have an associated MSA.
        """
        with model_device_context(self, self._load_model, self._cleanup_model, self.device):
            all_embeddings = []
            
            for sequence in tqdm.tqdm(X, desc="Generating embeddings", unit="sequence"):
                # Validate that the sequence has an MSA
                if not sequence.has_msa:
                    raise ValueError(f"Sequence {sequence.id} does not have an associated MSA.")
                
                # Validate that the MSA width matches the sequence length
                if not sequence.msa_same_width:
                    raise ValueError(f"Sequence {sequence.id} has an MSA with width {sequence.msa.width} which doesn't match sequence length {len(sequence)}.")
                
                # Check if MSA sequence is too long
                if len(sequence) > 1024:
                    logger.warning(f"Sequence {sequence.id} length {len(sequence)} exceeds the MSA Transformer model's limit of 1024. Truncating.")
                    sequence = ProteinSequence(str(sequence)[:1024], id=sequence.id)
                    sequence.msa = ProteinSequences([ProteinSequence(str(s)[:1024], id=s.id) for s in sequence.msa])
                
                # Get the sampled MSA for this sequence
                sampled_msa = self._get_sampled_msa(sequence.msa)
                
                sequence_embeddings = []
                batch_sizes = []
                
                for msa_batch in sampled_msa.iter_batches(self.batch_size - 1):
                    batch_data = self._prepare_msa_batch(msa_batch, sequence)
                    _, _, batch_tokens = self.batch_converter_(batch_data)
                    batch_tokens = batch_tokens.to(self.device)

                    # Determine layer to use
                    layer_idx = self.layer if self.layer >= 0 else self.model_.num_layers - 1
                    
                    with torch.no_grad():
                        results = self.model_(batch_tokens, repr_layers=[layer_idx], return_contacts=False)
                    
                    # Extract embeddings for the query sequence (last in batch)
                    # Remove the start token
                    query_embedding = results["representations"][layer_idx][0, -1, 1:len(sequence)+1, :].cpu().numpy()
                    sequence_embeddings.append(query_embedding)
                    batch_sizes.append(len(msa_batch))
                
                # Calculate weighted average of embeddings across all batches
                if len(sequence_embeddings) > 0:
                    avg_embedding = np.average(sequence_embeddings, axis=0, weights=batch_sizes)
                    # Return the raw embeddings - position selection and pooling will be handled by PositionSpecificMixin
                    all_embeddings.append(np.expand_dims(avg_embedding, axis=0))
                else:
                    logger.warning(f"No embeddings generated for sequence {sequence.id}")
                    # Create empty embedding of appropriate shape
                    shape = [1, len(sequence), self.embedding_dim_]
                    all_embeddings.append(np.zeros(shape))

            # Stack or return individual embeddings based on shape
            if len(all_embeddings) > 0 and all(e.shape == all_embeddings[0].shape for e in all_embeddings):
                return np.vstack(all_embeddings)
            else:
                return all_embeddings

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (Optional[List[str]]): Ignored. Present for API consistency.

        Returns:
            List[str]: Output feature names.
        """
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() before using this method.")
        
        positions = self.positions
        embedding_dim = self.embedding_dim_
        
        if self.pool:
            return [f"MSA_emb{i}" for i in range(embedding_dim)]
        elif self.flatten and positions is not None:
            return [f"pos{p}_emb{i}" for p in positions for i in range(embedding_dim)]
        elif positions is not None:
            return [f"pos{p}" for p in positions]
        else:
            # Without positions and pooling, we can't provide meaningful feature names
            raise ValueError("Cannot determine feature names without positions when not pooling")