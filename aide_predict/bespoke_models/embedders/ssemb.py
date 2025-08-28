# aide_predict/bespoke_models/embedders/ssemb.py
'''
* Author: Evan Komp
* Created: 5/6/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper for SSEmb embeddings:
Blaabjerg, L.M., Jonsson, N., Boomsma, W. et al. SSEmb: A joint embedding of protein sequence and structure enables robust variant effect predictions. 
    Nat Commun 15, 9646 (2024). https://doi.org/10.1038/s41467-024-53982-z
'''
import os
import sys
import subprocess
import tempfile
from typing import List, Union, Optional
import warnings
import numpy as np
import h5py

from aide_predict.bespoke_models.base import (
    ProteinModelWrapper, 
    PositionSpecificMixin,
    RequiresMSAPerSequenceMixin,
    RequiresStructureMixin,
    ExpectsNoFitMixin,
    CacheMixin
)
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence, ProteinStructure
from aide_predict.utils.common import MessageBool
import aide_predict

import logging
logger = logging.getLogger(__name__)

# Check environment setup
SSEMB_ENV = os.environ.get('SSEMB_CONDA_ENV')
SSEMB_REPO = os.environ.get('SSEMB_REPO')

SSEMB_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(aide_predict.__file__)),
    "external_calls",
    "ssemb",
    "_ssemb_emb.py"
)

if SSEMB_ENV is None or SSEMB_REPO is None:
    AVAILABLE = MessageBool(False, "SSEmb requires SSEMB_CONDA_ENV and SSEMB_REPO environment variables to be set")
elif not os.path.exists(SSEMB_REPO):
    AVAILABLE = MessageBool(False, f"SSEmb repository directory not found: {SSEMB_REPO}")
elif not os.path.exists(os.path.join(SSEMB_REPO, "weights")):
    AVAILABLE = MessageBool(False, f"SSEmb weights directory not found: {os.path.join(SSEMB_REPO, 'weights')}")
else:
    AVAILABLE = MessageBool(True, "SSEmb model is available")


class SSEmbEmbedding(
    PositionSpecificMixin,
    RequiresStructureMixin,
    RequiresMSAPerSequenceMixin,
    ExpectsNoFitMixin,
    CacheMixin,
    ProteinModelWrapper
):
    """
    A protein sequence embedder that uses the SSEmb model to generate joint sequence-structure embeddings.
    
    This class wraps the SSEmb model to provide embeddings for protein sequences.
    It requires a structure and MSA for the protein, and outputs embeddings that
    combine structural and evolutionary information.
    
    The embeddings are of shape (L, 256) where L is the sequence length, representing
    a 256-dimensional vector per residue position.
    
    Attributes:
        _available (MessageBool): Indicates whether SSEmb is available based on environment setup.
        gpu_id (int): GPU device ID to use for embedding generation.
        embedding_dim_ (int): The dimensionality of the embeddings (256).
        positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
        pool (bool): Whether to pool the encoded vectors across positions.
        flatten (bool): Whether to flatten the output array.
    """

    _available = AVAILABLE

    def __init__(
        self,
        metadata_folder: str = None,
        wt: Optional[Union[str, ProteinSequence]] = None,
        positions: Optional[List[int]] = None,
        flatten: bool = False,
        pool: bool = False,
        gpu_id: int = 0,
        use_cache: bool = True,
        batch_size: int = 5
    ):
        """
        Initialize the SSEmbEmbedding.

        Args:
            metadata_folder (str, optional): Folder to store metadata and intermediate files.
            wt (Optional[Union[str, ProteinSequence]]): Wild-type protein sequence.
            positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
            flatten (bool): Whether to flatten the output array.
            pool (bool): Whether to pool the encoded vectors across positions.
            gpu_id (int): GPU device ID to use. Defaults to 0.
            use_cache (bool): Whether to cache embeddings to avoid redundant computations.
            batch_size (int): Number of sequences to process in each batch.
        """
        super().__init__(
            metadata_folder=metadata_folder, 
            wt=wt, 
            positions=positions, 
            pool=pool, 
            flatten=flatten,
            use_cache=use_cache
        )
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.embedding_dim_ = 256  # SSEmb outputs 256-dimensional embeddings
        
        # Create necessary directories in metadata folder
        self._io_dir = os.path.join(self.metadata_folder, "ssemb_io")
        self._pdb_dir = os.path.join(self._io_dir, "pdbs")
        self._msa_dir = os.path.join(self._io_dir, "msas")
        self._emb_dir = os.path.join(self.metadata_folder, "embeddings")
        
        os.makedirs(self._io_dir, exist_ok=True)
        os.makedirs(self._pdb_dir, exist_ok=True)
        os.makedirs(self._msa_dir, exist_ok=True)
        os.makedirs(self._emb_dir, exist_ok=True)

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'SSEmbEmbedding':
        """
        Fit the SSEmb embedder (no actual fitting required as it's pre-trained).

        Args:
            X (ProteinSequences): Input protein sequences, each with its own MSA.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            SSEmbEmbedding: The fitted embedder.
        """
        
        self.fitted_ = True
        return self

    def _transform(self, X: ProteinSequences) -> List[np.ndarray]:
        """
        Generate SSEmb embeddings for the input sequences.

        Args:
            X (ProteinSequences): Input protein sequences to embed.

        Returns:
            List[np.ndarray]: SSEmb embeddings for the sequences. Each item is an array
            of shape (L, 256) where L is the sequence length.

        Raises:
            RuntimeError: If SSEmb embedding generation fails.
        """
        all_embeddings = []
        seq_to_idx = {}  # Map sequence to its index in X
        
        # Process sequences in batches
        for batch_start in range(0, len(X), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(X))
            batch = X[batch_start:batch_end]
            
            # Prepare data for this batch
            pdb_paths = []
            msa_paths = []
            sequence_ids = []
            
            # Setup files for each sequence in the batch
            for i, seq in enumerate(batch):
                seq_idx = batch_start + i
                seq_id = seq.id if seq.id else f"seq_{seq_idx}"
                seq_id = seq_id.replace("/", "_").replace(" ", "_")
                sequence_ids.append(seq_id)
                seq_to_idx[seq_id] = seq_idx
                
                # Get structure path
                if seq.structure is None:
                    assert self.wt.structure
                    structure_path = self.wt.structure.pdb_file
                    logger.warning(f"Using WT structure for sequence {seq_id}")
                else:
                    structure_path = seq.structure.pdb_file
                
                # Validate structure
                struct = ProteinStructure(pdb_file=structure_path)
                if not struct.validate_sequence(str(seq)):
                    logger.warning(f"Structure sequence doesn't exactly match {seq_id}. This may cause issues.")
                
                # Create symlink to PDB in the io directory
                pdb_dest = os.path.join(self._pdb_dir, f"{seq_id}.pdb")
                if os.path.exists(pdb_dest) and os.path.islink(pdb_dest):
                    os.unlink(pdb_dest)
                elif os.path.exists(pdb_dest):
                    os.remove(pdb_dest)
                
                os.symlink(os.path.abspath(structure_path), pdb_dest)
                pdb_paths.append(pdb_dest)
                
                # Save MSA to file
                if not seq.has_msa:
                    raise ValueError(f"Sequence {seq_id} must have an MSA")
                
                msa_dest = os.path.join(self._msa_dir, f"{seq_id}.a3m")
                seq.msa.to_fasta(msa_dest)
                msa_paths.append(msa_dest)
            
            # Only process sequences that don't already have embeddings
            to_process = []
            for i, seq_id in enumerate(sequence_ids):
                emb_path = os.path.join(self._emb_dir, f"{seq_id}.h5")
                if not os.path.exists(emb_path):
                    to_process.append(i)
            
            if not to_process:
                logger.info(f"All sequences in batch {batch_start//self.batch_size + 1} already have embeddings")
            else:
                # Filter paths to only include sequences that need processing
                process_pdb_paths = [pdb_paths[i] for i in to_process]
                process_msa_paths = [msa_paths[i] for i in to_process]
                process_seq_ids = [sequence_ids[i] for i in to_process]
                
                logger.info(f"Processing batch {batch_start//self.batch_size + 1} with {len(process_pdb_paths)} sequences")
                
                # Create batch output directory
                batch_dir = os.path.join(self._emb_dir, f"batch_{batch_start//self.batch_size}")
                os.makedirs(batch_dir, exist_ok=True)
                
                # Run SSEmb for the batch
                cmd = [
                    "conda", "run",
                    "-n", SSEMB_ENV,
                    "--no-capture-output",
                    "python", SSEMB_SCRIPT,
                    "--pdb"] + process_pdb_paths + [
                    "--msa"] + process_msa_paths + [
                    "--output", batch_dir,
                    "--weights", os.path.join(SSEMB_REPO, "weights"),
                    "--gpu-id", str(self.gpu_id)
                ]
                
                logger.info(f"Running SSEmb with command: {' '.join(cmd)}")
                
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    logger.debug(f"SSEmb stdout: {result.stdout}")
                    if result.stderr:
                        logger.warning(f"SSEmb stderr: {result.stderr}")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"SSEmb embeddings generation failed: {e.stderr}")
                
                # Move embeddings to their individual files
                combined_emb_file = os.path.join(batch_dir, "ssemb_embeddings.h5")
                if not os.path.exists(combined_emb_file):
                    raise RuntimeError(f"SSEmb failed to produce embeddings at {combined_emb_file}")
                
                # Extract embeddings for each sequence
                with h5py.File(combined_emb_file, 'r') as f:
                    for seq_id in process_seq_ids:
                        # Find the matching protein in the HDF5 file
                        # The exact key might vary based on how SSEmb names the groups
                        found = False
                        for protein_name in f.keys():
                            # Try to match by substring of protein name in case of key formatting differences
                            if seq_id in protein_name:
                                # Save to individual file
                                emb_path = os.path.join(self._emb_dir, f"{seq_id}.h5")
                                with h5py.File(emb_path, 'w') as out_f:
                                    out_f.create_dataset(protein_name, data=f[protein_name][:])
                                    for attr_name, attr_value in f[protein_name].attrs.items():
                                        out_f[protein_name].attrs[attr_name] = attr_value
                                found = True
                                break
                        
                        if not found:
                            raise RuntimeError(f"No embedding found for sequence {seq_id} in output file")
            
            # Load embeddings for all sequences in this batch
            for seq_id in sequence_ids:
                emb_path = os.path.join(self._emb_dir, f"{seq_id}.h5")
                if not os.path.exists(emb_path):
                    raise RuntimeError(f"No embedding file found for sequence {seq_id}")
                
                try:
                    with h5py.File(emb_path, 'r') as f:
                        protein_name = list(f.keys())[0]
                        embedding = f[protein_name][:]
                        all_embeddings.append(embedding)
                except Exception as e:
                    raise RuntimeError(f"Error loading embedding for sequence {seq_id}: {str(e)}")
        
        # Reorder embeddings to match original sequence order
        ordered_embeddings = [None] * len(X)
        for seq_id, idx in seq_to_idx.items():
            emb_idx = list(seq_to_idx.keys()).index(seq_id)
            # add a dimension at zero
            ordered_embeddings[idx] = np.expand_dims(all_embeddings[emb_idx], axis=0)
        
        return ordered_embeddings

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
        
        if self.pool:
            return [f"SSEmb_emb{i}" for i in range(self.embedding_dim_)]
        elif self.flatten and positions is not None:
            return [f"pos{p}_emb{i}" for p in positions for i in range(self.embedding_dim_)]
        elif positions is not None:
            return [f"pos{p}" for p in positions]
        else:
            # Without positions and pooling, we can't provide meaningful feature names
            raise ValueError("Cannot determine feature names without positions when not pooling")