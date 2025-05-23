# aide_predict/data_structures.py
'''
* Author: Evan Komp
* Created: 6/21/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Base data structures for the AIDE Predict package Where they do not exist in sklearn.
'''
from collections import UserList
import os
import warnings
import numpy as np
import hashlib

from aide_predict.io.bio_files import read_fasta, write_fasta, read_a3m
from aide_predict.utils.alignment_calls import sw_global_pairwise, mafft_align

from typing import List, Optional, Union, Iterator, Dict, Iterable, Any, Tuple

from ..constants import AA_SINGLE, GAP_CHARACTERS, NON_CONONICAL_AA_SINGLE
from .structures import ProteinStructure

HASHER = hashlib.sha256


############################################
# A class to store a single protein character and sequence
# They are treated like strings, eg you can do all that you expect from an AA string
# but also stores id, structure, and has additional methods to check for gaps, non-canonical AAs, etc.
############################################

class ProteinCharacter(str):
    """
    Represents a single character in a protein sequence.

    This class inherits from UserString and provides additional properties
    to check the nature of the amino acid character.
    """

    def __new__(cls, seq: str):
        """
        Create a new ProteinCharacter object.

        Args:
            seq (str): A single character representing an amino acid or gap.

        Returns:
            ProteinCharacter: The new ProteinCharacter object.
        """
        obj =  str.__new__(cls, seq)
        if len(obj) != 1:
            raise ValueError("ProteinCharacter must be initialized with a single character")
        if obj.upper() not in AA_SINGLE.union(GAP_CHARACTERS).union(NON_CONONICAL_AA_SINGLE):
            raise ValueError(f"Invalid character {obj} for protein sequence.")
        return obj

    @property
    def is_gap(self) -> bool:
        """Check if the character represents a gap in the sequence."""
        return self in GAP_CHARACTERS

    @property
    def is_non_canonical(self) -> bool:
        """Check if the character represents a non-canonical amino acid."""
        return self in NON_CONONICAL_AA_SINGLE
    
    @property
    def is_not_focus(self) -> bool:
        """
        Check if the character is not in focus.
        
        A character is considered not in focus if it's a gap or a lowercase letter.
        """
        return self.is_gap or self.islower()

    def __repr__(self) -> str:
        """Return a string representation of the ProteinCharacter."""
        return f"ProteinCharacter('{self}')"

#############################################
# A class to store a protein sequence
###############################################


class ProteinSequence(str):
    """
    Represents a protein sequence.

    This class inherits from UserString and provides additional methods and properties
    for analyzing and manipulating protein sequences.
    """
    def __new__(cls, seq: str, id: Optional[str] = None, structure: Optional[Union[str, "ProteinStructure"]] = None, msa: Optional["ProteinSequences"] = None):
        """
        Create a new ProteinSequence object.

        Args:
            seq (str): The amino acid sequence.
            id (Optional[str]): An identifier for the sequence.
            structure (Optional[Union[str, "ProteinStructure"]): The structure of the protein sequence.
            weights: Optional[np.ndarray]: Weights for each sequence. If None, initialized as ones.
        Returns:
            ProteinSequence: The new ProteinSequence object.
        """
        obj = str.__new__(cls, seq)
        obj._characters: List[ProteinCharacter] = [ProteinCharacter(c) for c in seq]
        obj._id: Optional[str] = None
        obj._structure: Optional[Union[str, "ProteinStructure"]] = None
        obj._msa: Optional["ProteinSequences"] = None

        if id is not None:
            obj.id = id
        if structure is not None:
            obj.structure = structure
        if msa is not None:
            obj.msa = msa

        return obj

    @property
    def id(self) -> Optional[str]:
        """Get the identifier of the sequence."""
        return self._id
    
    def __hash__(self) -> int:
        """Compute a hash value for the ProteinSequence."""
        if not self.has_msa:
            msa_hash = None
        else:
            msa_hash = hash(self.msa)

        fingerprint = str((tuple(self._characters), self._id, self._structure, msa_hash))
        hash_ob = HASHER(fingerprint.encode())
        hash_int = int(hash_ob.hexdigest(), 16)
        return hash_int
    
    def __eq__(self, other: object) -> bool:
        """Check if two ProteinSequence objects are equal."""
        return hash(self) == hash(other)
    
    def __ne__(self, other: object) -> bool:
        """Check if two ProteinSequence objects are not equal."""
        return not self == other

    def __repr__(self) -> str:
        """Return a string representation of the ProteinSequence."""
        return f"ProteinSequence(id={self._id!r}, seq='{self[:20]}{'...' if len(self) > 20 else ''}')"

    @property
    def has_gaps(self) -> bool:
        """Check if the sequence contains any gaps."""
        return any(c.is_gap for c in self._characters)
    
    @property
    def has_non_canonical(self) -> bool:
        """Check if the sequence contains any non-canonical amino acids."""
        return any(c.is_non_canonical for c in self._characters)

    def with_no_gaps(self) -> 'ProteinSequence':
        """Return a new ProteinSequence with all gaps removed."""
        return ProteinSequence("".join(c for c in self if c not in GAP_CHARACTERS),
                               id=self._id, structure=self._structure, msa=self._msa)
    
    @property
    def as_array(self) -> np.ndarray:
        """Convert the sequence to a numpy array of characters."""
        return np.array([c for c in self._characters]).reshape(1,-1)

    @property
    def num_gaps(self) -> int:
        """Get the number of gaps in the sequence."""
        return sum(c.is_gap for c in self._characters)
    
    @property
    def base_length(self) -> int:
        """Get the length of the sequence excluding gaps."""
        return len(self) - self.num_gaps

    def _mutate(self, position: int, new_character: str) -> 'ProteinSequence':
        """
        Create a new ProteinSequence with a mutation at the specified position.

        Args:
            position (int): The position to mutate.
            new_character (str): The new character to insert at the position.

        Returns:
            ProteinSequence: A new ProteinSequence with the mutation applied.
        """
        if position < 0 or position >= len(self):
            raise ValueError("Position out of range")
        new_seq = self[:position] + new_character + self[position+1:]
        return ProteinSequence(new_seq, structure=self._structure)  # Note: id is not passed to indicate mutation
    
    def mutate(self, mutations: Union[str, List[str]], one_indexed: bool = True):
        """Create a new ProteinSequence with mutations applied.
        
        Params
        ------
        mutations: Union[str, List[str]]
            A single mutation in the format 'A123B' or a list of mutations.
        one_indexed: bool
            If True, positions are one-indexed. If False, positions are zero-indexed.
        """
        if isinstance(mutations, str):
            mutations = [mutations]

        new_seq = self
        for mutation_string in mutations:
            original, position, new = mutation_string[0], int(mutation_string[1:-1]), mutation_string[-1]
            if one_indexed:
                position -= 1
            new_seq = new_seq._mutate(position, new)
        return new_seq


    def mutated_positions(self, other: Union[str, 'ProteinSequence']) -> List[int]:
        """
        Find positions where this sequence differs from another.

        Args:
            other (Union[str, ProteinSequence]): The sequence to compare against.

        Returns:
            List[int]: A list of positions where the sequences differ.
        """
        other_seq = other if isinstance(other, ProteinSequence) else other
        return [i for i, (a, b) in enumerate(zip(self, other_seq)) if a != b]
    
    def get_mutations(self, other: Union[str, 'ProteinSequence']) -> List[str]:
        """
        Find mutations between this sequence and another.

        Args:
            other (Union[str, ProteinSequence]): The sequence to compare against.

        Returns:
            List[str]: A list of mutations in the format 'A123B' where A is the original character,
            123 is the position, and B is the new character.
        """
        other_seq = other if isinstance(other, ProteinSequence) else other
        return [f"{a}{i+1}{b}" for i, (a, b) in enumerate(zip(self, other_seq)) if a != b]

    def get_protein_character(self, position: int) -> ProteinCharacter:
        """
        Get the ProteinCharacter at the specified position.

        Args:
            position (int): The position to get the character from.

        Returns:
            ProteinCharacter: The character at the specified position.
        """
        if position < 0 or position >= len(self):
            raise IndexError("Position out of range")
        return self._characters[position]

    def slice_as_protein_sequence(self, start: int, end: int) -> 'ProteinSequence':
        """
        Create a new ProteinSequence from a slice of this sequence.

        Args:
            start (int): The start position of the slice.
            end (int): The end position of the slice.

        Returns:
            ProteinSequence: A new ProteinSequence containing the specified slice.
        """
        return ProteinSequence(self[start:end], id=self._id, structure=self._structure)

    def iter_protein_characters(self) -> Iterator[ProteinCharacter]:
        """
        Iterate over the ProteinCharacters in the sequence.

        Returns:
            Iterator[ProteinCharacter]: An iterator over the ProteinCharacters.
        """
        return iter(self._characters)

    @property
    def id(self) -> Optional[str]:
        """Get the identifier of the sequence."""
        return self._id
    
    @id.setter
    def id(self, new_id: str) -> None:
        """Set the identifier of the sequence."""
        self._id = new_id

    @property
    def structure(self) -> Optional[str]:
        """Get the structure of the sequence."""
        return self._structure
    
    @structure.setter
    def structure(self, new_structure: str) -> None:
        """Set the structure of the sequence."""
        if isinstance(new_structure, str):
            new_structure = ProteinStructure(new_structure)
            if not len(self) == len(new_structure.get_sequence()):
                warnings.warn("Length of sequence and residues in structure do not match. This will likely cause problems with downstream predictors.")
        elif not isinstance(new_structure, ProteinStructure):
            raise ValueError("Structure must be a ProteinStructure object or a valid PDB file path.")
        
        self._structure = new_structure

    @property
    def has_structure(self) -> bool:
        """Check if the sequence has an associated structure."""
        return self._structure is not None

    @property
    def msa(self) -> Optional['ProteinSequences']:
        """Get the MSA (multiple sequence alignment) of the sequence."""
        return self._msa

    @msa.setter
    def msa(self, new_msa: 'ProteinSequences') -> None:
        """Set the MSA (multiple sequence alignment) of the sequence."""
        if not isinstance(new_msa, ProteinSequences):
            raise ValueError("MSA must be a ProteinSequences object.")
        if not new_msa.aligned:
            raise ValueError("MSA must be aligned to be assigned to a sequence.")
        self._msa = new_msa

    @property
    def has_msa(self) -> bool:
        """Check if the sequence has an associated MSA."""
        return self._msa is not None
    
    @property
    def msa_same_width(self) -> bool:
        """Check if the MSA has the same width as the sequence."""
        if self._msa is None:
            return False
        return self._msa.width == len(self)
    
    @property
    def is_in_msa(self) -> bool:
        """Check if the sequence is part of an MSA."""
        if not self.has_msa:
            return False
        self_str = str(self.with_no_gaps())
        for seq in self._msa:
            if str(seq.with_no_gaps()) == self_str:
                return True
        return False

    @classmethod
    def from_pdb(cls, pdb_file: str, chain: str = 'A', id: Optional[str] = None) -> 'ProteinSequence':
        """
        Create a ProteinSequence from a PDB file.

        This method extracts the amino acid sequence from the PDB file and creates
        a ProteinSequence object with the associated structure.

        Args:
            pdb_file (str): Path to the PDB file.
            chain (str): Chain identifier to extract sequence from. Defaults to 'A'.
            id (Optional[str]): Identifier for the sequence. If None, uses the PDB filename.

        Returns:
            ProteinSequence: A new ProteinSequence object with the extracted sequence and structure.

        Raises:
            FileNotFoundError: If the PDB file does not exist.
            ValueError: If the specified chain is not found in the PDB file.
            
        Example:
            >>> seq = ProteinSequence.from_pdb("1abc.pdb", chain='A', id='my_protein')
            >>> print(seq)
            'MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG...'
            >>> print(seq.structure)
            ProteinStructure(pdb_file='1abc.pdb', chain='A')
        """
        # Create structure object (this validates file existence)
        structure = ProteinStructure(pdb_file=pdb_file, chain=chain)
        
        # Extract sequence from structure
        sequence = structure.get_sequence()
        
        # Use PDB filename as ID if none provided
        if id is None:
            id = os.path.splitext(os.path.basename(pdb_file))[0]
        
        # Create sequence object with structure
        return cls(sequence, id=id, structure=structure)
    
    @classmethod
    def from_fasta(cls, fasta_file: str) -> 'ProteinSequence':
        """Create a ProteinSequence from a FASTA file, assuming the first sequence is the query."""
        msa = ProteinSequences.from_fasta(fasta_file)
        self_seq = msa[0].with_no_gaps()
        self_seq.msa = msa
        return self_seq

    @classmethod
    def from_a3m(cls, a3m_file: str, inserts: str='first') -> 'ProteinSequence':
        """Create a ProteinSequence from an A3M file, assuming the first sequence is the query."""
        msa = ProteinSequences.from_a3m(a3m_file, inserts=inserts)
        self_seq = msa[0].with_no_gaps()
        self_seq.msa = msa
        return self_seq
    
    def align(self, other: 'ProteinSequence') -> 'ProteinSequence':
        """
        Align this sequence with another using global pairwise alignment.

        Args:
            other (ProteinSequence): The sequence to align with.

        Returns:
            ProteinSequence: The aligned sequence.
        """
        if isinstance(other, ProteinSequence):
            base_self = self.with_no_gaps()
            base_other = other.with_no_gaps()

            aligned_seq, aligned_other = sw_global_pairwise(base_self, base_other)
            return aligned_seq, aligned_other
        elif isinstance(other, ProteinSequences):
            if not other.aligned:
                raise ValueError("Assuming the other is an alignment, but it is not aligned.")
            sequences_self = ProteinSequences([self])
            aligned_self = sequences_self.align_to(other, realign=False)
            aligned_self = aligned_self[0]
            aligned_self.msa = other
            return aligned_self
    
    def saturation_mutagenesis(self, positions: List[int]=None) -> List['ProteinSequence']:
        """
        Perform saturation mutagenesis at the specified positions.

        Args:
            positions (List[int]): The positions to mutate.

        Returns:
            ProteinSequences: A list of mutated sequences.
        """
        sequences = []
        if positions is None:
            positions = range(len(self))
        for i in positions:
            for aa in AA_SINGLE:
                if aa != self[i]:
                    mutated = self._mutate(i, aa)
                    mutated.id = f"{self[i]}{i+1}{aa}"
                    sequences.append(mutated)
        return ProteinSequences(sequences)
    
    def upper(self) -> 'ProteinSequence':
        """Return a new ProteinSequence with all characters converted to uppercase."""
        return ProteinSequence(str(self).upper(), id=self._id, structure=self._structure, msa=self._msa)


############################################
# A class to store multiple ProteinSequence objects
# Think of this as a dataset of protein sequences, no labels eg. the X for proteins
# Useful additional functionalities built in: alignment, fasta read/write, testing for alignment, gaps,
# fixed length, etc.
############################################

class ProteinSequences(UserList):
    """
    A collection of ProteinSequence objects with additional functionality.
    
    Attributes:
        aligned (bool): True if all sequences have the same length, False otherwise.
        fixed_length (bool): True if all sequences have the same base length, False otherwise.
        width (Optional[int]): The length of the sequences if aligned, None otherwise.
        has_gaps (bool): True if any sequence has gaps, False otherwise.
        mutated_positions (Optional[List[int]]): List of mutated positions if aligned, None otherwise.

    Methods:
        to_dict: Convert ProteinSequences to a dictionary.
        to_fasta: Write sequences to a FASTA file.
        from_fasta: Create a ProteinSequences object from a FASTA file.
    """

    def __init__(self, sequences: List[ProteinSequence], weights: Optional[np.ndarray] = None):
        """
        Initialize a ProteinSequences object.

        Args:
            sequences (List[ProteinSequence]): A list of ProteinSequence objects.
            weights (Optional[np.ndarray]): Weights for each sequence. If None, initialized as ones.
        """
        for s in sequences:
            if not isinstance(s, ProteinSequence):
                raise ValueError("All elements must be ProteinSequence objects")
        super().__init__(sequences)
        self._id_to_pos = None
        
        self._weights = None
        if weights is None:
            self.weights = np.ones(len(self))
        else:
            if len(weights) != len(self):
                raise ValueError("Length of weights must match the number of sequences")
            self.weights = weights

    @property
    def weights(self) -> np.ndarray:
        """Get the weights for each sequence."""
        return self._weights
    
    @weights.setter
    def weights(self, new_weights: np.ndarray) -> None:
        """Set the weights for each sequence."""
        if type(new_weights) != np.ndarray:
            new_weights = np.array(new_weights).reshape(-1)
        if len(new_weights) != len(self):
            raise ValueError("Length of weights must match the number of sequences")
        self._weights = new_weights

    @property
    def aligned(self) -> bool:
        """
        Check if all sequences are of equal length (including gaps).

        Returns:
            bool: True if all sequences have the same length, False otherwise.
        """
        return len(set(len(seq) for seq in self)) == 1 and (len(self) > 1 or self.has_gaps)

    @property
    def fixed_length(self) -> bool:
        """
        Check if all contained sequences have the same base length (excluding gaps).

        Returns:
            bool: True if all sequences have the same base length, False otherwise.
        """
        return len(set(seq.base_length for seq in self)) == 1

    @property
    def width(self) -> Optional[int]:
        """
        Get the length of the sequences if aligned.

        Returns:
            Optional[int]: The length of the sequences if aligned, None otherwise.
        """
        return len(self[0]) if self.aligned or len(self)==1 else None

    @property
    def has_gaps(self) -> bool:
        """
        Check if any sequences have gaps.

        Returns:
            bool: True if any sequence has gaps, False otherwise.
        """
        return any(seq.has_gaps for seq in self)

    @property
    def mutated_positions(self) -> Optional[List[int]]:
        """
        List columns that have more than one character, assuming sequences are aligned.

        Returns:
            Optional[List[int]]: List of mutated positions if aligned, None otherwise.
        """
        if not self.aligned:
            warnings.warn("Sequences are not aligned. Cannot determine mutated positions.")
            return None

        if not self:  # If the list is empty
            return []

        mutated = []
        seq_length = len(self[0])
        for i in range(seq_length):
            chars = set(seq.get_protein_character(i) for seq in self)
            if len(chars) > 1:
                mutated.append(i)
        return mutated
    
    def __getitem__(self, index: Union[int, slice, str]) -> Union[ProteinSequence, 'ProteinSequences']:
        """
        Get a ProteinSequence or a subset of sequences.

        Args:
            index (Union[int, slice, str]): Index, slice, or ID of the sequence(s).

        Returns:
            Union[ProteinSequence, ProteinSequences]: The requested sequence or a new ProteinSequences object.
        """
        if isinstance(index, str):
            index = self.id_mapping[index]
        elif isinstance(index, np.ndarray):
            return ProteinSequences([self[i] for i in index])
        return super().__getitem__(index)
    
    def __hash__(self) -> str:
        individual_hashes = [hash(seq) for seq in self]
        hash_obj = HASHER()
        fingerprint = str(tuple(individual_hashes))
        hash_obj.update(fingerprint.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        return hash_int

    def to_dict(self) -> Dict[str, str]:
        """
        Convert ProteinSequences to a dictionary.

        Returns:
            Dict[str, str]: A dictionary with sequence IDs as keys and sequences as values.
        """
        return {seq.id or hash(seq): str(seq) for seq in self}

    def to_fasta(self, output_path: str):
        """
        Write sequences to a FASTA file.

        Args:
            output_path (str): The path to the output FASTA file.
        """

        with open(output_path, 'w') as f:
            write_fasta(((seq.id or hash(seq), str(seq)) for seq in self), f)

    @classmethod
    def from_fasta(cls, input_path: str) -> 'ProteinSequences':
        """
        Create a ProteinSequences object from a FASTA file.

        Args:
            input_path (str): The path to the input FASTA file.

        Returns:
            ProteinSequences: A new ProteinSequences object containing the sequences from the FASTA file.
        """

        sequences = []
        with open(input_path, 'r') as f:
            for id, seq in read_fasta(f):
                sequences.append(ProteinSequence(seq, id=id))
        return cls(sequences)
    
    @classmethod
    def from_a3m(cls, input_path: str, inserts: str='first') -> 'ProteinSequences':
        """
        Create a ProteinSequences object from an A3M file.

        Args:
            input_path (str): The path to the input A3M file.

        Returns:
            ProteinSequences: A new ProteinSequences object containing the sequences from the A3M file.
        """

        sequences = []
        with open(input_path, 'r') as f:
            for id, seq in read_a3m(f, inserts).items():
                sequences.append(ProteinSequence(seq, id=id))
        return cls(sequences)
    
    @classmethod
    def from_dict(cls, sequences: Dict[str, str]) -> 'ProteinSequences':
        """
        Create a ProteinSequences object from a dictionary.

        Args:
            sequences (Dict[str, str]): A dictionary with sequence IDs as keys and sequences as values.

        Returns:
            ProteinSequences: A new ProteinSequences object containing the sequences from the dictionary.
        """
        return cls([ProteinSequence(seq, id=id) for id, seq in sequences.items()])
    
    @classmethod
    def from_list(cls, sequences: List[str]) -> 'ProteinSequences':
        """
        Create a ProteinSequences object from a list of sequences.

        Args:
            sequences (List[str]): A list of protein sequences.

        Returns:
            ProteinSequences: A new ProteinSequences object containing the sequences from the list.
        """
        return cls([ProteinSequence(seq) for seq in sequences])
    
    @classmethod
    def from_csv(cls, filepath: str, id_col: Optional[str] = None, seq_col: Optional[str] = None, 
                label_cols: Optional[Union[str, List[str]]] = None, **kwargs) -> Union['ProteinSequences', Tuple['ProteinSequences', np.ndarray]]:
        """
        Create a ProteinSequences object from a CSV file.

        Args:
            filepath (str): Path to the CSV file.
            id_col (Optional[str]): Name of column containing sequence IDs. If None, sequences will be assigned numeric IDs.
            seq_col (Optional[str]): Name of column containing sequences. If None, uses first column.
            label_cols (Optional[Union[str, List[str]]]): Name(s) of columns containing labels to return.
            **kwargs: Additional arguments passed to pandas.read_csv().

        Returns:
            Union[ProteinSequences, Tuple[ProteinSequences, np.ndarray]]: 
                - If label_cols is None: ProteinSequences object
                - If label_cols is provided: Tuple of (ProteinSequences, labels array)

        Raises:
            ValueError: If specified columns are not found in the CSV file.
            ValueError: If any sequence contains invalid characters.
        """
        import pandas as pd
        df = pd.read_csv(filepath, **kwargs)
        return cls.from_df(df, id_col=id_col, seq_col=seq_col, label_cols=label_cols)

    @classmethod
    def from_df(cls, df: 'pd.DataFrame', id_col: Optional[str] = None, seq_col: Optional[str] = None,
                label_cols: Optional[Union[str, List[str]]] = None) -> Union['ProteinSequences', Tuple['ProteinSequences', np.ndarray]]:
        """
        Create a ProteinSequences object from a pandas DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing sequences.
            id_col (Optional[str]): Name of column containing sequence IDs. If None, sequences will be assigned numeric IDs.
            seq_col (Optional[str]): Name of column containing sequences. If None, uses first column.
            label_cols (Optional[Union[str, List[str]]]): Name(s) of columns containing labels to return.

        Returns:
            Union[ProteinSequences, Tuple[ProteinSequences, np.ndarray]]: 
                - If label_cols is None: ProteinSequences object
                - If label_cols is provided: Tuple of (ProteinSequences, labels array)

        Raises:
            ValueError: If specified columns are not found in the DataFrame.
            ValueError: If any sequence contains invalid characters.
        """
        # Validate and get sequence column
        if seq_col is None:
            seq_col = df.columns[0]
        if seq_col not in df.columns:
            raise ValueError(f"Sequence column '{seq_col}' not found in DataFrame. Columns: {df.columns.tolist()}")

        # Get sequences
        sequences = df[seq_col].values

        # Create ProteinSequence objects
        protein_sequences = []
        for i, seq in enumerate(sequences):
            # Get ID if specified
            if id_col is not None:
                if id_col not in df.columns:
                    raise ValueError(f"ID column '{id_col}' not found in DataFrame. Columns: {df.columns.tolist()}")
                seq_id = str(df[id_col].iloc[i])
            else:
                seq_id = str(i)
            
            protein_sequences.append(ProteinSequence(str(seq), id=seq_id))

        # Create ProteinSequences object
        result = cls(protein_sequences)

        # Handle labels if requested
        if label_cols is not None:
            if isinstance(label_cols, str):
                label_cols = [label_cols]
            
            # Validate label columns exist
            missing_cols = [col for col in label_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Label column(s) not found in DataFrame: {missing_cols}")
            
            # Extract labels
            labels = df[label_cols].values
            if len(label_cols) == 1:
                labels = labels.reshape(-1, 1)
            
            return result, labels
        
        return result

    
    @property
    def id_mapping(self) -> Dict[str, int]:
        if self._id_to_pos is None or (self._id_to_pos and len(self._id_to_pos) != len(self)):
            self._id_to_pos = self.get_id_mapping()
        return self._id_to_pos
    
    def upper(self) -> 'ProteinSequences':
        """Return a new ProteinSequences with all sequences converted to uppercase."""
        return ProteinSequences([seq.upper() for seq in self])
    
    def has_lower(self) -> bool:
        """Check if any sequence contains lowercase characters."""
        if self.aligned:
            array = self.as_array()
            return np.any(np.char.islower(array))
        else:
            return any(c.upper() != c for seq in self for c in seq)

    def __repr__(self) -> str:
        """
        Return a string representation of the ProteinSequences object.

        Returns:
            str: A string representation of the object.
        """
        return f"ProteinSequences(count={len(self)})"
    
    def to_on_file(self, output_path: str) -> None:
        """
        Write sequences to a FASTA file.

        Args:
            output_path (str): The path to the output FASTA file.
        """
        with open(output_path, 'w') as f:
            write_fasta(((seq.id or hash(seq), str(seq)) for seq in self), f)
        return ProteinSequencesOnFile(output_path)
    
    def as_array(self) -> np.ndarray:
        """Convert the sequence to a numpy array of characters."""
        if not self.aligned and len(self) > 1:
            raise ValueError("Sequences must be aligned to convert to array.")
        return np.vstack([seq.as_array for seq in self])


    def iter_batches(self, batch_size: int) -> Iterable['ProteinSequences']:
        """
        Iterate over batches of sequences.

        Args:
            batch_size (int): The size of each batch.

        Yields:
            ProteinSequences: A batch of sequences.
        """
        for i in range(0, len(self), batch_size):
            yield ProteinSequences(self[i:i+batch_size])

    def msa_process(self, focus_seq_id: str=None, **kwargs) -> 'ProteinSequence':
        """
        Align this sequence with another using global pairwise alignment.

        Kwargs:
            **kwargs: Additional arguments to pass to MSAprocessing
        Returns:
            ProteinSequence: The aligned sequence.
        """
        if not self.aligned:
            raise ValueError("Sequences must be aligned to create an alignment mapping, align first.")
        
        from aide_predict.utils.msa import MSAProcessing
        msa_processor = MSAProcessing(**kwargs)
        return msa_processor.process(self, focus_seq_id=focus_seq_id)
    
    def sample(self, n: int, replace: bool = False, keep_first: bool=False, seed: int=None) -> 'ProteinSequences':
        """
        Sample n sequences from the ProteinSequences object.

        Args:
            n (int): Number of sequences to sample.
            replace (bool): Whether to sample with replacement. Default is False.

        Returns:
            ProteinSequences: A new ProteinSequences object containing the sampled sequences.

        Raises:
            ValueError: If n is greater than the number of sequences and replace is False.
        """
        if n > len(self) and not replace:
            raise ValueError(f"Cannot sample {n} sequences without replacement from a set of {len(self)} sequences.")

        weights = self.weights if hasattr(self, 'weights') else None
        if weights is None:
            weights = np.ones(len(self))

        if keep_first:
            n = n - 1
        
        if seed is not None:
            np.random.seed(seed)
        sampled_indices = np.random.choice(len(self), size=n, replace=replace, p=weights/np.sum(weights))
        if keep_first:
            sampled_indices = np.insert(sampled_indices, 0, 0)
        sampled_sequences = [self[i] for i in sampled_indices]
        
        new_sequences = ProteinSequences(sampled_sequences)
        new_sequences.weights = self.weights[sampled_indices] if hasattr(self, 'weights') else None
        
        return new_sequences

    def get_id_mapping(self) -> Dict[str, int]:
        """
        Create a mapping of sequence IDs to indices.

        Returns:
            Dict[str, int]: A dictionary where keys are sequence IDs and values are indices.
        """
        return {seq.id if seq.id else hash(seq): i for i, seq in enumerate(self)}
    
    @property
    def ids(self) -> List[str]:
        """Get a list of sequence IDs."""
        return [seq.id for seq in self]


    def align_all(self, output_fasta: Optional[str] = None) -> Union['ProteinSequences', 'ProteinSequencesOnFile']:
        """
        Align the sequences within this ProteinSequences object using MAFFT.

        Args:
            output_fasta (Optional[str]): Path to save the alignment. If None, a temporary file is used.

        Returns:
            Union[ProteinSequences, ProteinSequencesOnFile]: The aligned sequences, either in memory or on file 
            depending on output_fasta.

        Raises:
            ValueError: If the sequences already contain gaps.
            RuntimeError: If MAFFT alignment fails.
            FileNotFoundError: If MAFFT is not installed or not in PATH.
        """
        if self.has_gaps:
            raise ValueError("Sequences already contain gaps. Cannot perform alignment on gapped sequences.")

        return mafft_align(self, output_fasta=output_fasta)
    
    def align_to(self, existing_alignment: Union['ProteinSequences', 'ProteinSequencesOnFile'], 
                 realign: bool = False, return_only_new: bool = False,
                 output_fasta: Optional[str] = None) -> Union['ProteinSequences', 'ProteinSequencesOnFile']:
        """
        Align this ProteinSequences object to an existing alignment using MAFFT.

        Args:
            existing_alignment (Union[ProteinSequences, ProteinSequencesOnFile]): The existing alignment to align to.
            realign (bool): If True, realign all sequences from scratch. If False, add new sequences to existing alignment.\
            return_only_new (bool): If True, return only the newly aligned sequences. If False, return all sequences.
            output_fasta (Optional[str]): Path to save the alignment. If None, a temporary file is used.

        Returns:
            Union[ProteinSequences, ProteinSequencesOnFile]: The aligned sequences, either in memory or on file 
            depending on output_fasta.

        Raises:
            ValueError: If the sequences already contain gaps or if the existing alignment is not aligned.
            RuntimeError: If MAFFT alignment fails.
            FileNotFoundError: If MAFFT is not installed or not in PATH.
        """
        if self.has_gaps:
            raise ValueError("Sequences already contain gaps. Cannot perform alignment on gapped sequences.")

        if not existing_alignment.aligned:
            raise ValueError("Existing alignment must be aligned.")

        all_aligned = mafft_align(self, existing_alignment=existing_alignment, realign=realign, output_fasta=output_fasta)

        if return_only_new:
            id_mapping = all_aligned.get_id_mapping()
            new_indices = [id_mapping[seq.id if seq.id else str(hash(seq))] for seq in self]
            return ProteinSequences([all_aligned[i] for i in new_indices])
        
        return all_aligned

    
    def with_no_gaps(self) -> 'ProteinSequences':
        """Return a new ProteinSequences with all gaps removed."""
        return ProteinSequences([seq.with_no_gaps() for seq in self])

    def get_alignment_mapping(self) -> Dict[str, List[Optional[int]]]:
        """
        Create a mapping of original sequence positions to aligned positions for each sequence.

        Returns:
            Dict[str, List[Optional[int]]]: A dictionary where keys are sequence IDs or hashes and values are
            lists of integers. Each integer represents the position in the aligned sequence
            corresponding to the original sequence position. E.g., [0,1,2,5,6,7] indicates that
            there is a gap between amino acid 2 and 3, and 3 is in position 5 in the aligned sequence.

        Raises:
            ValueError: If the sequences are not aligned.
        """
        if not self.aligned:
            raise ValueError("Sequences must be aligned to create an alignment mapping.")

        mapping = {}
        for seq in self:
            seq_id = seq.id if seq.id else str(hash(seq))
            seq_mapping = []
            original_pos = 0
            for aligned_pos, char in enumerate(seq.iter_protein_characters()):
                if not char.is_gap:
                    seq_mapping.append(aligned_pos)
                    original_pos += 1
                else:
                    pass
            mapping[seq_id] = seq_mapping

        return mapping

    def apply_alignment_mapping(self, mapping: Dict[str, List[Optional[int]]]) -> 'ProteinSequences':
        """
        Apply an alignment mapping to the current sequences.

        Args:
            mapping (Dict[str, List[Optional[int]]]): The alignment mapping to apply.

        Returns:
            ProteinSequences: A new ProteinSequences object with aligned sequences.

        Raises:
            ValueError: If a sequence ID or hash is not found in the mapping or if the mapping is invalid.
        """
        if self.has_gaps:
            raise ValueError("Sequences already contain gaps. Cannot apply alignment mapping to gapped sequences.")

        aligned_sequences = []
        for seq in self:
            seq_id = seq.id if seq.id else str(hash(seq))
            if seq_id not in mapping:
                raise ValueError(f"Sequence ID or hash '{seq_id}' not found in the alignment mapping.")

            seq_mapping = mapping[seq_id]
            aligned_seq = ['-'] * (max(filter(None, seq_mapping)) + 1)
            for original_pos, aligned_pos in enumerate(seq_mapping):
                if original_pos >= len(seq):
                    raise ValueError(f"Invalid mapping for sequence '{seq_id}': original position {original_pos} out of range.")
                aligned_seq[aligned_pos] = seq[original_pos]

            aligned_sequences.append(ProteinSequence(''.join(aligned_seq), id=seq.id, structure=seq.structure))

        return ProteinSequences(aligned_sequences)

    
############################################
# A class with the same API as ProteinSequences but on file instead of in memory
############################################

class ProteinSequencesOnFile(ProteinSequences):
    """
    A memory-efficient representation of protein sequences stored in a FASTA file.
    
    This class maintains the same API as ProteinSequences but avoids loading all sequences
    into memory at once. It creates an index of the FASTA file for efficient access to
    individual sequences and precomputes some global properties for quick access.

    Attributes:
        aligned (bool): True if all sequences have the same length, False otherwise.
        fixed_length (bool): True if all sequences have the same base length, False otherwise.
        width (Optional[int]): The length of the sequences if aligned, None otherwise.
        has_gaps (bool): True if any sequence has gaps, False otherwise.
        mutated_positions (Optional[List[int]]): List of mutated positions if aligned, None otherwise.

    Methods:
        to_dict: Convert ProteinSequences to a dictionary.
        to_fasta: Write sequences to a FASTA file.
        from_fasta: Create a ProteinSequences object from a FASTA file.
    """

    def __init__(self, file_path: str, weights: Optional[np.ndarray] = None):
        """
        Initialize a ProteinSequencesOnFile object.

        Args:
            file_path (str): Path to the FASTA file containing protein sequences.
            weights (Optional[np.ndarray]): Weights for each sequence. If None, initialized as ones.
        """
        self.file_path: str = file_path
        self._index: Dict[str, Dict[str, Any]] = {}
        self._create_index()
        self._compute_global_properties()
        super().__init__([])  # Initialize with an empty list
        
        if weights is None:
            self.weights = np.ones(len(self._index))
        else:
            if len(weights) != len(self._index):
                raise ValueError("Length of weights must match the number of sequences in the file")
            self.weights = weights

    def _create_index(self) -> None:
        """
        Create an index of sequences in the FASTA file for efficient access.

        This method reads through the FASTA file once, creating an index with
        information about each sequence's position, length, and other properties.
        """
        with open(self.file_path, 'r') as f:
            current_id: Optional[str] = None
            seq_start: Optional[int] = None
            seq_length: int = 0
            has_gaps: bool = False
            base_length: int = 0
            file_pos: int = 0

            for line in f:
                if line.startswith('>'):
                    if current_id is not None:
                        self._index[current_id] = {
                            'start': seq_start,
                            'length': seq_length,
                            'has_gaps': has_gaps,
                            'base_length': base_length
                        }
                    current_id = line[1:].strip().split()[0]
                    seq_start = file_pos + len(line)
                    seq_length = 0
                    has_gaps = False
                    base_length = 0
                else:
                    seq = line.strip()
                    seq_length += len(seq)
                    has_gaps = has_gaps or ('-' in seq)
                    base_length += len(seq.replace('-', ''))
                file_pos += len(line)

            if current_id is not None:
                self._index[current_id] = {
                    'start': seq_start,
                    'length': seq_length,
                    'has_gaps': has_gaps,
                    'base_length': base_length
                }

    def _compute_global_properties(self) -> None:
        """
        Compute global properties based on the index.

        This method calculates properties like alignment status, fixed length,
        width, and presence of gaps across all sequences.
        """
        lengths: set = set(info['length'] for info in self._index.values())
        base_lengths: set = set(info['base_length'] for info in self._index.values())
        
        self._aligned: bool = len(lengths) == 1
        self._fixed_length: bool = len(base_lengths) == 1
        self._width: Optional[int] = next(iter(lengths)) if self._aligned else None
        self._has_gaps: bool = any(info['has_gaps'] for info in self._index.values())

    def __len__(self) -> int:
        """
        Return the number of sequences in the file.

        Returns:
            int: The number of sequences.
        """
        return len(self._index)

    def __getitem__(self, index: Union[int, str]) -> ProteinSequence:
        """
        Get a ProteinSequence by index or ID.

        Args:
            index (Union[int, str]): Index or ID of the sequence.

        Returns:
            ProteinSequence: The requested protein sequence.

        Raises:
            IndexError: If the index is out of range.
            KeyError: If the ID is not found.
        """
        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise IndexError("Index out of range")
            id = list(self._index.keys())[index]
        else:
            id = index

        if id not in self._index:
            raise KeyError(f"Sequence ID '{id}' not found")

        info = self._index[id]
        with open(self.file_path, 'r') as f:
            f.seek(info['start'])
            sequence = []
            remaining_length = info['length']
            while remaining_length > 0:
                line = f.readline().strip()
                sequence.append(line)
                remaining_length -= len(line)

        sequence = ''.join(sequence)
        return ProteinSequence(sequence, id=id)

    def __iter__(self) -> Iterable[ProteinSequence]:
        """
        Iterate over all sequences in the file.

        Yields:
            ProteinSequence: Each protein sequence in the file.
        """
        with open(self.file_path, 'r') as f:
            for id, seq in read_fasta(f):
                yield ProteinSequence(seq, id=id)

    @property
    def aligned(self) -> bool:
        """
        Check if all sequences are of equal length (including gaps).

        Returns:
            bool: True if all sequences have the same length, False otherwise.
        """
        return self._aligned

    @property
    def fixed_length(self) -> bool:
        """
        Check if all contained sequences have the same base length (excluding gaps).

        Returns:
            bool: True if all sequences have the same base length, False otherwise.
        """
        return self._fixed_length

    @property
    def width(self) -> Optional[int]:
        """
        Get the length of the sequences if aligned.

        Returns:
            Optional[int]: The length of the sequences if aligned, None otherwise.
        """
        return self._width

    @property
    def has_gaps(self) -> bool:
        """
        Check if any sequences have gaps.

        Returns:
            bool: True if any sequence has gaps, False otherwise.
        """
        return self._has_gaps

    @property
    def mutated_positions(self) -> Optional[List[int]]:
        """
        List columns that have more than one character, assuming sequences are aligned.

        Returns:
            Optional[List[int]]: List of mutated positions if aligned, None otherwise.
        """
        if not self.aligned:
            return None
        positions: List[set] = [set() for _ in range(self.width)]
        for seq in self:
            for i, char in enumerate(str(seq)):
                positions[i].add(char)
        return [i for i, chars in enumerate(positions) if len(chars) > 1]
    
    @property
    def ids(self) -> List[str]:
        """Get a list of sequence IDs."""
        return list(self._index.keys())

    def to_dict(self) -> Dict[str, str]:
        """
        Convert sequences to a dictionary.

        Returns:
            Dict[str, str]: A dictionary with sequence IDs as keys and sequences as values.
        """
        return {id: str(self[id]) for id in self._index}

    def to_fasta(self, output_path: str) -> None:
        """
        Write sequences to a FASTA file.

        Args:
            output_path (str): The path to the output FASTA file.
        """
        with open(output_path, 'w') as f:
            write_fasta(((id, str(self[id])) for id in self._index), f)

    @classmethod
    def from_fasta(cls, input_path: str) -> 'ProteinSequencesOnFile':
        """
        Create a ProteinSequencesOnFile object from a FASTA file.

        Args:
            input_path (str): The path to the input FASTA file.

        Returns:
            ProteinSequencesOnFile: A new ProteinSequencesOnFile object.
        """
        return cls(input_path)
    
    @classmethod
    def from_dict(cls, sequences: Dict[str, str]) -> ProteinSequences:
        return super().from_dict(sequences)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the ProteinSequencesOnFile object.

        Returns:
            str: A string representation of the object.
        """
        return f"ProteinSequencesOnFile(file='{self.file_path}', count={len(self)})"
    
    def to_memory(self) -> ProteinSequences:
        """
        Load all sequences into memory as a ProteinSequences object.

        Returns:
            ProteinSequences: A new ProteinSequences object containing all sequences.
        """
        return ProteinSequences(list(self))
    
    def iter_batches(self, batch_size: int) -> Iterable[ProteinSequences]:
        """
        Iterate over batches of sequences.

        Args:
            batch_size (int): The size of each batch.

        Yields:
            ProteinSequences: A batch of sequences.
        """
        for i in range(0, len(self), batch_size):
            yield ProteinSequences([self[id] for id in list(self._index.keys())[i:i+batch_size]])
    

    
