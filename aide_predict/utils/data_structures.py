# aide_predict/data_structures.py
'''
* Author: Evan Komp
* Created: 6/21/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Base data structures for the AIDE Predict package Where they do not exist in sklearn.
'''
from dataclasses import dataclass
from collections.abc import MutableSequence
import shutil
import tempfile
import os

from aide_predict.io.bio_files import read_fasta, write_fasta

from .constants import AA_SINGLE, GAP_CHARACTERS, NON_CONONICAL_AA_SINGLE

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True)
class ProteinCharacter(str):
    character: str

    def __new__(cls, character: str):
        return str.__new__(cls, character)

    def __post_init__(self):
        if self.character not in AA_SINGLE.union(GAP_CHARACTERS).union(NON_CONONICAL_AA_SINGLE):
            raise ValueError(f"Invalid character {self.character} for protein sequence.")
        
    @property
    def is_gap(self):
        return self.character in GAP_CHARACTERS

    @property
    def is_non_canonical(self):
        return self.character in NON_CONONICAL_AA_SINGLE


class ProteinSequence(str, MutableSequence):

    def __new__(cls, sequence: str):
        obj = str.__new__(cls, sequence)
        characters = [ProteinCharacter(c) for c in sequence]
        obj._sequence = characters
        return obj

    def __getitem__(self, index):
        return self._sequence[index]
    
    def __setitem__(self, index, value):
        self._sequence[index] = value

    def __delitem__(self, index):
        del self._sequence[index]

    def __len__(self):
        return len(self._sequence)
    
    def insert(self, index, value):
        self._sequence.insert(index, value)

    def __repr__(self):
        return str(self.__class__) + f"{(c.character for c in self._sequence)}"
    
    @property
    def has_gaps(self):
        return any(c.is_gap for c in self._sequence)
    
    @property
    def has_non_canonical(self):
        return any(c.is_non_canonical for c in self._sequence)

    @property
    def with_no_gaps(self):
        return "".join(c.character for c in self._sequence if not c.is_gap)
    
    def __hash__(self):
        return hash(self.with_no_gaps)
    
    def __eq__(self, other):
        if isinstance(other, ProteinSequence):
            return self.with_no_gaps == other.with_no_gaps and self.width == other.width
        else:
            return False
    
class ProteinSequences(MutableSequence):

    def __init__(self, sequences: list):
        self._sequences = [ProteinSequence(s) for s in sequences]

    def __getitem__(self, index):
        return self._sequences[index]
    
    def __setitem__(self, index, value):
        self._sequences[index] = value

    def __delitem__(self, index):
        del self._sequences[index]

    def __len__(self):
        return len(self._sequences)
    
    def insert(self, index, value):
        self._sequences.insert(index, value)

    @property
    def aligned(self):
        return all(len(s) == len(self._sequences[0]) for s in self._sequences)
    
    @property
    def has_gaps(self):
        return any(s.has_gaps for s in self._sequences)
    
    @property
    def has_non_canonical(self):
        return any(s.has_non_canonical for s in self._sequences)
    
    @property
    def width(self):
        if self.aligned:
            return len(self._sequences[0])
        else:
            return None
        
    @property
    def num_sequences(self):
        return len(self._sequences)
    
    @property
    def mutated_positions(self):
        positions = []
        if self.aligned:
            for i in range(self.width):
                unique_characters = set(s[i] for s in self._sequences)
                if len(unique_characters) > 1:
                    positions.append(i)
        return positions


    def __repr__(self):
        return str(self.__class__) + f"(num_sequences={self.num_sequences}, aligned={self.aligned}, has_gaps={self.has_gaps}, has_non_canonical={self.has_non_canonical}, width={self.width})"
    
    def to_dict(self):
        return {hash(s): s for s in self._sequences}
    
    def to_fasta(self, outfile: str):
        as_dict = self.to_dict()
        sequences = zip(*as_dict.items())
        with open(outfile, "w") as f:
            write_fasta(sequences, f)

    @classmethod
    def from_fasta(cls, infile: str):
        sequences = []
        with open(infile, "r") as f:
            for header, seq in read_fasta(f):
                sequences.append(seq)
        return cls(sequences)
    
    def align_all(self, to_outfile: str = None):
        from aide_predict.utils.alignment_calls import mafft
        if self.aligned:
            return self
        else:
            return mafft(self, to_outfile=to_outfile)
        
    def align_with(self, existing_alignment, realign: bool = False, to_outfile: str = None):
        from aide_predict.utils.alignment_calls import mafft
        return mafft(self, existing_alignment, realign=realign, to_outfile=to_outfile)
    
class ProteinSequencesOnFile(ProteinSequences):

    def __init__(self, fasta_file: str):
        self.fasta_file = fasta_file

    @property
    def _sequences(self):
        for _, seq in read_fasta(self.fasta_file):
            yield ProteinSequence(seq)

    def __len__(self):
        counter = 0
        for _ in self._sequences:
            counter += 1
        return counter
    
    def __getitem__(self, index):
        for i, seq in enumerate(self._sequences):
            if i == index:
                return seq
        raise IndexError("Index out of range")
    
    def __setitem__(self, index, value):
        raise NotImplementedError("Cannot set values on file.")
    
    def __delitem__(self, index):
        raise NotImplementedError("Cannot delete values on file.")
    
    def insert(self, index, value):
        raise NotImplementedError("Cannot insert values on file.")
    
    @property
    def aligned(self):
        lengths = set(len(s) for s in self._sequences)
        return len(lengths) == 1
    
    
    @property
    def width(self):
        if self.aligned:
            return len(next(self._sequences))
        else:
            return None
        
    def to_dict(self):
        return super().to_dict()
    
    def to_fasta(self, outfile: str):
        if outfile == self.fasta_file:
            pass
        with open(outfile, "w") as f:
            for seq in self._sequences:
                write_fasta(seq, f)

    @classmethod
    def from_fasta(cls, infile: str):
        return cls(infile)
    
    def align_all(self, to_outfile: str = None):
        from aide_predict.utils.alignment_calls import mafft
        if self.aligned:
            if to_outfile is not None:
                shutil.copy(self.fasta_file, to_outfile)
                return ProteinSequencesOnFile(to_outfile)
            else:
                return self
        else:
            # if outfile passed, align to that location and consstruct self from that file
            # otherwise align to a temporary file, copy the results to self.fasta_file and return self
            if to_outfile is not None:
                mafft(self, to_outfile=to_outfile)
                return ProteinSequencesOnFile(to_outfile)
            else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file = os.path.join(temp_dir, 'temp.fasta')
                    mafft(self, to_outfile=temp_file)
                    shutil.copy(temp_file, self.fasta_file)
                    return self
        
    def align_with(self, existing_alignment, realign: bool = False, to_outfile: str = None):
        from aide_predict.utils.alignment_calls import mafft
        return mafft(self, existing_alignment, realign=realign, to_outfile=to_outfile)
    
    def to_memory(self):
        return ProteinSequences([s for s in self._sequences])


        
    

    