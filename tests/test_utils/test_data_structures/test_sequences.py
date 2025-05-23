# tests/utils/data_structures.py
'''
* Author: Evan Komp
* Created: 6/26/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

We should be able to handle on file and in memory protein sequences, both aligned and unaligned.
'''

import pytest
from tempfile import NamedTemporaryFile
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from aide_predict.utils.data_structures import ProteinCharacter, ProteinSequence, ProteinSequences, ProteinSequencesOnFile
from aide_predict.utils.constants import AA_SINGLE, GAP_CHARACTERS, NON_CONONICAL_AA_SINGLE
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom

# Test data
VALID_AA = AA_SINGLE
INVALID_AA = "2"
GAP_CHAR = "-"
    

class TestProteinCharacter:
    def test_valid_initialization(self):
        for aa in VALID_AA:
            pc = ProteinCharacter(aa)
            assert pc == aa

    def test_invalid_initialization(self):
        with pytest.raises(ValueError):
            ProteinCharacter(INVALID_AA)

    def test_gap_character(self):
        pc = ProteinCharacter(GAP_CHAR)
        assert pc.is_gap

    def test_non_canonical(self):
        pc = ProteinCharacter("X")
        assert pc.is_non_canonical

    def test_equality(self):
        pc1 = ProteinCharacter("A")
        pc2 = ProteinCharacter("A")
        assert pc1 == pc2
        assert pc1 == "A"

    def test_hash(self):
        pc = ProteinCharacter("A")
        assert hash(pc) == hash("A")

    def test_is_not_focus(self):
        assert ProteinCharacter('-').is_not_focus
        assert ProteinCharacter('a').is_not_focus
        assert not ProteinCharacter('A').is_not_focus

class TestProteinSequence:
    @pytest.fixture(scope="class")
    def temp_pdb_file(self, tmp_path_factory):
        # Create a temporary directory
        temp_dir = tmp_path_factory.mktemp("pdb_test")
        
        # Create PDB content
        pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  
ATOM      3  C   ALA A   1       2.009   1.362   0.000  1.00  0.00           C  
ATOM      4  O   ALA A   1       1.702   2.144   0.907  1.00  0.00           O  
ATOM      5  N   CYS A   2       2.831   1.687  -0.987  1.00  0.00           N  
ATOM      6  CA  CYS A   2       3.396   3.037  -1.009  1.00  0.00           C  
ATOM      7  C   CYS A   2       2.362   4.089  -1.408  1.00  0.00           C  
ATOM      8  O   CYS A   2       2.730   5.261  -1.509  1.00  0.00           O  
ATOM      9  N   ASP A   3       1.103   3.682  -1.556  1.00  0.00           N  
ATOM     10  CA  ASP A   3       0.042   4.619  -1.936  1.00  0.00           C  
ATOM     11  C   ASP A   3      -0.914   4.931  -0.778  1.00  0.00           C  
ATOM     12  O   ASP A   3      -1.725   5.853  -0.870  1.00  0.00           O  
ATOM     13  N   GLU A   4      -0.826   4.180   0.327  1.00  0.00           N  
ATOM     14  CA  GLU A   4      -1.692   4.399   1.480  1.00  0.00           C  
ATOM     15  C   GLU A   4      -1.138   5.493   2.395  1.00  0.00           C  
ATOM     16  O   GLU A   4      -1.891   6.161   3.102  1.00  0.00           O  
ATOM     17  N   PHE A   5       0.183   5.637   2.410  1.00  0.00           N  
ATOM     18  CA  PHE A   5       0.827   6.661   3.240  1.00  0.00           C  
ATOM     19  C   PHE A   5       0.450   8.077   2.809  1.00  0.00           C  
ATOM     20  O   PHE A   5       0.313   8.973   3.650  1.00  0.00           O  
ATOM     21  N   GLY A   6       0.280   8.272   1.503  1.00  0.00           N  
ATOM     22  CA  GLY A   6      -0.071   9.598   1.000  1.00  0.00           C  
ATOM     23  C   GLY A   6      -1.530   9.928   1.220  1.00  0.00           C  
ATOM     24  O   GLY A   6      -1.862  11.094   1.413  1.00  0.00           O  
ATOM     25  N   HIS A   7      -2.407   8.925   1.194  1.00  0.00           N  
ATOM     26  CA  HIS A   7      -3.837   9.133   1.397  1.00  0.00           C  
ATOM     27  C   HIS A   7      -4.097   9.653   2.811  1.00  0.00           C  
ATOM     28  O   HIS A   7      -4.826  10.627   3.001  1.00  0.00           O  
ATOM     29  N   ILE A   8      -3.504   8.991   3.801  1.00  0.00           N  
ATOM     30  CA  ILE A   8      -3.683   9.403   5.192  1.00  0.00           C  
ATOM     31  C   ILE A   8      -3.017  10.748   5.472  1.00  0.00           C  
ATOM     32  O   ILE A   8      -3.550  11.568   6.228  1.00  0.00           O  
ATOM     33  N   LYS A   9      -1.856  10.986   4.853  1.00  0.00           N  
ATOM     34  CA  LYS A   9      -1.133  12.243   5.043  1.00  0.00           C  
ATOM     35  C   LYS A   9      -1.961  13.430   4.573  1.00  0.00           C  
ATOM     36  O   LYS A   9      -2.047  14.452   5.259  1.00  0.00           O  
ATOM     37  N   LEU A  10      -2.579  13.280   3.403  1.00  0.00           N  
ATOM     38  CA  LEU A  10      -3.399  14.352   2.859  1.00  0.00           C  
ATOM     39  C   LEU A  10      -4.635  14.610   3.722  1.00  0.00           C  
ATOM     40  O   LEU A  10      -4.940  15.770   4.012  1.00  0.00           O  
ATOM     41  N   MET A  11      -5.342  13.551   4.124  1.00  0.00           N  
ATOM     42  CA  MET A  11      -6.541  13.692   4.954  1.00  0.00           C  
ATOM     43  C   MET A  11      -6.180  14.281   6.316  1.00  0.00           C  
ATOM     44  O   MET A  11      -6.859  15.193   6.796  1.00  0.00           O  
ATOM     45  N   ASN A  12      -5.107  13.780   6.923  1.00  0.00           N  
ATOM     46  CA  ASN A  12      -4.667  14.268   8.230  1.00  0.00           C  
ATOM     47  C   ASN A  12      -4.230  15.728   8.165  1.00  0.00           C  
ATOM     48  O   ASN A  12      -4.582  16.542   9.019  1.00  0.00           O  
ATOM     49  N   PRO A  13      -3.437  16.075   7.152  1.00  0.00           N  
ATOM     50  CA  PRO A  13      -2.968  17.458   7.008  1.00  0.00           C  
ATOM     51  C   PRO A  13      -4.092  18.425   6.651  1.00  0.00           C  
ATOM     52  O   PRO A  13      -4.062  19.589   7.052  1.00  0.00           O  
ATOM     53  N   GLN A  14      -5.092  17.920   5.919  1.00  0.00           N  
ATOM     54  CA  GLN A  14      -6.223  18.760   5.524  1.00  0.00           C  
ATOM     55  C   GLN A  14      -7.109  19.165   6.705  1.00  0.00           C  
ATOM     56  O   GLN A  14      -7.466  20.340   6.837  1.00  0.00           O  
ATOM     57  N   ARG A  15      -7.467  18.205   7.569  1.00  0.00           N  
ATOM     58  CA  ARG A  15      -8.308  18.491   8.731  1.00  0.00           C  
ATOM     59  C   ARG A  15      -7.568  19.318   9.779  1.00  0.00           C  
ATOM     60  O   ARG A  15      -8.169  20.205  10.388  1.00  0.00           O  
ATOM     61  N   SER A  16      -6.271  19.056   9.953  1.00  0.00           N  
ATOM     62  CA  SER A  16      -5.463  19.792  10.929  1.00  0.00           C  
ATOM     63  C   SER A  16      -5.350  21.273  10.576  1.00  0.00           C  
ATOM     64  O   SER A  16      -5.504  22.146  11.433  1.00  0.00           O  
ATOM     65  N   THR A  17      -5.068  21.550   9.302  1.00  0.00           N  
ATOM     66  CA  THR A  17      -4.937  22.938   8.860  1.00  0.00           C  
ATOM     67  C   THR A  17      -6.265  23.564   8.452  1.00  0.00           C  
ATOM     68  O   THR A  17      -6.340  24.787   8.364  1.00  0.00           O  
ATOM     69  N   VAL A  18      -7.295  22.743   8.185  1.00  0.00           N  
ATOM     70  CA  VAL A  18      -8.616  23.239   7.786  1.00  0.00           C  
ATOM     71  C   VAL A  18      -9.386  23.824   8.967  1.00  0.00           C  
ATOM     72  O   VAL A  18      -9.913  24.932   8.876  1.00  0.00           O  
ATOM     73  N   TRP A  19      -9.462  23.085  10.077  1.00  0.00           N  
ATOM     74  CA  TRP A  19     -10.174  23.551  11.267  1.00  0.00           C  
ATOM     75  C   TRP A  19      -9.459  24.729  11.926  1.00  0.00           C  
ATOM     76  O   TRP A  19     -10.094  25.726  12.287  1.00  0.00           O  
ATOM     77  N   TYR A  20      -8.144  24.625  12.110  1.00  0.00           N  
ATOM     78  CA  TYR A  20      -7.365  25.700  12.730  1.00  0.00           C  
ATOM     79  C   TYR A  20      -7.292  26.939  11.843  1.00  0.00           C  
ATOM     80  O   TYR A  20      -7.448  28.068  12.301  1.00  0.00           O  
TER      81      TYR A  20
END
"""
        
        # Create a PDB file
        pdb_path = temp_dir / "sample.pdb"
        pdb_path.write_text(pdb_content)
        
        return pdb_path

    @pytest.fixture
    def sample_sequence(self, temp_pdb_file):
        return ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="sample", structure=str(temp_pdb_file))
    
    def test_as_array(self, sample_sequence):
        arr = sample_sequence.as_array
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, 20)
        assert ''.join(arr[0]) == str(sample_sequence)

    def test_initialization(self, sample_sequence, temp_pdb_file):
        assert str(sample_sequence) == "ACDEFGHIKLMNPQRSTVWY"
        assert sample_sequence.id == "sample"
        assert sample_sequence.structure.pdb_file == str(temp_pdb_file)
    
    def test_structure_setter_invalid_file(self):
        seq = ProteinSequence("ACDE")
        with pytest.raises(FileNotFoundError):
            seq.structure = "non_existent_file.pdb"

    def test_structure_setter_invalid_structure(self, temp_pdb_file):
        # should raise warning
        with pytest.warns(Warning):
            seq = ProteinSequence("ACDE", structure=str(temp_pdb_file))
    
    def test_properties(self, sample_sequence):
        assert not sample_sequence.has_gaps
        assert not sample_sequence.has_non_canonical
        assert sample_sequence.num_gaps == 0
        assert sample_sequence.base_length == 20

        gapped = ProteinSequence("ACD-")
        assert gapped.has_gaps
        assert gapped.num_gaps == 1
        assert gapped.base_length == 3
        assert gapped.with_no_gaps() == ProteinSequence("ACD")


    def test__mutate(self, sample_sequence):
        mutated = sample_sequence._mutate(0, "M")
        assert str(mutated) == "MCDEFGHIKLMNPQRSTVWY"
        assert mutated.id is None  # id should not be passed to indicate mutation

    def test_mutate(self, sample_sequence):
        mutated = sample_sequence.mutate("A0M", one_indexed=False)
        assert str(mutated) == "MCDEFGHIKLMNPQRSTVWY"
        assert mutated.id is None  # id should not be passed to indicate mutation

    def test_mutated_positions(self, sample_sequence):
        other = ProteinSequence("MCDEFGHIKLMNPQRSTVWY")
        assert sample_sequence.mutated_positions(other) == [0]

    def test_get_protein_character(self, sample_sequence):
        assert isinstance(sample_sequence.get_protein_character(0), ProteinCharacter)
        assert sample_sequence.get_protein_character(0) == "A"

    def test_slice(self, sample_sequence):
        sliced = sample_sequence.slice_as_protein_sequence(0, 5)
        assert str(sliced) == "ACDEF"
        assert isinstance(sliced, ProteinSequence)

    def test_from_pdb(self, temp_pdb_file):
        # Test successful creation
        seq = ProteinSequence.from_pdb(str(temp_pdb_file))
        assert isinstance(seq, ProteinSequence)
        assert str(seq) == "ACDEFGHIKLMNPQRSTVWY"  # The sequence in our test PDB
        assert seq.id == "sample"  # From basename of temp_pdb_file
        assert seq.structure.pdb_file == str(temp_pdb_file)
        assert seq.structure.chain == "A"
        
        # Test with custom chain and ID
        seq = ProteinSequence.from_pdb(str(temp_pdb_file), chain="A", id="my_protein")
        assert seq.id == "my_protein"
        
        # Test with invalid PDB file
        with pytest.raises(FileNotFoundError):
            ProteinSequence.from_pdb("nonexistent.pdb")
            

    def test_iter_protein_characters(self, sample_sequence):
        chars = list(sample_sequence.iter_protein_characters())
        assert all(isinstance(c, ProteinCharacter) for c in chars)
        assert "".join(str(c) for c in chars) == str(sample_sequence)

    def test_align(self, sample_sequence):
        # Create another sequence to align with
        other_sequence = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="other")
        
        # Introduce a gap in the middle of other_sequence
        other_sequence = ProteinSequence("ACDEFGHIKMNPQRSTVWY", id="other")
        
        # Align the sequences
        aligned_self, aligned_other = sample_sequence.align(other_sequence)
        
        # Check that the aligned sequences are ProteinSequence objects
        assert isinstance(aligned_self, ProteinSequence)
        assert isinstance(aligned_other, ProteinSequence)
        
        # Check that the aligned sequences have the same length
        assert len(aligned_self) == len(aligned_other)
        
        # Check that the original IDs are preserved
        assert aligned_self.id == "sample"
        assert aligned_other.id == "other"
        
        # Check that the alignment introduced gaps in the right places
        assert "-" in str(aligned_other)
        assert str(aligned_other) == "ACDEFGHIK-MNPQRSTVWY"
        
        # Check that the non-gap parts of the sequences match the originals
        assert aligned_self.with_no_gaps() == sample_sequence
        assert aligned_other.with_no_gaps() == other_sequence.with_no_gaps()
        
        # Check that the structures are preserved
        assert aligned_self.structure == sample_sequence.structure
        assert aligned_other.structure == other_sequence.structure

    def test_hash(self, sample_sequence):
        assert isinstance(hash(sample_sequence), int)

    def test_equality(self, sample_sequence):
        same_sequence = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="sample")
        same_sequence_same_structure = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="sample", structure=sample_sequence.structure)
        different_sequence = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="different")
        assert sample_sequence != same_sequence # because of different structure
        assert sample_sequence != different_sequence
        assert sample_sequence == same_sequence_same_structure

    def test_get_mutations(self, sample_sequence):
        other = ProteinSequence("MCDEFGHIKLMNPQRSTVWY")
        mutations = sample_sequence.get_mutations(other)
        assert mutations == [f"A1M"]

    def test_saturation_mutagenesis(self, sample_sequence):
        mutants = sample_sequence.saturation_mutagenesis()
        assert len(mutants) == 380 # 20 * 20 - 20 (no mutation) = 380
        assert all(isinstance(mutant, ProteinSequence) for mutant in mutants)
        assert len(mutants.mutated_positions) == 20

    def test_msa_property(self, sample_sequence):
        """Test setting and getting MSA property on a ProteinSequence."""
        # Create a test MSA
        msa = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq1"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq2"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq3")
        ])
        
        # Set MSA on the sequence
        sample_sequence.msa = msa
        
        # Check that the MSA is set correctly
        assert sample_sequence.msa is msa
        assert sample_sequence.has_msa
        
        # Check that we can't set a non-aligned MSA
        non_aligned_msa = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACD", id="seq2")
        ])
        with pytest.raises(ValueError):
            sample_sequence.msa = non_aligned_msa
        
        # Check that we can't set a non-ProteinSequences object
        with pytest.raises(ValueError):
            sample_sequence.msa = "not a ProteinSequences object"

    def test_msa_same_width(self, sample_sequence):
        """Test msa_same_width property."""
        # Create MSAs of different widths
        same_width_msa = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq1"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq2")
        ])
        
        different_width_msa = ProteinSequences([
            ProteinSequence("ACDEFGHIKLM", id="seq1"),
            ProteinSequence("ACDEFGHIKLM", id="seq2")
        ])
        
        # Set MSA with same width
        sample_sequence.msa = same_width_msa
        assert sample_sequence.msa_same_width
        
        # Set MSA with different width
        sample_sequence.msa = different_width_msa
        assert not sample_sequence.msa_same_width
        
        # No MSA
        sample_sequence._msa = None
        assert not sample_sequence.msa_same_width

    def test_is_in_msa(self, sample_sequence):
        """Test is_in_msa property."""
        # Create MSA that includes the sequence
        including_msa = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq1"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq2"),
        ])
        
        # Create MSA that doesn't include the sequence
        excluding_msa = ProteinSequences([
            ProteinSequence("MCDEFGHIKLMNPQRSTVWY", id="seq1"),
            ProteinSequence("MCDEFGHIKLMNPQRSTVWY", id="seq2"),
        ])
        
        # Create MSA with gaps that includes the sequence
        gapped_msa = ProteinSequences([
            ProteinSequence("ACDE-FGHIKLMNPQRSTVWY", id="seq1"),
            ProteinSequence("ACDE-FGHIKLMNPQRSTVWY", id="seq2"),
        ])
        
        # Test when sequence is in MSA
        sample_sequence.msa = including_msa
        assert sample_sequence.is_in_msa
        
        # Test when sequence is not in MSA
        sample_sequence.msa = excluding_msa
        assert not sample_sequence.is_in_msa
        
        # Test when sequence is in MSA but MSA has gaps
        sample_sequence.msa = gapped_msa
        assert sample_sequence.is_in_msa
        
        # No MSA
        sample_sequence._msa = None
        assert not sample_sequence.is_in_msa

    def test_hash_with_msa(self, sample_sequence):
        """Test that hash includes MSA."""
        # Hash without MSA
        hash_without_msa = hash(sample_sequence)
        
        # Add MSA
        msa = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq1"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq2"),
        ])
        sample_sequence.msa = msa
        
        # Hash with MSA
        hash_with_msa = hash(sample_sequence)
        
        # Hashes should be different
        assert hash_without_msa != hash_with_msa
        
        # Different MSAs should yield different hashes
        different_msa = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq1"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq2"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq3"),
        ])
        sample_sequence.msa = different_msa
        hash_with_different_msa = hash(sample_sequence)
        assert hash_with_msa != hash_with_different_msa

    def test_from_fasta(self, tmp_path):
        """Test creating a ProteinSequence from a FASTA file with concrete files."""
        # Create test FASTA file
        fasta_content = ">seq1\nACDE\n>seq2\nACDF\n>seq3\nACDG\n"
        fasta_path = tmp_path / "test.fasta"
        fasta_path.write_text(fasta_content)
        
        # Load sequence with MSA
        seq = ProteinSequence.from_fasta(str(fasta_path))
        
        # Verify sequence and MSA
        assert str(seq) == "ACDE"
        assert seq.id == "seq1"
        assert seq.has_msa
        assert len(seq.msa) == 3
        assert str(seq.msa[0]) == "ACDE"
        assert str(seq.msa[1]) == "ACDF"
        assert str(seq.msa[2]) == "ACDG"
        
        # Verify sequence without gaps is returned
        fasta_content_with_gaps = ">seq1\nAC-DE\n>seq2\nAC-DF\n>seq3\nAC-DG\n"
        fasta_gap_path = tmp_path / "test_gaps.fasta"
        fasta_gap_path.write_text(fasta_content_with_gaps)
        
        seq_from_gapped = ProteinSequence.from_fasta(str(fasta_gap_path))
        assert str(seq_from_gapped) == "ACDE"  # Gaps removed in self_seq
        assert seq_from_gapped.has_msa
        assert str(seq_from_gapped.msa[0]) == "AC-DE"  # MSA preserves gaps

    def test_from_a3m(self, tmp_path):
        """Test creating a ProteinSequence from an A3M file with concrete files."""
        # Create test A3M file
        a3m_content = ">seq1\nACDE\n>seq2\nACDF\n>seq3\nACDG\n"
        a3m_path = tmp_path / "test.a3m"
        a3m_path.write_text(a3m_content)
        
        # Load sequence with MSA
        seq = ProteinSequence.from_a3m(str(a3m_path))
        
        # Verify sequence and MSA
        assert str(seq) == "ACDE"
        assert seq.id == "seq1"
        assert seq.has_msa
        assert len(seq.msa) == 3
        assert str(seq.msa[0]) == "ACDE"
        assert str(seq.msa[1]) == "ACDF"
        assert str(seq.msa[2]) == "ACDG"
        
        # Test inserts parameter
        seq_first_inserts = ProteinSequence.from_a3m(str(a3m_path), inserts='first')
        assert seq_first_inserts.has_msa

# ProteinSequences tests
class TestProteinSequences:
    @pytest.fixture
    def sample_sequences(self):
        return ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDF", id="seq2"),
            ProteinSequence("ACD-", id="seq3")
        ])

    def test_add_weights(self):
        sample_sequences = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDF", id="seq2"),
            ProteinSequence("ACD-", id="seq3")
        ], weights=[1, 2, 3])
        assert sample_sequences.weights.shape == (3,)

        # bad weights length
        with pytest.raises(ValueError):
            ProteinSequences([
                ProteinSequence("ACDE", id="seq1"),
                ProteinSequence("ACDF", id="seq2"),
                ProteinSequence("ACD-", id="seq3")
            ], weights=[1, 2])


    def test_properties(self, sample_sequences):
        assert sample_sequences.aligned
        assert not sample_sequences.fixed_length
        assert sample_sequences.width == 4
        assert sample_sequences.has_gaps

    def test_mutated_positions(self, sample_sequences):
        assert sample_sequences.mutated_positions == [3]

        # try without aligned sequences
        sample_sequences = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDF", id="seq2"),
            ProteinSequence("ACD", id="seq3")
        ])
        assert sample_sequences.mutated_positions == None

    def test___getitem__(self, sample_sequences):
        assert str(sample_sequences[0]) == "ACDE"
        assert str(sample_sequences["seq2"]) == "ACDF"
        with pytest.raises(IndexError):
            sample_sequences[3]
        with pytest.raises(KeyError):
            sample_sequences["nonexistent"]

        assert len(sample_sequences[:2]) == 2
        assert len(sample_sequences[np.array([0,1])]) == 2

    def test_to_dict(self, sample_sequences):
        d = sample_sequences.to_dict()
        assert d == {"seq1": "ACDE", "seq2": "ACDF", "seq3": "ACD-"}

    def test_to_fasta(self, sample_sequences):
        with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            sample_sequences.to_fasta(temp_file.name)
            temp_file.seek(0)
            content = temp_file.read()
        assert content == ">seq1\nACDE\n>seq2\nACDF\n>seq3\nACD-\n"
        os.unlink(temp_file.name)

    def test_from_fasta(self):
        fasta_content = ">seq1\nACDE\n>seq2\nACDF\n>seq3\nACD-\n"
        with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(fasta_content)
            temp_file.flush()
            sequences = ProteinSequences.from_fasta(temp_file.name)
        assert len(sequences) == 3
        assert str(sequences[0]) == "ACDE"
        assert sequences[0].id == "seq1"
        os.unlink(temp_file.name)

    def test_from_dict(self):
        d = {"seq1": "ACDE", "seq2": "ACDF", "seq3": "ACD-"}
        sequences = ProteinSequences.from_dict(d)
        assert len(sequences) == 3
        assert str(sequences[0]) == "ACDE"
        assert sequences[0].id == "seq1"

    def test_to_on_file(self, sample_sequences, tmp_path):
        output_path = tmp_path / "test.fasta"
        on_file = sample_sequences.to_on_file(str(output_path))
        assert isinstance(on_file, ProteinSequencesOnFile)
        assert on_file.file_path == str(output_path)

    def test_with_no_gaps(self, sample_sequences):
        no_gaps = sample_sequences.with_no_gaps()
        assert isinstance(no_gaps, ProteinSequences)
        assert len(no_gaps) == 3
        assert str(no_gaps[2]) == "ACD"

    def test_align_to(self, sample_sequences):
        sample_sequences = sample_sequences[:2]
        others = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="other1"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="other2")
        ])
        aligned = sample_sequences.align_to(others, realign=False, return_only_new=False)
        assert aligned.width == 20
        assert aligned[0].id == "other1"
        assert len(aligned) == 4
        
        # only new
        aligned = sample_sequences.align_to(others, realign=False, return_only_new=True)
        assert aligned.width == 20
        assert len(aligned) == 2

    def test_align_all(self):
        sample_sequences = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACD", id="seq2"),])
        aligned = sample_sequences.align_all()
        assert aligned.width == 4 

    def test_get_alignment_mapping(self, sample_sequences):
        sample_sequences.append(ProteinSequence("A-DE", id="seq4"))
        mapping = sample_sequences.get_alignment_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) == 4
        
        # Assuming sample_sequences have IDs, otherwise we'll use hash
        seq4_id = "seq4"
        assert mapping[seq4_id] == [0, 2, 3]  # this should not include the gap at position 1

    def test_apply_alignment_mapping(self, sample_sequences):
        sample_sequences = sample_sequences.with_no_gaps()
        
        # Get the IDs or hashes of the sequences
        seq_ids = [seq.id if seq.id else str(hash(seq)) for seq in sample_sequences]
        
        mapping = {
            seq_ids[0]: [0, 1, 2, 5],
            seq_ids[1]: [0, 1, 2, 5],
            seq_ids[2]: [0, 1, 5]
        }
        aligned = sample_sequences.apply_alignment_mapping(mapping)
        assert isinstance(aligned, ProteinSequences)
        assert len(aligned) == 3
        assert str(aligned[0]) == "ACD--E"
        assert str(aligned[1]) == "ACD--F"
        assert str(aligned[2]) == "AC---D"

    def test_get_alignment_mapping_with_mixed_ids(self):
        sequences = ProteinSequences([
            ProteinSequence("ACD-E", id="seq1"),
            ProteinSequence("ACD-F", id="seq2"),
            ProteinSequence("AC--D")  # No ID provided
        ])
        mapping = sequences.get_alignment_mapping()
        assert len(mapping) == 3
        assert mapping["seq1"] == [0, 1, 2, 4]
        assert mapping["seq2"] == [0, 1, 2, 4]
        assert str(hash(sequences[2])) in mapping
        assert mapping[str(hash(sequences[2]))] == [0, 1, 4]

    def test_apply_alignment_mapping_with_mixed_ids(self):
        sequences = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDF", id="seq2"),
            ProteinSequence("ACD")  # No ID provided
        ])
        seq3_hash = str(hash(sequences[2]))
        mapping = {
            "seq1": [0, 1, 2, 5],
            "seq2": [0, 1, 2, 5],
            seq3_hash: [0, 1, 5]
        }
        aligned = sequences.apply_alignment_mapping(mapping)
        assert len(aligned) == 3
        assert str(aligned[0]) == "ACD--E"
        assert str(aligned[1]) == "ACD--F"
        assert str(aligned[2]) == "AC---D"
        assert aligned[2].id is None  # Ensure the ID (or lack thereof) is preserved

    def test_apply_alignment_mapping_error(self, sample_sequences):
        sample_sequences = sample_sequences.with_no_gaps()
        invalid_mapping = {"nonexistent_id": [0, 1, 2]}
        with pytest.raises(ValueError):
            sample_sequences.apply_alignment_mapping(invalid_mapping)

    def test_as_array(self, sample_sequences):
        arr = sample_sequences.as_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 4)
        assert ''.join(arr[0]) == "ACDE"

    def test_iter_batches(self, sample_sequences):
        batches = list(sample_sequences.iter_batches(2))
        assert len(batches) == 2
        assert isinstance(batches[0], ProteinSequences)
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    def test_has_lower(self, sample_sequences):
        assert not sample_sequences.has_lower()
        sample_sequences.append(ProteinSequence("acde", id="seq4"))
        assert sample_sequences.has_lower()

        # try without aligned sequences
        sample_sequences = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDF", id="seq2"),
            ProteinSequence("ACD", id="seq3")
        ])
        assert not sample_sequences.has_lower()

    def test_msa_process(self, sample_sequences):
        with patch("aide_predict.utils.msa.MSAProcessing") as mock_msa_processing:
            sample_sequences.msa_process()
            mock_msa_processing = mock_msa_processing.return_value
            mock_msa_processing.process.assert_called_once()

    def test_sample(self, sample_sequences):
        sample_sequences.weights = np.array([0.0, 0.5, 0.5])
        sampled = sample_sequences.sample(2)
        assert len(sampled) == 2
        assert sample_sequences[0] not in sampled

    def test_from_csv(self, tmp_path):
        # Create a test CSV file
        csv_path = tmp_path / "test.csv"
        csv_content = """sequence,id,activity,solubility
ACDE,seq1,0.5,0.8
ACDF,seq2,0.6,0.7
ACD-,seq3,0.7,0.6"""
        
        with open(csv_path, 'w') as f:
            f.write(csv_content)

        # Test default behavior (first column)
        sequences = ProteinSequences.from_csv(csv_path, seq_col="sequence")
        assert len(sequences) == 3
        assert str(sequences[0]) == "ACDE"

        # Test with ID column
        sequences = ProteinSequences.from_csv(csv_path, seq_col="sequence", id_col="id")
        assert len(sequences) == 3
        assert sequences[0].id == "seq1"
        assert str(sequences["seq2"]) == "ACDF"

        # Test with single label column
        sequences, labels = ProteinSequences.from_csv(
            csv_path, 
            seq_col="sequence",
            id_col="id",
            label_cols="activity"
        )
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (3, 1)
        assert labels[0, 0] == 0.5

        # Test with multiple label columns
        sequences, labels = ProteinSequences.from_csv(
            csv_path,
            seq_col="sequence",
            id_col="id",
            label_cols=["activity", "solubility"]
        )
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (3, 2)
        assert labels[0, 0] == 0.5
        assert labels[0, 1] == 0.8

        # Test pandas kwargs
        sequences = ProteinSequences.from_csv(csv_path, sep=",", skiprows=2, header=None)
        assert len(sequences) == 2

        # Test error with missing columns
        with pytest.raises(ValueError):
            ProteinSequences.from_csv(csv_path, seq_col="nonexistent")
        
        with pytest.raises(ValueError):
            ProteinSequences.from_csv(csv_path, seq_col="sequence", id_col="nonexistent")
        
        with pytest.raises(ValueError):
            ProteinSequences.from_csv(csv_path, seq_col="sequence", label_cols="nonexistent")

    def test_from_df(self):
        # Create test DataFrame
        df = pd.DataFrame({
            'sequence': ['ACDE', 'ACDF', 'ACD-'],
            'id': ['seq1', 'seq2', 'seq3'],
            'activity': [0.5, 0.6, 0.7],
            'solubility': [0.8, 0.7, 0.6]
        })

        # Test default behavior (first column)
        sequences = ProteinSequences.from_df(df, seq_col="sequence")
        assert len(sequences) == 3
        assert str(sequences[0]) == "ACDE"

        # Test with ID column
        sequences = ProteinSequences.from_df(df, seq_col="sequence", id_col="id")
        assert len(sequences) == 3
        assert sequences[0].id == "seq1"
        assert str(sequences["seq2"]) == "ACDF"

        # Test with single label column
        sequences, labels = ProteinSequences.from_df(
            df, 
            seq_col="sequence",
            id_col="id",
            label_cols="activity"
        )
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (3, 1)
        assert labels[0, 0] == 0.5

        # Test with multiple label columns
        sequences, labels = ProteinSequences.from_df(
            df,
            seq_col="sequence",
            id_col="id",
            label_cols=["activity", "solubility"]
        )
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (3, 2)
        assert labels[0, 0] == 0.5
        assert labels[0, 1] == 0.8

        # Test error with missing columns
        with pytest.raises(ValueError):
            ProteinSequences.from_df(df, seq_col="nonexistent")
        
        with pytest.raises(ValueError):
            ProteinSequences.from_df(df, seq_col="sequence", id_col="nonexistent")
        
        with pytest.raises(ValueError):
            ProteinSequences.from_df(df, seq_col="sequence", label_cols="nonexistent")

        # Test error with empty DataFrame
        with pytest.raises(IndexError):
            ProteinSequences.from_df(pd.DataFrame())

    def test_equality_with_msa(self):
        """Test equality comparison when sequences have MSAs attached."""
        # Create a base sequence
        seq1 = ProteinSequence("ACDE", id="seq1")
        seq2 = ProteinSequence("ACDE", id="seq1") 
        
        # Initial equality - should be equal as they have same sequence and ID
        assert seq1 == seq2
        
        # Create an MSA and attach to seq1
        msa = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDE", id="seq2")
        ])
        seq1.msa = msa
        
        # Should no longer be equal since one has MSA and the other doesn't
        assert seq1 != seq2
        
        # Attach the same MSA to seq2
        seq2.msa = msa
        
        # Should be equal again
        assert seq1 == seq2
        
        # Attach a different MSA to seq2
        msa2 = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDE", id="seq2"),
            ProteinSequence("ACDE", id="seq3")
        ])
        seq2.msa = msa2
        
        # Should not be equal with different MSAs
        assert seq1 != seq2

    def test_copy_with_msa(self):
        """Test that MSA is properly handled when copying sequences."""
        # Create sequence with MSA
        seq = ProteinSequence("ACDE", id="seq1")
        msa = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDE", id="seq2")
        ])
        seq.msa = msa
        
        # Create new sequence using the same parameters
        seq2 = ProteinSequence("ACDE", id="seq1", msa=msa)
        
        # Should be equal
        assert seq == seq2
        assert seq.msa == seq2.msa

    def test_protein_sequences_hash_with_msas(self):
        """Test hashing of ProteinSequences when contained sequences have MSAs."""
        # Create sequences without MSAs
        seq1 = ProteinSequence("ACDE", id="seq1")
        seq2 = ProteinSequence("FGHI", id="seq2")
        sequences = ProteinSequences([seq1, seq2])
        
        # Hash without MSAs
        hash_without_msas = hash(sequences)
        
        # Add MSA to seq1
        msa = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDE", id="seq_other")
        ])
        seq1.msa = msa
        
        # Hash should now be different
        hash_with_msa = hash(sequences)
        assert hash_without_msas != hash_with_msa

    def test_sequence_with_msa_mutation_operations(self):
        """Test behavior of mutation operations with MSAs attached."""
        # Create a sequence with MSA
        seq = ProteinSequence("ACDE", id="seq1")
        msa = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDE", id="seq2")
        ])
        seq.msa = msa
        
        # Test mutate - MSA should not be transferred
        mutated = seq.mutate("A1M")
        assert not mutated.has_msa
        
        # Test _mutate - MSA should not be transferred
        mutated = seq._mutate(0, "M")
        assert not mutated.has_msa
        
        # Make sure original seq still has MSA
        assert seq.has_msa

    def test_with_no_gaps_preserves_msa(self):
        """Test that with_no_gaps preserves the MSA reference."""
        # Create sequence with gaps and MSA
        seq = ProteinSequence("AC-DE", id="seq1")
        msa = ProteinSequences([
            ProteinSequence("AC-DE", id="seq1"),
            ProteinSequence("AC-DE", id="seq2")
        ])
        seq.msa = msa
        
        # Remove gaps
        no_gaps = seq.with_no_gaps()
        
        # MSA should still be attached
        assert no_gaps.has_msa
        assert no_gaps.msa == msa

    def test_upper_preserves_msa(self):
        """Test that upper preserves the MSA reference."""
        # Create sequence with lowercase and MSA
        seq = ProteinSequence("acde", id="seq1")
        msa = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDE", id="seq2")
        ])
        seq.msa = msa
        
        # Convert to uppercase
        upper_seq = seq.upper()
        
        # MSA should still be attached
        assert upper_seq.has_msa
        assert upper_seq.msa == msa


# ProteinSequencesOnFile tests
class TestProteinSequencesOnFile:
    @pytest.fixture
    def sample_fasta_file(self):
        content = ">seq1\nACDE\n>seq2\nACDF\n>seq3\nACD-\n"
        with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            yield temp_file.name
        os.unlink(temp_file.name)

    @pytest.fixture
    def multi_line_fasta_file(self):
        content = ">seq1\nACDE\nFGHI\nJKLM\n>seq2\nACDF\n>seq3\nACD-\n"
        with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            yield temp_file.name
        os.unlink(temp_file.name)

    def test_initialization(self, sample_fasta_file):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        assert len(sequences) == 3
        assert sequences.aligned
        assert not sequences.fixed_length
        assert sequences.width == 4
        assert sequences.has_gaps

    def test_getitem(self, sample_fasta_file):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        assert str(sequences[0]) == "ACDE"
        assert str(sequences["seq2"]) == "ACDF"
        with pytest.raises(IndexError):
            sequences[3]
        with pytest.raises(KeyError):
            sequences["nonexistent"]

    def test_iteration(self, sample_fasta_file):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        seqs = list(sequences)
        assert len(seqs) == 3
        assert all(isinstance(seq, ProteinSequence) for seq in seqs)

    def test_mutated_positions(self, sample_fasta_file):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        assert sequences.mutated_positions == [3]

    def test_to_memory(self, sample_fasta_file):
        on_file = ProteinSequencesOnFile(sample_fasta_file)
        in_memory = on_file.to_memory()
        assert isinstance(in_memory, ProteinSequences)
        assert len(in_memory) == 3
        assert str(in_memory[0]) == "ACDE"

    def test_to_dict(self, sample_fasta_file):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        d = sequences.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 3
        assert d["seq1"] == "ACDE"

    def test_to_fasta(self, sample_fasta_file, tmp_path):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        output_path = tmp_path / "output.fasta"
        sequences.to_fasta(str(output_path))
        assert output_path.is_file()

    def test_iter_batches(self, sample_fasta_file):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        batches = list(sequences.iter_batches(2))
        assert len(batches) == 2
        assert isinstance(batches[0], ProteinSequences)
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    def test_multi_line_sequence(self, multi_line_fasta_file):
        sequences = ProteinSequencesOnFile(multi_line_fasta_file)
        assert len(sequences) == 3
        assert not sequences.aligned
        assert not sequences.fixed_length
        assert sequences.has_gaps

        # Check if the multi-line sequence is read correctly
        assert str(sequences[0]) == "ACDEFGHIJKLM"
        assert str(sequences["seq1"]) == "ACDEFGHIJKLM"
        assert len(sequences[0]) == 12

        # Check other sequences
        assert str(sequences[1]) == "ACDF"
        assert str(sequences[2]) == "ACD-"

        # Test iteration
        seqs = list(sequences)
        assert len(seqs) == 3
        assert str(seqs[0]) == "ACDEFGHIJKLM"
        assert str(seqs[1]) == "ACDF"
        assert str(seqs[2]) == "ACD-"

        # Test to_memory
        in_memory = sequences.to_memory()
        assert isinstance(in_memory, ProteinSequences)
        assert len(in_memory) == 3
        assert str(in_memory[0]) == "ACDEFGHIJKLM"

        # Test to_dict
        d = sequences.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 3
        assert d["seq1"] == "ACDEFGHIJKLM"
        assert d["seq2"] == "ACDF"
        assert d["seq3"] == "ACD-"

# Integration tests
def test_integration():
    # Create a ProteinSequence
    seq = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="test_seq")
    
    # Mutate it
    mutated_seq = seq.mutate("A1M")
    
    # Create ProteinSequences
    sequences = ProteinSequences([seq, mutated_seq])
    
    # Check properties
    assert sequences.aligned
    assert sequences.fixed_length
    assert sequences.width == 20
    assert not sequences.has_gaps
    assert sequences.mutated_positions == [0]
    
    # Write to FASTA and read back
    with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        sequences.to_fasta(temp_file.name)
        on_file = ProteinSequencesOnFile.from_fasta(temp_file.name)
    
    # Check if the data is preserved
    assert len(on_file) == 2
    assert str(on_file[0]) == str(seq)
    assert str(on_file[1]) == str(mutated_seq)
    
    os.unlink(temp_file.name)


class TestMSAIntegration:
    """Integration tests for MSA functionality with other library components."""
    
    @pytest.fixture
    def sample_msa_file(self, tmp_path):
        """Create a sample MSA file."""
        msa_content = ">seq1\nACDE\n>seq2\nACDF\n>seq3\nACDG\n"
        msa_path = tmp_path / "sample.fasta"
        msa_path.write_text(msa_content)
        return str(msa_path)
    
    @pytest.fixture
    def sample_a3m_file(self, tmp_path):
        """Create a sample A3M file."""
        a3m_content = ">seq1\nACDE\n>seq2\nACDF\n>seq3\nACDG\n"
        a3m_path = tmp_path / "sample.a3m"
        a3m_path.write_text(a3m_content)
        return str(a3m_path)
    
    def test_load_from_fasta_and_msa_operations(self, sample_msa_file):
        """Test loading from FASTA and performing operations with MSA."""
        # Load sequence with MSA from FASTA
        seq = ProteinSequence.from_fasta(sample_msa_file)
        
        # Verify basic properties
        assert str(seq) == "ACDE"
        assert seq.id == "seq1"
        assert seq.has_msa
        assert len(seq.msa) == 3
        assert seq.is_in_msa
        
        # Test that sequence is found in the MSA
        assert str(seq.msa[0]) == "ACDE"
        assert seq.msa[0].id == "seq1"
        
        # Create a mutated version
        mutated = seq.mutate("A1M")
        assert str(mutated) == "MCDE"
        assert not mutated.has_msa  # Mutation should not preserve MSA
        
        # Demonstrate how to transfer MSA to mutated sequence
        mutated.msa = seq.msa
        assert mutated.has_msa
        assert not mutated.is_in_msa  # Mutated sequence is not in the MSA
    
    def test_load_from_a3m_and_msa_operations(self, sample_a3m_file):
        """Test loading from A3M and performing operations with MSA."""
        # Load sequence with MSA from A3M
        seq = ProteinSequence.from_a3m(sample_a3m_file)
        
        # Verify basic properties
        assert str(seq) == "ACDE"
        assert seq.id == "seq1"
        assert seq.has_msa
        assert len(seq.msa) == 3
        assert seq.is_in_msa
        
        # Test that sequence is found in the MSA
        assert str(seq.msa[0]) == "ACDE"
        assert seq.msa[0].id == "seq1"
    
    def test_hash_stability_with_msa(self, sample_msa_file):
        """Test hash stability with MSA."""
        # Load sequence with MSA
        seq1 = ProteinSequence.from_fasta(sample_msa_file)
        
        # Calculate hash
        hash1 = hash(seq1)
        
        # Load again - should have same hash
        seq2 = ProteinSequence.from_fasta(sample_msa_file)
        hash2 = hash(seq2)
        
        # Hashes should be the same for identical sequences with same MSA
        assert hash1 == hash2
        
        # Modify MSA and verify hash changes
        seq2.msa = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDF", id="seq2")
        ])
        hash3 = hash(seq2)
        assert hash1 != hash3
    
    @patch('aide_predict.utils.data_structures.sequences.mafft_align')
    def test_align_operations_with_msa(self, mock_mafft_align, sample_msa_file):
        """Test alignment operations with sequence having MSA."""
        # Load sequence with MSA
        seq = ProteinSequence.from_fasta(sample_msa_file)
        
        # Create a second sequence without MSA
        seq2 = ProteinSequence("ACDX", id="seq4")
        
        # Mock the alignment function to preserve MSA properties
        def mock_align(sequences, **kwargs):
            # Create aligned sequences but preserve MSA properties
            aligned_seqs = ProteinSequences([
                ProteinSequence("ACDE", id="seq1"),
                ProteinSequence("ACDX", id="seq4")
            ])
            # Transfer MSA from original sequences to aligned ones where present
            for i, orig_seq in enumerate(sequences):
                if orig_seq.has_msa:
                    aligned_seqs[i]._msa = orig_seq.msa
            return aligned_seqs
        
        mock_mafft_align.side_effect = mock_align
        
        # Create a collection including both sequences
        sequences = ProteinSequences([seq, seq2])
        
        # Align the sequences
        aligned = sequences.align_all()
        
        # Check mafft_align was called
        mock_mafft_align.assert_called_once()
        
        # Verify MSA is preserved in the aligned result
        assert aligned[0].has_msa
        assert aligned[0].msa == seq.msa
        
        # The second sequence should still not have an MSA
        assert not aligned[1].has_msa
    
    def test_msa_with_protein_sequences_operations(self, sample_msa_file):
        """Test MSA interaction with ProteinSequences operations."""
        # Load sequence with MSA
        seq = ProteinSequence.from_fasta(sample_msa_file)
        
        # Create a ProteinSequences object containing this sequence
        sequences = ProteinSequences([seq])
        
        # Test that the MSA is preserved in the collection
        assert sequences[0].has_msa
        assert sequences[0].msa == seq.msa
        
        # Test that operations preserve MSA
        upper_sequences = sequences.upper()
        assert upper_sequences[0].has_msa
        assert upper_sequences[0].msa == seq.msa
        
        # Add sequences without MSA
        sequences.append(ProteinSequence("ACDF", id="new_seq"))
        
        # The original sequence with MSA should still have it
        assert sequences[0].has_msa
        
        # The new sequence should not have 

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])