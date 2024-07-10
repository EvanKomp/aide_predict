# tests/test_utils/test_data_structures/test_structures.py
'''
* Author: Evan Komp
* Created: 7/10/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import pytest
import os
import tempfile
import json
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from aide_predict.utils.data_structures import ProteinStructure  # Assuming the class is in a file named ProteinStructure.py

class TestProteinStructure:
    @pytest.fixture
    def temp_pdb_file(self):
        # Create a temporary PDB file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write("""ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  
ATOM      3  C   ALA A   1       2.009   1.362   0.000  1.00  0.00           C  
ATOM      4  O   ALA A   1       1.702   2.144   0.907  1.00  0.00           O  
ATOM      5  N   GLY A   2       2.831   1.687  -0.987  1.00  0.00           N  
ATOM      6  CA  GLY A   2       3.396   3.037  -1.009  1.00  0.00           C  
ATOM      7  C   GLY A   2       2.362   4.089  -1.408  1.00  0.00           C  
ATOM      8  O   GLY A   2       2.730   5.261  -1.509  1.00  0.00           O  
END
""")
        yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_plddt_file(self):
        # Create a temporary pLDDT file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"plddt": [90.0, 95.0]}, f)
        yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_af2_folder(self, temp_pdb_file, temp_plddt_file):
        # Create a temporary folder with AlphaFold2-like structure
        with tempfile.TemporaryDirectory() as tmpdirname:
            pdb_name = os.path.basename(temp_pdb_file)
            plddt_name = os.path.basename(temp_plddt_file)
            os.symlink(temp_pdb_file, os.path.join(tmpdirname, f"rank_001_model_1_relaxed_.pdb"))
            os.symlink(temp_plddt_file, os.path.join(tmpdirname, f"rank_001_model_1_scores_.json"))
            yield tmpdirname

    def test_initialization(self, temp_pdb_file, temp_plddt_file):
        structure = ProteinStructure(temp_pdb_file, plddt_file=temp_plddt_file)
        assert structure.pdb_file == temp_pdb_file
        assert structure.plddt_file == temp_plddt_file
        assert structure.chain == 'A'

    def test_get_sequence(self, temp_pdb_file):
        structure = ProteinStructure(temp_pdb_file)
        assert structure.get_sequence() == 'AG'

    def test_get_plddt(self, temp_pdb_file, temp_plddt_file):
        structure = ProteinStructure(temp_pdb_file, plddt_file=temp_plddt_file)
        plddt = structure.get_plddt()
        assert isinstance(plddt, np.ndarray)
        np.testing.assert_array_equal(plddt, np.array([90.0, 95.0]))

    def test_validate_sequence(self, temp_pdb_file):
        structure = ProteinStructure(temp_pdb_file)
        assert structure.validate_sequence('AG') == True
        assert structure.validate_sequence('GA') == False

    def test_get_structure(self, temp_pdb_file):
        structure = ProteinStructure(temp_pdb_file)
        assert isinstance(structure.get_structure(), Structure)

    def test_get_chain(self, temp_pdb_file):
        structure = ProteinStructure(temp_pdb_file)
        assert structure.get_chain().id == 'A'

    def test_get_residue_positions(self, temp_pdb_file):
        structure = ProteinStructure(temp_pdb_file)
        assert structure.get_residue_positions() == [1, 2]

    def test_from_af2_folder(self, temp_af2_folder):
        structure = ProteinStructure.from_af2_folder(temp_af2_folder)
        assert os.path.basename(structure.pdb_file).startswith("rank_001_model_1_relaxed_")
        assert os.path.basename(structure.plddt_file).startswith("rank_001_model_1_scores_")

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ProteinStructure("non_existent_file.pdb")

    def test_plddt_file_not_found(self, temp_pdb_file):
        with pytest.raises(FileNotFoundError):
            ProteinStructure(temp_pdb_file, plddt_file="non_existent_file.json")

    def test_from_af2_folder_no_suitable_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProteinStructure.from_af2_folder(str(tmp_path))