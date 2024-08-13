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
from aide_predict.utils.data_structures import ProteinStructure, StructureMapper
from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences    

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


class TestStructureMapper:
    @pytest.fixture
    def temp_structure_folder(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a PDB file
            with open(os.path.join(tmpdirname, "protein1.pdb"), "w") as f:
                f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N")
            
            # Create an AlphaFold2-like folder
            af2_folder = os.path.join(tmpdirname, "protein2")
            os.mkdir(af2_folder)
            with open(os.path.join(af2_folder, "rank_001_model_1_relaxed_.pdb"), "w") as f:
                f.write("ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N")
            with open(os.path.join(af2_folder, "rank_001_model_1_scores_.json"), "w") as f:
                f.write('{"plddt": [90.0]}')

            yield tmpdirname

    def test_initialization(self, temp_structure_folder):
        mapper = StructureMapper(temp_structure_folder)
        assert isinstance(mapper, StructureMapper)
        assert mapper.structure_folder == temp_structure_folder
        assert len(mapper.structure_map) == 2

    def test_scan_folder(self, temp_structure_folder):
        mapper = StructureMapper(temp_structure_folder)
        assert "protein1" in mapper.structure_map
        assert "protein2" in mapper.structure_map
        assert isinstance(mapper.structure_map["protein1"], ProteinStructure)
        assert isinstance(mapper.structure_map["protein2"], ProteinStructure)

    def test_is_af2_folder(self, temp_structure_folder):
        mapper = StructureMapper(temp_structure_folder)
        assert mapper._is_af2_folder(os.path.join(temp_structure_folder, "protein2"))
        assert not mapper._is_af2_folder(temp_structure_folder)

    def test_assign_structures_to_sequence(self, temp_structure_folder):
        mapper = StructureMapper(temp_structure_folder)
        seq = ProteinSequence("ACGT", id="protein1")
        assigned_seq = mapper.assign_structures(seq)
        assert assigned_seq.structure is not None
        assert isinstance(assigned_seq.structure, ProteinStructure)

    def test_assign_structures_to_sequences(self, temp_structure_folder):
        mapper = StructureMapper(temp_structure_folder)
        seqs = ProteinSequences([
            ProteinSequence("ACGT", id="protein1"),
            ProteinSequence("TGCA", id="protein2"),
            ProteinSequence("AAAA", id="protein3")
        ])
        assigned_seqs = mapper.assign_structures(seqs)
        assert assigned_seqs[0].structure is not None
        assert assigned_seqs[1].structure is not None
        assert assigned_seqs[2].structure is None

    def test_get_available_structures(self, temp_structure_folder):
        mapper = StructureMapper(temp_structure_folder)
        available_structures = mapper.get_available_structures()
        assert set(available_structures) == {"protein1", "protein2"}

    def test_repr(self, temp_structure_folder):
        mapper = StructureMapper(temp_structure_folder)
        assert repr(mapper) == f"StructureMapper(structure_folder='{temp_structure_folder}', available_structures=2)"

    def test_assign_structures_invalid_input(self, temp_structure_folder):
        mapper = StructureMapper(temp_structure_folder)
        with pytest.raises(ValueError):
            mapper.assign_structures("invalid input")