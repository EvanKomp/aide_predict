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

    @pytest.fixture
    def temp_multichain_pdb(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write("""ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.362   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.702   2.144   0.907  1.00  0.00           O
ATOM      5  N   GLY A   2       2.831   1.687  -0.987  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.396   3.037  -1.009  1.00  0.00           C
ATOM      7  C   GLY A   2       2.362   4.089  -1.408  1.00  0.00           C
ATOM      8  O   GLY A   2       2.730   5.261  -1.509  1.00  0.00           O
TER       9      GLY A   2
ATOM     10  N   CYS B   1      20.000  20.000  20.000  1.00  0.00           N
ATOM     11  CA  CYS B   1      21.458  20.000  20.000  1.00  0.00           C
ATOM     12  C   CYS B   1      22.009  21.362  20.000  1.00  0.00           C
ATOM     13  O   CYS B   1      21.702  22.144  20.907  1.00  0.00           O
ATOM     14  N   ASP B   2      22.831  21.687  19.013  1.00  0.00           N
ATOM     15  CA  ASP B   2      23.396  23.037  18.991  1.00  0.00           C
ATOM     16  C   ASP B   2      22.362  24.089  18.592  1.00  0.00           C
ATOM     17  O   ASP B   2      22.730  25.261  18.491  1.00  0.00           O
TER      18      ASP B   2
END
""")
        yield f.name
        os.unlink(f.name)

    def test_context_chains_tuple_normalization(self, temp_multichain_pdb):
        # Lists are normalized to tuples for hashability.
        s = ProteinStructure(temp_multichain_pdb, chain='A', context_chains=['B'])
        assert s.context_chains == ('B',)

    def test_context_chains_default_none(self, temp_multichain_pdb):
        s = ProteinStructure(temp_multichain_pdb, chain='A')
        assert s.context_chains is None

    def test_context_chains_in_hash(self, temp_multichain_pdb):
        s1 = ProteinStructure(temp_multichain_pdb, chain='A')
        s2 = ProteinStructure(temp_multichain_pdb, chain='A', context_chains=('B',))
        assert hash(s1) != hash(s2)

    def test_context_chains_in_repr(self, temp_multichain_pdb):
        s1 = ProteinStructure(temp_multichain_pdb, chain='A')
        s2 = ProteinStructure(temp_multichain_pdb, chain='A', context_chains=('B',))
        assert 'context_chains' not in repr(s1)
        assert "context_chains=('B',)" in repr(s2)

    def test_context_chains_cannot_include_primary(self, temp_multichain_pdb):
        with pytest.raises(ValueError, match="primary chain"):
            ProteinStructure(temp_multichain_pdb, chain='A', context_chains=('A',))

    def test_context_chains_missing_raises(self, temp_multichain_pdb):
        with pytest.raises(ValueError, match="not found"):
            ProteinStructure(temp_multichain_pdb, chain='A', context_chains=('Z',))

    def test_get_all_chain_ids(self, temp_multichain_pdb):
        s = ProteinStructure(temp_multichain_pdb, chain='A')
        assert set(s.get_all_chain_ids()) == {'A', 'B'}

    @pytest.fixture
    def temp_pdb_with_ligand_chain(self):
        # Chain A has protein residues (ALA, GLY); chain Z has only HETATM (a heme
        # cofactor). Default get_all_chain_ids should hide Z.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write("""ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.362   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.702   2.144   0.907  1.00  0.00           O
ATOM      5  N   GLY A   2       2.831   1.687  -0.987  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.396   3.037  -1.009  1.00  0.00           C
ATOM      7  C   GLY A   2       2.362   4.089  -1.408  1.00  0.00           C
ATOM      8  O   GLY A   2       2.730   5.261  -1.509  1.00  0.00           O
TER       9      GLY A   2
HETATM   10 FE   HEM Z   1      30.000  30.000  30.000  1.00  0.00          FE
HETATM   11  NA  HEM Z   1      30.500  30.000  30.000  1.00  0.00           N
END
""")
        yield f.name
        os.unlink(f.name)

    def test_get_all_chain_ids_filters_non_protein(self, temp_pdb_with_ligand_chain):
        s = ProteinStructure(temp_pdb_with_ligand_chain, chain='A')
        # Default: protein chains only.
        assert s.get_all_chain_ids() == ['A']
        # Opt-in to the full list.
        assert set(s.get_all_chain_ids(protein_only=False)) == {'A', 'Z'}

    def test_context_chains_accept_non_protein_explicit(self, temp_pdb_with_ligand_chain):
        # Power users can still attach a ligand chain as context explicitly.
        s = ProteinStructure(
            temp_pdb_with_ligand_chain, chain='A', context_chains=('Z',),
        )
        assert s.context_chains == ('Z',)

    def test_set_target_chain_auto_context_skips_non_protein(self, temp_pdb_with_ligand_chain):
        s = ProteinStructure(temp_pdb_with_ligand_chain, chain='A')
        # Re-target chain A (no-op for chain but exercises the auto_context path).
        s.set_target_chain('A')
        # The HEM cofactor chain Z must NOT be pulled in as auto context.
        assert s.context_chains is None

    def test_get_chain_coords_shape_and_dtype(self, temp_multichain_pdb):
        s = ProteinStructure(temp_multichain_pdb, chain='A')
        coords = s.get_chain_coords('A')
        assert coords.shape == (2, 3, 3)
        assert coords.dtype == np.float32
        assert not np.any(np.isnan(coords))
        np.testing.assert_allclose(coords[0, 0], [0.0, 0.0, 0.0], rtol=1e-5)

    def test_get_chain_coords_context_chain(self, temp_multichain_pdb):
        s = ProteinStructure(temp_multichain_pdb, chain='A', context_chains=('B',))
        coords_b = s.get_chain_coords('B')
        assert coords_b.shape == (2, 3, 3)
        np.testing.assert_allclose(coords_b[0, 0], [20.0, 20.0, 20.0], rtol=1e-5)

    def test_set_target_chain(self, temp_multichain_pdb):
        s = ProteinStructure(temp_multichain_pdb, chain='A')
        assert s.get_sequence() == 'AG'
        s.set_target_chain('B')
        assert s.chain == 'B'
        assert s.context_chains == ('A',)
        # Cache must be invalidated so the new chain's sequence is returned.
        assert s.get_sequence() == 'CD'

    def test_set_target_chain_no_auto_context(self, temp_multichain_pdb):
        s = ProteinStructure(temp_multichain_pdb, chain='A', context_chains=('B',))
        s.set_target_chain('B', auto_context=False)
        assert s.chain == 'B'
        assert s.context_chains is None

    def test_set_target_chain_invalid(self, temp_multichain_pdb):
        s = ProteinStructure(temp_multichain_pdb, chain='A')
        with pytest.raises(ValueError, match="not found"):
            s.set_target_chain('Z')


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

    @pytest.fixture
    def temp_multichain_folder(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(os.path.join(tmpdirname, "complex.pdb"), "w") as f:
                f.write("""ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.362   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.702   2.144   0.907  1.00  0.00           O
ATOM      5  N   GLY A   2       2.831   1.687  -0.987  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.396   3.037  -1.009  1.00  0.00           C
ATOM      7  C   GLY A   2       2.362   4.089  -1.408  1.00  0.00           C
ATOM      8  O   GLY A   2       2.730   5.261  -1.509  1.00  0.00           O
TER       9      GLY A   2
ATOM     10  N   CYS B   1      20.000  20.000  20.000  1.00  0.00           N
ATOM     11  CA  CYS B   1      21.458  20.000  20.000  1.00  0.00           C
ATOM     12  C   CYS B   1      22.009  21.362  20.000  1.00  0.00           C
ATOM     13  O   CYS B   1      21.702  22.144  20.907  1.00  0.00           O
ATOM     14  N   ASP B   2      22.831  21.687  19.013  1.00  0.00           N
ATOM     15  CA  ASP B   2      23.396  23.037  18.991  1.00  0.00           C
ATOM     16  C   ASP B   2      22.362  24.089  18.592  1.00  0.00           C
ATOM     17  O   ASP B   2      22.730  25.261  18.491  1.00  0.00           O
TER      18      ASP B   2
END
""")
            yield tmpdirname

    def test_get_protein_sequences_uniform_chain(self, temp_multichain_folder):
        mapper = StructureMapper(temp_multichain_folder)
        seqs = mapper.get_protein_sequences(target_chain='A')
        assert len(seqs) == 1
        assert str(seqs[0]) == 'AG'
        assert seqs[0].structure.chain == 'A'
        assert seqs[0].structure.context_chains == ('B',)

    def test_get_protein_sequences_other_chain(self, temp_multichain_folder):
        mapper = StructureMapper(temp_multichain_folder)
        seqs = mapper.get_protein_sequences(target_chain='B')
        assert str(seqs[0]) == 'CD'
        assert seqs[0].structure.chain == 'B'
        assert seqs[0].structure.context_chains == ('A',)

    def test_get_protein_sequences_json_map(self, temp_multichain_folder, tmp_path):
        mapper = StructureMapper(temp_multichain_folder)
        chain_map = {"complex": "B"}
        json_path = tmp_path / "chain_map.json"
        json_path.write_text(json.dumps(chain_map))
        seqs = mapper.get_protein_sequences(target_chain=str(json_path))
        assert seqs[0].structure.chain == 'B'
        assert str(seqs[0]) == 'CD'

    def test_get_protein_sequences_no_auto_context(self, temp_multichain_folder):
        mapper = StructureMapper(temp_multichain_folder)
        seqs = mapper.get_protein_sequences(target_chain='A', auto_context=False)
        assert seqs[0].structure.context_chains is None