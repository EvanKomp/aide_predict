# aide_predict/utils/data_structures/structures.py
'''
* Author: Evan Komp
* Created: 7/10/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
from typing import Optional, Dict, List, Union
import warnings
from dataclasses import dataclass
import glob
import re
import json
from ..constants import AA_MAP

THREE_TO_ONE_AA = {v: k for k, v in AA_MAP.items()}

import numpy as np
from Bio.PDB import PDBParser, Structure, Chain
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

@dataclass(eq=True)
class ProteinStructure:
    pdb_file: str
    chain: str = 'A'
    plddt_file: Optional[str] = None

    def __post_init__(self):
        if not os.path.exists(self.pdb_file):
            raise FileNotFoundError(f"PDB file not found: {self.pdb_file}")
        
        if self.plddt_file and not os.path.exists(self.plddt_file):
            raise FileNotFoundError(f"pLDDT file not found: {self.plddt_file}")

        self._sequence: Optional[str] = None
        self._plddt: Optional[np.ndarray] = None
        self._dssp: Optional[Dict[str, str]] = None

    def get_sequence(self) -> str:
        """
        Get the amino acid sequence from the PDB file.

        Returns:
            str: The amino acid sequence.
        """
        if self._sequence is None:
            parser = PDBParser()
            structure = parser.get_structure("protein", self.pdb_file)
            chain = structure[0][self.chain]
            self._sequence = "".join(
                THREE_TO_ONE_AA[residue.resname]
                for residue in chain
                if residue.id[0] == " ")
        return self._sequence

    def get_plddt(self) -> Optional[np.ndarray]:
        """
        Get the pLDDT scores if available.

        Returns:
            Optional[np.ndarray]: Array of pLDDT scores or None if not available.
        """
        if self.plddt_file and self._plddt is None:
            with open(self.plddt_file) as f:
                data = json.load(f)['plddt']
            self._plddt = np.array(data)
        return self._plddt

    def get_dssp(self) -> Dict[str, str]:
        """
        Get the DSSP secondary structure assignments.

        Returns:
            Dict[str, str]: Dictionary of DSSP assignments.
        """
        if self._dssp is None:
            self._dssp = dssp_dict_from_pdb_file(self.pdb_file)[0]
        return self._dssp

    def validate_sequence(self, protein_sequence: str) -> bool:
        """
        Validate if the given sequence matches the structure's sequence.

        Args:
            protein_sequence (str): The sequence to validate.

        Returns:
            bool: True if the sequences match, False otherwise.
        """
        structure_sequence = self.get_sequence()
        return protein_sequence == structure_sequence

    def get_structure(self) -> Structure:
        """
        Load and return the complete structure.

        Returns:
            Structure: The complete protein structure.
        """
        parser = PDBParser()
        return parser.get_structure("protein", self.pdb_file)

    def get_chain(self) -> Chain:
        """
        Load and return the specified chain.

        Returns:
            Chain: The specified protein chain.
        """
        structure = self.get_structure()
        return structure[0][self.chain]

    def get_residue_positions(self) -> List[int]:
        """
        Get the residue positions present in the structure.

        Returns:
            List[int]: List of residue positions.
        """
        chain = self.get_chain()
        return [residue.id[1] for residue in chain if residue.id[0] == " "]
    
    @classmethod
    def from_af2_folder(cls, folder_path: str, chain: str = 'A') -> 'ProteinStructure':
        """
        Create a ProteinStructure object from an AlphaFold2 prediction folder.

        This method prioritizes the top-ranked relaxed structure. If no relaxed structures
        are available, it selects the top-ranked unrelaxed structure.

        Args:
            folder_path (str): Path to the folder containing AlphaFold2 predictions.
            chain (str): Chain identifier (default is 'A').

        Returns:
            ProteinStructure: A new ProteinStructure object.

        Raises:
            FileNotFoundError: If no suitable PDB file is found in the folder.
        """
        def get_rank(filename):
            match = re.search(r'rank_(\d+)', filename)
            return int(match.group(1)) if match else float('inf')

        # Search for relaxed PDB files
        relaxed_pdbs = glob.glob(os.path.join(folder_path, '*relaxed*.pdb'))
        if relaxed_pdbs:
            pdb_file = min(relaxed_pdbs, key=get_rank)
        else:
            # If no relaxed PDFs, search for ranked PDBs
            ranked_pdbs = glob.glob(os.path.join(folder_path, '*rank_*.pdb'))
            if ranked_pdbs:
                pdb_file = min(ranked_pdbs, key=get_rank)
            else:
                raise FileNotFoundError(f"No suitable PDB file found in {folder_path}")

        # Search for corresponding pLDDT file
        plddt_file = pdb_file.replace('.pdb', '.json')
        plddt_file = plddt_file.replace('_unrelaxed_', '_scores_')
        plddt_file = plddt_file.replace('_relaxed_', '_scores_')
        if not os.path.exists(plddt_file):
            raise FileNotFoundError(f"pLDDT file not found: {plddt_file}")

        return cls(pdb_file, chain, plddt_file)
    
    def __hash__(self) -> int:
        return hash((self.pdb_file, self.chain, self.plddt_file))

    def __repr__(self) -> str:
        return f"ProteinStructure(pdb_file='{self.pdb_file}', chain='{self.chain}')"
    

class StructureMapper:
    """
    A class for mapping protein structures to sequences based on files in a given folder.

    This class scans a specified folder for PDB files and AlphaFold2 prediction folders,
    creates ProteinStructure objects, and can assign these structures to ProteinSequence
    or ProteinSequences objects based on their IDs.

    Attributes:
        structure_folder (str): The path to the folder containing structure files.
        structure_map (Dict[str, ProteinStructure]): A dictionary mapping protein IDs to ProteinStructure objects.
    """

    def __init__(self, structure_folder: str):
        """
        Initialize the StructureMapper with a folder containing structure files.

        Args:
            structure_folder (str): The path to the folder containing structure files.
        """
        self.structure_folder = structure_folder
        self.structure_map: Dict[str, ProteinStructure] = {}
        self._scan_folder()

    def _scan_folder(self):
        """
        Scan the structure folder and populate the structure_map.

        This method looks for .pdb files and AlphaFold2 prediction folders in the
        specified structure_folder and creates ProteinStructure objects for each.
        """
        for item in os.listdir(self.structure_folder):
            item_path = os.path.join(self.structure_folder, item)
            if item.endswith('.pdb'):
                structure_id = os.path.splitext(item)[0]
                self.structure_map[structure_id] = ProteinStructure(item_path)
            elif os.path.isdir(item_path):
                if self._is_af2_folder(item_path):
                    structure_id = item
                    self.structure_map[structure_id] = ProteinStructure.from_af2_folder(item_path)

    def _is_af2_folder(self, folder_path: str) -> bool:
        """
        Check if a folder contains AlphaFold2 prediction outputs.

        Args:
            folder_path (str): The path to the folder to check.

        Returns:
            bool: True if the folder appears to contain AlphaFold2 outputs, False otherwise.
        """
        pdb_files = [f for f in os.listdir(folder_path) if f.endswith('.pdb')]
        return any('rank' in pdb_file for pdb_file in pdb_files)

    def assign_structures(self, sequences: Union['ProteinSequence', 'ProteinSequences']) -> Union['ProteinSequence', 'ProteinSequences']:
        """
        Assign structures to the given protein sequence(s).

        This method attempts to assign a structure to each protein sequence based on its ID.
        If a matching structure is found in the structure_map, it is assigned to the sequence.

        Args:
            sequences (Union['ProteinSequence', 'ProteinSequences']): The protein sequence(s) to assign structures to.

        Returns:
            Union['ProteinSequence', 'ProteinSequences']: The input sequence(s) with structures assigned where possible.

        Raises:
            ValueError: If the input is neither a ProteinSequence nor a ProteinSequences object.
        """
        from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences
        
        if isinstance(sequences, ProteinSequence):
            return self._assign_structure_to_sequence(sequences)
        elif isinstance(sequences, ProteinSequences):
            return ProteinSequences([self._assign_structure_to_sequence(seq) for seq in sequences])
        else:
            raise ValueError("Input must be either ProteinSequence or ProteinSequences")

    def _assign_structure_to_sequence(self, sequence: 'ProteinSequence') -> 'ProteinSequence':
        """
        Assign a structure to a single protein sequence.

        Args:
            sequence (ProteinSequence): The protein sequence to assign a structure to.

        Returns:
            ProteinSequence: The input sequence with a structure assigned if one was found.
        """
        if sequence.id in self.structure_map:
            sequence.structure = self.structure_map[sequence.id]
        return sequence

    def get_available_structures(self) -> List[str]:
        """
        Get a list of all available structure IDs.

        Returns:
            List[str]: A list of structure IDs available in the structure_map.
        """
        return list(self.structure_map.keys())

    def __repr__(self):
        """
        Return a string representation of the StructureMapper object.

        Returns:
            str: A string representation of the StructureMapper.
        """
        return f"StructureMapper(structure_folder='{self.structure_folder}', available_structures={len(self.structure_map)})"