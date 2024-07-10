# aide_predict/utils/data_structures/structures.py
'''
* Author: Evan Komp
* Created: 7/10/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
from typing import Optional, Dict, List
import warnings
import glob

import numpy as np
from Bio.PDB import PDBParser, Structure, Chain
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

class ProteinStructure:
    def __init__(self, pdb_file: str, chain: str = 'A', plddt_file: Optional[str] = None):
        """
        Initialize a ProteinStructure object.

        Args:
            pdb_file (str): Path to the PDB file.
            chain (str): Chain identifier (default is 'A').
            plddt_file (Optional[str]): Path to the pLDDT file, if available.
        """
        self.pdb_file = pdb_file
        self.chain = chain
        self.plddt_file = plddt_file

        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        if plddt_file and not os.path.exists(plddt_file):
            raise FileNotFoundError(f"pLDDT file not found: {plddt_file}")

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
            self._sequence = "".join(residue.resname for residue in chain if residue.id[0] == " ")
        return self._sequence

    def get_plddt(self) -> Optional[np.ndarray]:
        """
        Get the pLDDT scores if available.

        Returns:
            Optional[np.ndarray]: Array of pLDDT scores or None if not available.
        """
        if self.plddt_file and self._plddt is None:
            self._plddt = np.loadtxt(self.plddt_file)
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
        plddt_file = pdb_file.replace('.pdb', '_plddt.txt')
        if not os.path.exists(plddt_file):
            warnings.warn(f"No pLDDT file found for {pdb_file}. pLDDT information will not be available.")
            plddt_file = None

        return cls(pdb_file, chain, plddt_file)

    def __repr__(self) -> str:
        return f"ProteinStructure(pdb_file='{self.pdb_file}', chain='{self.chain}')"