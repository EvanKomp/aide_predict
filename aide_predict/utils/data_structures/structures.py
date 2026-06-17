# aide_predict/utils/data_structures/structures.py
'''
* Author: Evan Komp
* Created: 7/10/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import re
import glob
import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, Structure, Chain
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from typing import Union
from ..constants import AA_MAP

THREE_TO_ONE_AA = {v: k for k, v in AA_MAP.items()}


@dataclass(eq=True)
class ProteinStructure:
    structure_file: str  # Renamed from pdb_file to be more general
    chain: str = 'A'
    plddt_file: Optional[str] = None
    context_chains: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if not os.path.exists(self.structure_file):
            raise FileNotFoundError(f"Structure file not found: {self.structure_file}")

        if self.plddt_file and not os.path.exists(self.plddt_file):
            raise FileNotFoundError(f"pLDDT file not found: {self.plddt_file}")

        self._sequence: Optional[str] = None
        self._plddt: Optional[np.ndarray] = None
        self._dssp: Optional[Dict[str, str]] = None
        self._file_format: Optional[str] = None

        if self.context_chains is not None:
            self.context_chains = tuple(self.context_chains)
            if self.chain in self.context_chains:
                raise ValueError(
                    f"context_chains must not include the primary chain '{self.chain}'."
                )
            # Validate against the full chain list so users can explicitly pass
            # ligand/cofactor chains as context if they want.
            available = set(self.get_all_chain_ids(protein_only=False))
            missing = [c for c in self.context_chains if c not in available]
            if missing:
                raise ValueError(
                    f"context_chains {missing} not found in {self.structure_file}. "
                    f"Available chains: {sorted(available)}"
                )

    @property
    def file_format(self) -> str:
        """
        Determine the file format based on file extension.
        
        Returns:
            str: 'pdb' or 'cif'
        """
        if self._file_format is None:
            ext = os.path.splitext(self.structure_file)[1].lower()
            if ext in ['.cif', '.mmcif']:
                self._file_format = 'cif'
            elif ext in ['.pdb', '.ent']:
                self._file_format = 'pdb'
            else:
                # Try to determine from content if extension is ambiguous
                try:
                    with open(self.structure_file, 'r') as f:
                        first_line = f.readline().strip()
                    if first_line.startswith('data_'):
                        self._file_format = 'cif'
                    else:
                        self._file_format = 'pdb'
                except:
                    # Default to PDB if we can't determine
                    self._file_format = 'pdb'
        return self._file_format

    def _get_parser(self):
        """
        Get the appropriate parser based on file format.
        
        Returns:
            Parser: Either PDBParser or MMCIFParser
        """
        if self.file_format == 'cif':
            return MMCIFParser(QUIET=True)
        else:
            return PDBParser(QUIET=True)

    def get_sequence(self) -> str:
        """
        Get the amino acid sequence from the structure file.

        Returns:
            str: The amino acid sequence.
        """
        if self._sequence is None:
            parser = self._get_parser()
            structure = parser.get_structure("protein", self.structure_file)
            chain = structure[0][self.chain]
            self._sequence = "".join(
                THREE_TO_ONE_AA[residue.resname]
                for residue in chain
                if residue.id[0] == " " and residue.resname in THREE_TO_ONE_AA
            )
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
        
        Note: DSSP requires PDB format. If using CIF, consider converting first
        or using alternative secondary structure assignment methods.

        Returns:
            Dict[str, str]: Dictionary of DSSP assignments.
        """
        if self._dssp is None:
            if self.file_format == 'cif':
                # DSSP typically works with PDB format
                # You might want to implement CIF->PDB conversion or use alternative methods
                raise NotImplementedError("DSSP analysis directly from CIF files is not yet supported. "
                                        "Consider converting to PDB format first.")
            self._dssp = dssp_dict_from_pdb_file(self.structure_file)[0]
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
        parser = self._get_parser()
        return parser.get_structure("protein", self.structure_file)

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

    def get_all_chain_ids(self, protein_only: bool = True) -> List[str]:
        """
        Return chain IDs present in the structure file (first BioPython model).

        Most PDB files include non-protein "chains" — ligands, waters, ions,
        glycans — that aren't useful as ESM-IF context. With the default
        ``protein_only=True`` only chains with at least one canonical amino-acid
        residue (id[0] == ' ' and resname in THREE_TO_ONE_AA) are returned. Set
        ``protein_only=False`` to get the unfiltered list (used internally for
        validation so users can still pass non-protein chain IDs explicitly to
        ``context_chains`` if they want).

        Args:
            protein_only: When True, filter out chains with no canonical
                amino-acid residues. Default True.

        Returns:
            List[str]: Chain IDs in BioPython iteration order.
        """
        structure = self._get_parser().get_structure("protein", self.structure_file)
        chain_ids = []
        for chain in structure[0]:
            if protein_only and not any(
                residue.id[0] == " " and residue.resname in THREE_TO_ONE_AA
                for residue in chain
            ):
                continue
            chain_ids.append(chain.id)
        return chain_ids

    def get_chain_coords(self, chain_id: str) -> np.ndarray:
        """
        Extract N / CA / C backbone coordinates for a chain.

        Missing atoms are filled with NaN, matching the convention used by
        ``esm.inverse_folding.util.extract_coords_from_structure`` — the GVP
        encoder auto-masks NaN coords.

        Args:
            chain_id: The chain to extract. May be ``self.chain`` or any
                entry of ``self.context_chains``.

        Returns:
            np.ndarray of shape ``[L, 3, 3]`` and dtype ``float32`` where the
            second axis is ordered (N, CA, C).
        """
        structure = self._get_parser().get_structure("protein", self.structure_file)
        chain = structure[0][chain_id]
        coords = []
        for residue in chain:
            if residue.id[0] != " " or residue.resname not in THREE_TO_ONE_AA:
                continue
            residue_coords = np.full((3, 3), np.nan, dtype=np.float32)
            for i, atom_name in enumerate(("N", "CA", "C")):
                if atom_name in residue:
                    residue_coords[i] = residue[atom_name].coord
            coords.append(residue_coords)
        return np.stack(coords) if coords else np.zeros((0, 3, 3), dtype=np.float32)

    def set_target_chain(self, new_chain: str, auto_context: bool = True) -> None:
        """
        Switch the primary chain in-place, optionally repopulating ``context_chains``.

        Args:
            new_chain: The chain ID to become the new primary chain. Must
                exist in the structure file.
            auto_context: When True (default), set ``self.context_chains`` to
                a tuple of all *other* chains present in the file. When False,
                ``self.context_chains`` is set to None.

        Raises:
            ValueError: If ``new_chain`` is not present in the structure file.
        """
        # Validate against the full chain list (users may target any chain in the file).
        available_all = self.get_all_chain_ids(protein_only=False)
        if new_chain not in available_all:
            raise ValueError(
                f"Chain '{new_chain}' not found in {self.structure_file}. "
                f"Available chains: {available_all}"
            )
        self.chain = new_chain
        if auto_context:
            # Auto-populate only protein chains as context — ligands/waters would
            # contribute no meaningful structural signal and just bloat the input.
            available_protein = self.get_all_chain_ids(protein_only=True)
            others = tuple(c for c in available_protein if c != new_chain)
            self.context_chains = others if others else None
        else:
            self.context_chains = None
        self._sequence = None
        self._dssp = None

    @classmethod
    def from_af2_folder(cls, folder_path: str, chain: str = 'A') -> 'ProteinStructure':
        """
        Create a ProteinStructure object from an AlphaFold2 prediction folder.

        This method prioritizes the top-ranked relaxed structure. If no relaxed structures
        are available, it selects the top-ranked unrelaxed structure.
        Now supports both PDB and CIF formats.

        Args:
            folder_path (str): Path to the folder containing AlphaFold2 predictions.
            chain (str): Chain identifier (default is 'A').

        Returns:
            ProteinStructure: A new ProteinStructure object.

        Raises:
            FileNotFoundError: If no suitable structure file is found in the folder.
        """
        def get_rank(filename):
            match = re.search(r'rank_(\d+)', filename)
            return int(match.group(1)) if match else float('inf')

        # Search for relaxed structure files (both PDB and CIF)
        relaxed_files = []
        for ext in ['*.pdb', '*.cif', '*.mmcif']:
            relaxed_files.extend(glob.glob(os.path.join(folder_path, f'*relaxed*{ext}')))
        
        if relaxed_files:
            structure_file = min(relaxed_files, key=get_rank)
        else:
            # If no relaxed files, search for ranked files
            ranked_files = []
            for ext in ['*.pdb', '*.cif', '*.mmcif']:
                ranked_files.extend(glob.glob(os.path.join(folder_path, f'*rank*{ext}')))
            
            if ranked_files:
                structure_file = min(ranked_files, key=get_rank)
            else:
                raise FileNotFoundError(f"No suitable structure file found in {folder_path}")

        # Search for corresponding pLDDT file
        base_name = os.path.splitext(structure_file)[0]
        plddt_file = base_name.replace('_unrelaxed', '_scores').replace('_relaxed', '_scores') + '.json'
        
        if not os.path.exists(plddt_file):
            # Try alternative naming conventions
            alt_plddt = structure_file.replace('.pdb', '.json').replace('.cif', '.json')
            alt_plddt = alt_plddt.replace('_unrelaxed_', '_scores_').replace('_relaxed_', '_scores_')
            if os.path.exists(alt_plddt):
                plddt_file = alt_plddt
            else:
                plddt_file = None  # Don't raise error, just set to None

        return cls(structure_file, chain, plddt_file)

    # Backward compatibility properties
    @property
    def pdb_file(self) -> str:
        """
        Backward compatibility property for pdb_file.
        
        Returns:
            str: The structure file path.
        """
        return self.structure_file
    
    @pdb_file.setter
    def pdb_file(self, value: str) -> None:
        """
        Backward compatibility setter for pdb_file.
        
        Args:
            value (str): The structure file path.
        """
        self.structure_file = value
    
    def __hash__(self) -> int:
        return hash((self.structure_file, self.chain, self.plddt_file, self.context_chains))

    def __repr__(self) -> str:
        base = f"ProteinStructure(structure_file='{self.structure_file}', chain='{self.chain}', format='{self.file_format}'"
        if self.context_chains is not None:
            base += f", context_chains={self.context_chains}"
        return base + ")"
    

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
            if item.endswith('.pdb') or item.endswith('.cif') or item.endswith('.mmcif'):
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
    
    def get_protein_sequences(
        self,
        target_chain: Union[str, os.PathLike] = 'A',
        auto_context: bool = True,
    ) -> 'ProteinSequences':
        """
        Build a ProteinSequences from every structure discovered in the folder.

        Args:
            target_chain: Either a single chain ID applied uniformly to every
                structure (e.g. 'A', 'B'), or a filesystem path to a JSON file
                mapping ``{structure_id: chain_id}`` for per-file overrides.
                The path form is detected via ``os.path.isfile``; otherwise the
                value is treated as a chain-ID literal. Files not listed in the
                JSON fall back to chain 'A'. The structure_id keys match those
                used by ``_scan_folder`` (filename without extension, or AF2
                folder name).
            auto_context: When True (default), every produced ProteinStructure
                has ``context_chains`` populated with the other chains present
                in its file. Set to False for strict single-chain mode.

        Returns:
            ProteinSequences: One ProteinSequence per discovered structure,
            with sequence drawn from the resolved primary chain.
        """
        from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence

        chain_map: Dict[str, str] = {}
        default_chain: str = 'A'
        if isinstance(target_chain, (str, os.PathLike)) and os.path.isfile(str(target_chain)):
            with open(target_chain) as f:
                chain_map = json.load(f)
        else:
            default_chain = str(target_chain)

        result = []
        for struct_id, struct in self.structure_map.items():
            desired = chain_map.get(struct_id, default_chain)
            struct.set_target_chain(desired, auto_context=auto_context)
            result.append(ProteinSequence(struct.get_sequence(), id=struct_id, structure=struct))
        return ProteinSequences(result)

    def __repr__(self):
        """
        Return a string representation of the StructureMapper object.

        Returns:
            str: A string representation of the StructureMapper.
        """
        return f"StructureMapper(structure_folder='{self.structure_folder}', available_structures={len(self.structure_map)})"