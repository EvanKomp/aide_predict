# aide_predict/utils/mmseqs_search.py
'''
* Author: Evan Komp
* Created: 12/12/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Tool to run MMseqs2 for MSA construction
'''
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import subprocess
import logging
import shutil
from typing import Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)

class SearchMode(Enum):
    STANDARD = 0  # ColabFold defaults, thorough search
    FAST = 1      # Faster, less sensitive search
    SMALL = 2     # fast search returning a small MSA

@dataclass
class MMseqsParams:
    """Parameters for MMseqs2 sequence search and alignment.
    
    These parameters control the sensitivity and thoroughness of the search and alignment process.
    The defaults are set to match ColabFold's standard settings for UniRef searches.
    """
    # Search parameters
    num_iterations: int = 3
    sensitivity: float = 8.0
    prefilter_mode: int = 0
    max_seqs: int = 10000
    e_value_search: float = 0.1
    
    # Expansion parameters
    expansion_mode: int = 0
    e_value_expand: float = float('inf')
    max_seq_id: float = 0.95
    
    # Alignment parameters
    e_value_align: float = 10.0
    qsc: float = 0.8
    max_accept: int = 100000
    alt_ali: int = 10
    
    # Filter parameters
    filter_msa: bool = True
    filter_min_enable: int = 1000
    diff: int = 3000
    qid_thresholds: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    extract_lines: int = 0
    
    # Computational parameters
    threads: int = 4
    db_load_mode: int = 0

    @classmethod
    def from_mode(cls, mode: SearchMode) -> 'MMseqsParams':
        """Create parameter set based on search mode."""
        if mode == SearchMode.STANDARD:
            return cls()  # Use defaults
        elif mode == SearchMode.FAST:
            return cls(
                sensitivity=4.0,
                max_seqs=300,
                e_value_search=0.001,
                e_value_expand=0.01,
                max_accept=10000,
            )
        elif mode == SearchMode.SMALL:
            return cls(
                sensitivity=4.0,
                max_seqs=300,
                e_value_search=0.001,
                e_value_expand=0.01,
                max_accept=10000,
                filter_min_enable=100,
                diff=256,
                extract_lines=256
            )
        else:
            raise ValueError(f"Unknown search mode: {mode}")

def run_mmseqs_command(mmseqs_binary: Path, command: list) -> None:
    """Run an MMseqs2 command with logging and error handling."""
    cmd_str = " ".join(str(x) for x in command)
    logger.info(f"Running: {mmseqs_binary} {cmd_str}")
    try:
        subprocess.run([mmseqs_binary] + command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"MMseqs2 command failed: {e.stderr}")
        raise

def create_msa(input_fasta: Union[str, Path], 
               uniref_db: Union[str, Path],
               output_dir: Union[str, Path],
               mode: SearchMode = SearchMode.STANDARD,
               mmseqs_binary: Union[str, Path] = "mmseqs",
               params: Optional[MMseqsParams] = None) -> None:
    """
    Create Multiple Sequence Alignments using MMseqs2 and UniRef database.
    
    Args:
        input_fasta: Path to input FASTA file
        uniref_db: Path to UniRef database
        output_dir: Directory to store output MSAs
        mode: Search mode (STANDARD or FAST)
        mmseqs_binary: Path to MMseqs2 binary
        params: Optional custom MMseqsParams object. If None, parameters are set based on mode.
    
    The function creates one MSA file per sequence in the input FASTA,
    named <sequence_id>.a3m in the output directory.
    """
    input_fasta = Path(input_fasta)
    uniref_db = Path(uniref_db)
    output_dir = Path(output_dir)
    mmseqs_binary = Path(mmseqs_binary)

    # Validate inputs
    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA file not found: {input_fasta}")
    if not uniref_db.with_suffix('.dbtype').exists():
        raise FileNotFoundError(f"UniRef database not found: {uniref_db}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided params or create from mode
    params = params or MMseqsParams.from_mode(mode)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create MMseqs2 database from input sequences
        run_mmseqs_command(mmseqs_binary, [
            "createdb", 
            input_fasta, 
            tmp_path / "qdb"
        ])
        
        # Search against UniRef
        search_params = [
            "--num-iterations", str(params.num_iterations),
            "--db-load-mode", str(params.db_load_mode),
            "-a", "-e", str(params.e_value_search),
            "--max-seqs", str(params.max_seqs),
            "--prefilter-mode", str(params.prefilter_mode),
            "-s", f"{params.sensitivity:.1f}",
            "--threads", str(params.threads)
        ]
        
        run_mmseqs_command(mmseqs_binary, [
            "search",
            tmp_path / "qdb",
            uniref_db,
            tmp_path / "res",
            tmp_path / "tmp",
        ] + search_params)
        
        # Create profile database
        run_mmseqs_command(mmseqs_binary, [
            "mvdb",
            tmp_path / "tmp/latest/profile_1",
            tmp_path / "prof_res"
        ])
        
        # Expand alignments
        expand_params = [
            "--expansion-mode", str(params.expansion_mode),
            "-e", str(params.e_value_expand),
            "--expand-filter-clusters", str(params.filter_msa),
            "--max-seq-id", str(params.max_seq_id),
            "--db-load-mode", str(params.db_load_mode),
            "--threads", str(params.threads)
        ]
        
        run_mmseqs_command(mmseqs_binary, [
            "expandaln",
            tmp_path / "qdb",
            uniref_db,
            tmp_path / "res",
            uniref_db,
            tmp_path / "res_exp",
        ] + expand_params)
        
        # Create final alignments
        align_params = [
            "-e", str(params.e_value_align),
            "--max-accept", str(params.max_accept),
            "--alt-ali", str(params.alt_ali),
            "-a",
            "--db-load-mode", str(params.db_load_mode),
            "--threads", str(params.threads)
        ]
        
        run_mmseqs_command(mmseqs_binary, [
            "align",
            tmp_path / "prof_res",
            uniref_db,
            tmp_path / "res_exp",
            tmp_path / "res_exp_realign",
        ] + align_params)
        
        # Filter results
        filter_params = [
            "--db-load-mode", str(params.db_load_mode),
            "--qid", "0",
            "--qsc", str(params.qsc),
            "--diff", "0",
            "--threads", str(params.threads),
            "--max-seq-id", "1.0",
            "--filter-min-enable", str(params.filter_min_enable)
        ]
        
        run_mmseqs_command(mmseqs_binary, [
            "filterresult",
            tmp_path / "qdb",
            uniref_db,
            tmp_path / "res_exp_realign",
            tmp_path / "res_exp_realign_filter",
        ] + filter_params)
        
        # Convert to A3M format and apply final filtering
        result_params = [
            "--msa-format-mode", "6",
            "--db-load-mode", str(params.db_load_mode),
            "--threads", str(params.threads),
            "--filter-msa", str(params.filter_msa),
            "--filter-min-enable", str(params.filter_min_enable),
            "--diff", str(params.diff),
            "--qid", params.qid_thresholds,
            "--qsc", "0",
            "--max-seq-id", str(params.max_seq_id),
        ]
        
        run_mmseqs_command(mmseqs_binary, [
            "result2msa",
            tmp_path / "qdb",
            uniref_db,
            tmp_path / "res_exp_realign_filter",
            tmp_path / "final",
            
        ] + result_params)
        
        # Unpack results to individual A3M files
        run_mmseqs_command(mmseqs_binary, [
            "unpackdb",
            tmp_path / "final",
            output_dir,
            "--unpack-name-mode", "1",
            "--unpack-suffix", ".a3m"
        ])

def main():
    """Command-line interface for MSA creation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create Multiple Sequence Alignments using MMseqs2 and UniRef database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                      help='Input FASTA file')
    parser.add_argument('-d', '--db', type=str, required=True,
                      help='Path to UniRef database')
    parser.add_argument('-o', '--output', type=str, required=True,
                      help='Output directory for MSAs')
    parser.add_argument('--mode', type=int, choices=[0, 1], default=0,
                      help='Search mode: 0 (standard) or 1 (fast)')
    parser.add_argument('--mmseqs', type=str, default='mmseqs',
                      help='Path to MMseqs2 binary')
    parser.add_argument('--threads', type=int, default=4,
                      help='Number of CPU threads to use')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create MSAs
    try:
        mode = SearchMode.STANDARD if args.mode == 0 else SearchMode.FAST
        params = MMseqsParams.from_mode(mode)
        params.threads = args.threads
        
        create_msa(
            input_fasta=args.input,
            uniref_db=args.db,
            output_dir=args.output,
            mode=mode,
            mmseqs_binary=args.mmseqs,
            params=params
        )
    except Exception as e:
        logger.error(f"MSA creation failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()