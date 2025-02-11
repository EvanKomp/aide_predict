# aide_predict/utils/mmseqs_msa_search.py
'''
* Author: Evan Komp
* Created: 2/10/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Union, Optional

from aide_predict.utils.data_structures import ProteinSequences
from aide_predict.utils.common import MessageBool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Check if mmseqs2 is available
try:
    MMSEQS = shutil.which("mmseqs")
    if not MMSEQS:
        raise FileNotFoundError("mmseqs executable not found in PATH")
    AVAILABLE = MessageBool(True, "MMseqs2 is available")
except Exception as e:
    AVAILABLE = MessageBool(False, "MMseqs2 not available: " + str(e))

def run_mmseqs_command(
    params: List[Union[str, Path]], 
    capture_stderr: bool = False,
) -> None:
    """Run an MMseqs2 command with logging."""
    params_str = " ".join(str(p) for p in params)
    logger.info(f"Running mmseqs {params_str}")
    
    subprocess_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT if capture_stderr else subprocess.PIPE,
        "text": True,
        "check": True
    }

    result = subprocess.run([MMSEQS] + params, **subprocess_kwargs)
    if result.stdout:
        logger.info(result.stdout)

def run_mmseqs_search(
    sequences: ProteinSequences,
    uniref_db: Union[str, Path],
    output_dir: Union[str, Path],
    metagenomic_db: Optional[Union[str, Path]] = None,
    mode: str = 'standard',
    threads: int = 4,
    remove_tmp: bool = True
) -> List[str]:
    """
    Generate MSAs for protein sequences using MMseqs2.
    
    Args:
        sequences: Input sequences to generate MSAs for
        uniref_db: Path to UniRef30 MMseqs2 database
        output_dir: Directory to save MSAs
        metagenomic_db: Optional path to environmental sequence database
        mode: Search sensitivity:
            - 'fast': Quick search (sensitivity 4.0)
            - 'standard': Balanced (sensitivity 5.7)
            - 'sensitive': More thorough (sensitivity 7.5)
        threads: Number of CPU threads to use
        remove_tmp: Whether to remove temporary files
    
    Returns:
        List of paths to generated MSA files (one per sequence)
    """
    if not AVAILABLE:
        raise RuntimeError(AVAILABLE.message)
    
    # Convert paths
    output_dir = Path(output_dir)
    uniref_db = Path(uniref_db)
    if metagenomic_db:
        metagenomic_db = Path(metagenomic_db)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up parameters based on mode
    sensitivity = {
        'fast': 3.0,
        'standard': 5.7, 
        'sensitive': 7.5
    }[mode]

    # Check database suffixes following original protocol
    if not (uniref_db.with_suffix('.idx').exists() or uniref_db.with_suffix('.idx.index').exists()):
        logger.info("Search does not use index")
        db_load_mode = 0
        dbSuffix1 = "_seq"
        dbSuffix2 = "_aln"
        dbSuffix3 = ""
    else:
        db_load_mode = 2  # Use mmap for indexed DBs
        dbSuffix1 = dbSuffix2 = dbSuffix3 = ".idx"

    search_params = [
        "--num-iterations", "3",
        "--db-load-mode", str(db_load_mode),
        "-a",  # alignment output
        "-e", "0.001",  # E-value threshold
        "--max-seqs", "10000",
        "--prefilter-mode", "1",
        "-s", f"{sensitivity:.1f}",
        "--cov-mode", "2",
        "--cov", "0.5",  # minimum coverage
    ]

    # Create temporary working directory
    tmp_path = output_dir / "tmp"
    tmp_path.mkdir(exist_ok=True)
        
    # Write sequences to FASTA and create MMseqs2 DB
    query_fasta = tmp_path / "query.fasta"
    sequences.to_fasta(query_fasta)
    query_db = tmp_path / "querydb"
    run_mmseqs_command(["createdb", query_fasta, query_db])

    # Initial search and profile construction
    run_mmseqs_command(["search", query_db, uniref_db, 
                      tmp_path / "res", tmp_path / "tmp",
                      "--threads", str(threads)] + search_params)
    
    # Move and link profile data following original protocol
    run_mmseqs_command(["mvdb", tmp_path / "tmp/latest/profile_1", 
                      tmp_path / "prof_res"])
    run_mmseqs_command(["lndb", query_db / "_h", 
                      tmp_path / "prof_res_h"])

    # Expansion step using proper database paths
    run_mmseqs_command(["expandaln", 
        query_db,
        uniref_db.parent / f"{uniref_db.name}{dbSuffix1}",
        tmp_path / "res",
        uniref_db.parent / f"{uniref_db.name}{dbSuffix2}",
        tmp_path / "res_exp",
        "--db-load-mode", str(db_load_mode),
        "--threads", str(threads),
        "--expansion-mode", "0",
        "-e", "inf",
        "--expand-filter-clusters", "1",
        "--max-seq-id", "0.95",
        "--cov", "0.5",
        "--cov-mode", "2"
    ])
    
    # Realignment
    run_mmseqs_command(["align", 
        tmp_path / "prof_res",
        uniref_db.parent / f"{uniref_db.name}{dbSuffix1}",
        tmp_path / "res_exp",
        tmp_path / "res_exp_realign",
        "--db-load-mode", str(db_load_mode),
        "--threads", str(threads),
        "-e", "10",
        "--max-accept", "10000",
        "--max-rejected", "10000",
        "--alt-ali", "10",
        "-a"
    ])
    
    # Filter results
    run_mmseqs_command(["filterresult",
        query_db,
        uniref_db.parent / f"{uniref_db.name}{dbSuffix1}",
        tmp_path / "res_exp_realign",
        tmp_path / "res_exp_realign_filter",
        "--db-load-mode", str(db_load_mode),
        "--threads", str(threads),
        "--qsc", "-20.0",
        "--max-seq-id", "0.95",
        "--cov", "0.8",
        "--filter-min-enable", "256",
        "--diff", "256"
    ])

    # Create MSA
    run_mmseqs_command(["result2msa",
        query_db,
        uniref_db.parent / f"{uniref_db.name}{dbSuffix1}",
        tmp_path / "res_exp_realign_filter",
        tmp_path / "uniref.a3m",
        "--msa-format-mode", "6",
        "--db-load-mode", str(db_load_mode),
        "--threads", str(threads)
    ])

    # Environmental search if requested
    if metagenomic_db:
        logger.info("Starting environmental sequence search")
        
        # Check metagenomic database suffixes
        if not (metagenomic_db.with_suffix('.idx').exists() or 
                metagenomic_db.with_suffix('.idx.index').exists()):
            logger.info("Metagenomic search does not use index")
            meta_dbSuffix1 = "_seq"
            meta_dbSuffix2 = "_aln"
            meta_dbSuffix3 = ""
        else:
            meta_dbSuffix1 = meta_dbSuffix2 = meta_dbSuffix3 = ".idx"
        
        # Run metagenomic search with same protocol as UniRef
        run_mmseqs_command(["search",
            tmp_path / "prof_res",
            metagenomic_db,
            tmp_path / "res_env",
            tmp_path / "tmp_env",
            "--threads", str(threads)
        ] + search_params)
        
        run_mmseqs_command(["expandaln",
            tmp_path / "prof_res",
            metagenomic_db.parent / f"{metagenomic_db.name}{meta_dbSuffix1}",
            tmp_path / "res_env",
            metagenomic_db.parent / f"{metagenomic_db.name}{meta_dbSuffix2}",
            tmp_path / "res_env_exp",
            "--db-load-mode", str(db_load_mode),
            "--threads", str(threads),
            "-e", "inf"
        ])
        
        run_mmseqs_command(["align",
            tmp_path / "tmp_env/latest/profile_1",
            metagenomic_db.parent / f"{metagenomic_db.name}{meta_dbSuffix1}",
            tmp_path / "res_env_exp",
            tmp_path / "res_env_exp_realign",
            "--db-load-mode", str(db_load_mode),
            "--threads", str(threads),
            "-e", "10",
            "--max-accept", "10000",
            "--max-rejected", "10000",
            "--alt-ali", "10",
            "-a"
        ])
        
        run_mmseqs_command(["filterresult",
            query_db,
            metagenomic_db.parent / f"{metagenomic_db.name}{meta_dbSuffix1}",
            tmp_path / "res_env_exp_realign",
            tmp_path / "res_env_exp_realign_filter",
            "--db-load-mode", str(db_load_mode),
            "--threads", str(threads),
            "--qsc", "-20.0",
            "--max-seq-id", "0.95",
            "--cov", "0.8",
            "--filter-min-enable", "256",
            "--diff", "256"
        ])
        
        run_mmseqs_command(["result2msa",
            query_db,
            metagenomic_db.parent / f"{metagenomic_db.name}{meta_dbSuffix1}",
            tmp_path / "res_env_exp_realign_filter",
            tmp_path / "env.a3m",
            "--msa-format-mode", "6",
            "--db-load-mode", str(db_load_mode),
            "--threads", str(threads)
        ])

        # Merge UniRef and metagenomic results
        run_mmseqs_command(["mergedbs",
            query_db,
            tmp_path / "final.a3m",
            tmp_path / "uniref.a3m",
            tmp_path / "env.a3m",
            "--compressed", "0"
        ])
        
        if remove_tmp:
            for file in ["env.a3m", "uniref.a3m"]:
                run_mmseqs_command(["rmdb", tmp_path / file])
    else:
        # Just use UniRef results
        run_mmseqs_command(["mvdb", tmp_path / "uniref.a3m", 
                          tmp_path / "final.a3m"])
        if remove_tmp:
            run_mmseqs_command(["rmdb", tmp_path / "uniref.a3m"])

    # Extract individual MSAs
    output_paths = []
    for i, seq in enumerate(sequences):
        seq_id = seq.id if seq.id else f"seq_{hash(seq)}"
        output_path = output_dir / f"{seq_id}.a3m"
        
        run_mmseqs_command([
            "extractalignments",
            tmp_path / "final.a3m",
            query_db,
            output_path,
            "--extract-mode", "1",
            "--first-seq-index", str(i)
        ])
        
        output_paths.append(str(output_path))

    # Cleanup temporary files
    if remove_tmp:
        for file in ["prof_res", "prof_res_h", "final.a3m"]:
            run_mmseqs_command(["rmdb", tmp_path / file])
        shutil.rmtree(tmp_path)

    return output_paths

def main():
    """Command line interface."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate MSAs using MMseqs2')
    parser.add_argument('fasta', help='Input FASTA file')
    parser.add_argument('uniref_db', help='Path to UniRef30 database')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--env_db', help='Optional environmental sequence database')
    parser.add_argument('--mode', choices=['fast', 'standard', 'sensitive'], 
                       default='standard', help='Search sensitivity')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads')
    parser.add_argument('--keep-tmp', action='store_true', help='Keep temporary files')
    args = parser.parse_args()
    
    # Set up logging
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    sequences = ProteinSequences.from_fasta(args.fasta)
    msa_paths = run_mmseqs_search(
        sequences=sequences,
        uniref_db=args.uniref_db,
        output_dir=args.output_dir,
        metagenomic_db=args.env_db,
        mode=args.mode,
        threads=args.threads,
        remove_tmp=not args.keep_tmp
    )
    
    print(f"Generated {len(msa_paths)} MSAs in {args.output_dir}")

if __name__ == "__main__":
    main()