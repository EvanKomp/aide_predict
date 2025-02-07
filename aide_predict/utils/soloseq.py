# aide_predict/utils/soloseq.py
'''
* Author: Evan Komp
* Created: 2/7/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper for SoloSeq structure prediction. See:
https://github.com/aqlaboratory/openfold/blob/main/docs/inference.md#soloseq-inference

Requires:
1. OpenFold repo cloned and environment installed
2. Model weights downloaded
3. Environment variables set:
   - OPENFOLD_ENV_NAME: Name of conda environment
   - OPENFOLD_DIR: Path to OpenFold repo
'''
import os
import subprocess
import shutil
import argparse
import logging
from pathlib import Path
from typing import Optional, Union, List
from aide_predict.utils.data_structures import ProteinSequences
from aide_predict.utils.common import MessageBool

logger = logging.getLogger(__name__)

try:
    OPENFOLD_ENV = os.environ.get('OPENFOLD_CONDA_ENV')
    OPENFOLD_DIR = os.environ.get('OPENFOLD_REPO')
    if not OPENFOLD_ENV or not OPENFOLD_DIR:
        raise KeyError("Missing environment variables")
    AVAILABLE = MessageBool(True, "SoloSeq is available")
    if not os.path.exists(os.path.join(OPENFOLD_DIR, "openfold", "resources", "openfold_soloseq_params", "seq_model_esm1b_ptm.pt")):
        raise FileNotFoundError("Model weights not found")
        
except Exception as e:
    AVAILABLE = MessageBool(False, 
        "SoloSeq requires OPENFOLD_ENV_NAME and OPENFOLD_DIR environment variables. "
        "Please set these after installing OpenFold."
    )

def run_soloseq(
    sequences: ProteinSequences,
    output_dir: str,
    use_gpu: bool = True,
    skip_relaxation: bool = False,
    save_embeddings: bool = False,
    device: str = "cuda:0",
    force: bool = False
) -> List[str]:
    """
    Run SoloSeq structure prediction on a set of sequences.
    
    Args:
        sequences: Input sequences to predict
        output_dir: Directory to save results
        use_gpu: Whether to use GPU
        skip_relaxation: Skip relaxation step
        save_embeddings: Save ESM embeddings
        device: GPU device to use
        force: If True, rerun predictions even if they exist
        
    Returns:
        List of paths to predicted structure files
        
    Note: Sequences longer than 1022 residues will be truncated.
    """
    if not AVAILABLE:
        raise RuntimeError(AVAILABLE.message)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Filter sequences that already have predictions
    sequences_to_predict = []
    existing_paths = []
    for seq in sequences:
        seq_id = seq.id if seq.id else f"seq_{len(sequences_to_predict)}"
        pdb_path = os.path.join(output_dir, f"{seq_id}.pdb")
        if os.path.exists(pdb_path) and not force:
            logger.info(f"Skipping {seq_id} - prediction already exists")
            existing_paths.append(pdb_path)
        else:
            sequences_to_predict.append(seq)

    if not sequences_to_predict:
        logger.info("All sequences already have predictions")
        return existing_paths

    # Create working directories in ./tmp
    tmp_dir = Path("./tmp/soloseq")
    fasta_dir = tmp_dir / "fasta"
    embeddings_dir = tmp_dir / "embeddings"
    
    # Create directories if they don't exist
    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    try:
        # Write individual FASTA files for sequences that need prediction
        for i, seq in enumerate(sequences_to_predict):
            seq_id = seq.id if seq.id else f"seq_{i}"
            with open(fasta_dir / f"{seq_id}.fasta", 'w') as f:
                f.write(f">{seq_id}\n{str(seq)}\n")
        
        # Run embedding generation
        cmd = [
            "conda", "run", "-n", OPENFOLD_ENV,
            "python", f"{OPENFOLD_DIR}/scripts/precompute_embeddings.py",
            str(fasta_dir),
            str(embeddings_dir)
        ]
        subprocess.run(cmd, check=True)
        
        # Run structure prediction
        ## need a dummy cif file in the tmp directory
        with open(tmp_dir / "empty.cif", 'w') as f:
            f.write("")

        cmd = [
            "conda", "run", "-n", OPENFOLD_ENV,
            "python", f"{OPENFOLD_DIR}/run_pretrained_openfold.py",
            str(fasta_dir),
            "./tmp/soloseq",
            "--use_precomputed_alignments", str(embeddings_dir),
            "--output_dir", output_dir,
            "--model_device", device if use_gpu else "cpu",
            "--config_preset", "seq_model_esm1b_ptm",
            "--openfold_checkpoint_path",
            f"{OPENFOLD_DIR}/openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt"
        ]
        
        if skip_relaxation:
            cmd.append("--skip_relaxation")
            
        subprocess.run(cmd, check=True)
        
        # move all pdb files from output_dir/predictions to outputdir
        # also remove unrelaxed structures if relaxed is present
        for file in os.listdir(os.path.join(output_dir, "predictions")):
            if file.endswith(".pdb"):
                shutil.move(os.path.join(output_dir, "predictions", file), output_dir)
        shutil.rmtree(os.path.join(output_dir, "predictions"))

        for file in os.listdir(output_dir):
            if file.endswith(".pdb"):
                if "unrelaxed" in file and os.path.exists(os.path.join(output_dir, file.replace("unrelaxed.pdb", "relaxed.pdb"))):
                    os.remove(os.path.join(output_dir, file))
        
        # update all names to be the sequence id
        for id in sequences.ids:
            for file in os.listdir(output_dir):
                if file.startswith(str(id)+'_'):
                    os.rename(os.path.join(output_dir, file), os.path.join(output_dir, f"{id}.pdb"))

        # Get all output PDB paths (both new and existing)
        new_paths = []
        for seq in sequences_to_predict:
            seq_id = seq.id if seq.id else f"seq_{i}"
            pdb_path = os.path.join(output_dir, f"{seq_id}.pdb")
            if os.path.exists(pdb_path):
                new_paths.append(pdb_path)
                
        # Clean up if not saving embeddings
        if not save_embeddings:
            shutil.rmtree(embeddings_dir)
        
        return existing_paths + new_paths
    
    finally:
        # Clean up fasta directory
        if os.path.exists(fasta_dir):
            shutil.rmtree(fasta_dir)
            
def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Run SoloSeq structure prediction')
    parser.add_argument('fasta', help='Input FASTA file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--skip_relaxation', action='store_true', help='Skip relaxation')
    parser.add_argument('--save_embeddings', action='store_true', help='Save ESM embeddings')
    parser.add_argument('--device', default="cuda:0", help='GPU device')
    args = parser.parse_args()
    
    sequences = ProteinSequences.from_fasta(args.fasta)
    output_paths = run_soloseq(
        sequences=sequences,
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu,
        skip_relaxation=args.skip_relaxation,
        save_embeddings=args.save_embeddings,
        device=args.device
    )
    
    print(f"Generated {len(output_paths)} structures in {args.output_dir}")

if __name__ == "__main__":
    main()