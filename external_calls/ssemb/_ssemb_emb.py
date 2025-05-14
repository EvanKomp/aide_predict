# external_calls/ssemb/_ssemb_embed.py
'''
* Author: Kristoffer Enoe Johansson, Evan Komp
* Created: 5/6/2025
* License: MIT

SSEmb Embeddings Extraction Script for AIDE

This script extracts embeddings from the SSEmb model for protein structures
and multiple sequence alignments. It processes each protein-MSA pair to generate
embeddings that combine sequence and structural information.

The script requires:
1. A list of PDB files containing protein structures
2. A matching list of MSA files in A3M format
3. SSEmb model weights

Environment Setup:
- SSEMB_REPO: Path to the SSEmb repository
- SSEMB_ENV: Name of the conda environment with SSEmb dependencies

Example:
    python extract_ssemb_embeddings.py --pdb_list pdb_files.txt --msa_list msa_files.txt \
        --output ./embeddings --weights ./weights --gpu-id 0
'''

import os
import sys
import json
import random
import pickle
import re
import argparse
from pathlib import Path
import numpy as np
import h5py

import torch
import torch_geometric

from collections import OrderedDict
import pandas as pd

# Add SSEmb source directory to path
SSEMB_REPO = os.environ.get('SSEMB_REPO', '../')
ssemb_src_dir = os.path.join(SSEMB_REPO, 'src/')
sys.path.append(ssemb_src_dir)

try:
    from helpers import remove_insertions, read_msa, forward, loop_getemb
    import pdb_parser_scripts.clean_pdb as clean_pdb
    import pdb_parser_scripts.parse_pdbs as parse_pdbs
    from models.msa_transformer.model import MSATransformer
    from models.gvp.models import SSEmbGNN
    import models.gvp.data as gvp_data
except ImportError as e:
    sys.exit(f"Error importing SSEmb modules. Please check SSEMB_REPO environment variable: {e}")


def load_files_list(list_file):
    """
    Load a list of files from a text file.
    
    Args:
        list_file (str): Path to a file containing a list of file paths, one per line.
        
    Returns:
        list: List of file paths.
    """
    with open(list_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def extract_embeddings(pdb_paths, msa_paths, output_dir, weights_path, device=0):
    """
    Extract embeddings from the SSEmb model for each protein-MSA pair.
    
    Args:
        pdb_paths (list): List of paths to PDB files.
        msa_paths (list): List of paths to MSA files (in A3M format).
        output_dir (str): Directory for output embeddings and intermediate files.
        weights_path (str): Directory containing model weights.
        device (int, optional): GPU device ID to use. Defaults to 0.
    
    Returns:
        dict: Dictionary mapping protein names to embeddings.
    """
    # Check for same number of inputs
    if len(pdb_paths) != len(msa_paths):
        raise ValueError("Different number of PDB and MSA files. Input should be matched lists.")

    # Create run directory
    run_path = os.path.abspath(output_dir)
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    # Create structure directory
    struc_dir = os.path.join(run_path, "structure")
    if not os.path.isdir(struc_dir):
        os.mkdir(struc_dir)

    # Create cleaned directory for processed PDBs
    cleaned_dir = os.path.join(struc_dir, "cleaned")
    if not os.path.isdir(cleaned_dir):
        os.mkdir(cleaned_dir)
    else:
        # Remove input files from previous runs
        for file_or_dir in os.listdir(cleaned_dir):
            path = os.path.join(cleaned_dir, file_or_dir)
            if os.path.isfile(path) and file_or_dir.lower().endswith('.pdb'):
                os.remove(path)

    # Check PDB files exist
    for pdb_path in pdb_paths:
        if not os.path.isfile(pdb_path):
            raise ValueError(f"Cannot find PDB file {pdb_path}")

    # Process PDB files
    for pdb_path in pdb_paths:
        # Clean PDB file
        clean_pdb.clean_pdb(pdb_path, cleaned_dir)
        
        # Get PDB ID from filename
        pdbid = os.path.basename(pdb_path).split(".pdb")[0]
        pdb_path_clean = os.path.join(cleaned_dir, f"{pdbid}.pdb")
        
        # Check cleaned PDB was created
        if not os.path.isfile(pdb_path_clean):
            raise ValueError(f"Failed to produce clean PDB file for {pdbid}")

    # Parse all processed PDB files to generate coords.json and seqs.fasta
    parse_pdbs.parse(struc_dir)

    # Load all structures
    with open(os.path.join(struc_dir, "coords.json")) as json_file:
        data = json.load(json_file)

    # Assign MSA to each structure input
    if len(msa_paths) != len(data):
        raise ValueError(f"Mismatch between number of MSA files ({len(msa_paths)}) and processed structures ({len(data)})")
    
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    for i, msa_path in enumerate(msa_paths):
        print(f"For entry {data[i]['name']}, reading MSA from file {msa_path}")
        # Make 5 sub-samples of MSA
        data[i]["msa"] = []
        for j in range(5):
            msa = read_msa(msa_path)
            msa_sub = [msa[0]]  # Always include first sequence (query)
            k = min(len(msa) - 1, 16 - 1)  # Sample up to 15 additional sequences
            msa_sub += [msa[idx] for idx in sorted(random.sample(range(1, len(msa)), k))]
            data[i]["msa"].append(msa_sub)

    # Save data for model processing
    with open(os.path.join(run_path, "data_with_msas.pkl"), "wb") as fp:
        pickle.dump(data, fp)

    # Validate data
    for entry in data:
        print(f"Protein {entry['name']} length {len(entry['seq'])}")
        
        # Check coordinates
        if 'coords' not in entry:
            raise ValueError(f"Structure of {entry['name']} in JSON format did not return coordinates")
        
        if len(entry['coords']) != len(entry["seq"]):
            raise ValueError(f"Structure of {entry['name']} in JSON format length {len(entry['seq'])} has {len(entry['coords'])} coordinates")
        
        # Check MSA
        if "msa" not in entry:
            raise ValueError(f"Missing MSA data for {entry['name']}")
        
        n_subsamp = len(entry['msa'])
        print(f"  Found {n_subsamp} MSA subsamples")
        
        for iss in range(n_subsamp):
            # Verify first sequence in MSA matches structure sequence
            if entry['msa'][iss][0][1] != entry["seq"]:
                print(entry["seq"])
                print(entry['msa'][iss][0][1])
                raise ValueError(f"Protein {entry['name']} MSA {iss} first sequence is different from the sequence in the structure input")
            
            # Verify all sequences have same length
            n_res = set(len(seq) for _, seq in entry["msa"][iss])
            if n_res != set([len(entry["seq"])]):
                raise ValueError(f"Some sequences of Protein {entry['name']} MSA {iss} have length different from {len(entry['seq'])}: {n_res}")
    
    # Set device
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load MSA Transformer
    alphabet_path = os.path.join(weights_path, "msa_alphabet.pkl")
    if not os.path.exists(alphabet_path):
        raise ValueError(f"MSA alphabet file not found: {alphabet_path}")
        
    with open(alphabet_path, 'rb') as fh:
        msa_alphabet = pickle.load(fh)
    
    model_msa = MSATransformer()
    model_msa = model_msa.to(device)
    msa_batch_converter = msa_alphabet.get_batch_converter()

    # Load MSA weights
    msa_weights_path = os.path.join(weights_path, "final_cath_msa_transformer_110.pt")
    if not os.path.exists(msa_weights_path):
        raise ValueError(f"MSA weights file not found: {msa_weights_path}")
    
    state_dict_msa = torch.load(msa_weights_path, map_location=device)
    model_dict = OrderedDict()
    
    pattern = re.compile("module.")
    for k, v in state_dict_msa.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
        else:
            model_dict = state_dict_msa

    model_msa.load_state_dict(model_dict)

    # Load GVP model
    node_dim = (256, 64)
    edge_dim = (32, 1)
    model_gvp = SSEmbGNN((6, 3), node_dim, (32, 1), edge_dim)
    model_gvp = model_gvp.to(device)
    
    # Load GVP weights
    gvp_weights_path = os.path.join(weights_path, "final_cath_gvp_110.pt")
    if not os.path.exists(gvp_weights_path):
        raise ValueError(f"GVP weights file not found: {gvp_weights_path}")
    
    state_dict_gvp = torch.load(gvp_weights_path, map_location=device)
    model_dict = OrderedDict()
    
    for k, v in state_dict_gvp.items():
        if k.startswith("module"):
            model_dict[k[7:]] = v
        else:
            model_dict = state_dict_gvp
        
    model_gvp.load_state_dict(model_dict)

    # Initialize data loader
    testset = gvp_data.ProteinGraphData(data)
    test_loader = torch_geometric.loader.DataLoader(testset, batch_size=1, shuffle=False)

    # Extract embeddings
    print("Extracting SSEmb embeddings...")
    model_msa.eval()
    model_gvp.eval()

    with torch.no_grad():
        emb_dict = loop_getemb(
            model_msa,
            model_gvp,
            msa_batch_converter,
            test_loader,
            device=device,
        )

    # Save embeddings to HDF5 file
    embeddings_file = os.path.join(output_dir, "ssemb_embeddings.h5")
    with h5py.File(embeddings_file, 'w') as f:
        # Create a group for each protein
        for protein_name, embedding in emb_dict.items():
            # Convert embeddings to numpy arrays
            embedding_np = np.array([tensor.cpu().numpy() for tensor in embedding])
            # Create dataset for protein
            f.create_dataset(protein_name, data=embedding_np, compression="gzip")
            
            # Add protein length as an attribute
            f[protein_name].attrs['length'] = len(embedding)
            f[protein_name].attrs['dims'] = embedding_np.shape[1]
    
    print(f"Embeddings saved to: {embeddings_file}")
    return emb_dict


def main():
    """Main function to parse command line arguments and run embedding extraction."""
    parser = argparse.ArgumentParser(description="Extract SSEmb embeddings for protein-MSA pairs")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdb_list", help="File containing a list of PDB file paths")
    input_group.add_argument("--pdb", nargs='+', help="One or more PDB files")
    
    msa_group = parser.add_mutually_exclusive_group(required=True)
    msa_group.add_argument("--msa_list", help="File containing a list of MSA file paths")
    msa_group.add_argument("--msa", nargs='+', help="One or more MSA files")
    
    # Output options
    parser.add_argument("--output", required=True, help="Output directory for embeddings")
    parser.add_argument("--weights", default=os.path.join(SSEMB_REPO, "weights"),
                        help="Directory with weight files for the SSEmb model")
    
    # Runtime options
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")
    
    args = parser.parse_args()
    
    # Handle input file lists
    if args.pdb_list:
        pdb_paths = load_files_list(args.pdb_list)
    else:
        pdb_paths = args.pdb
        
    if args.msa_list:
        msa_paths = load_files_list(args.msa_list)
    else:
        msa_paths = args.msa
    
    # Check that we have the same number of PDBs and MSAs
    if len(pdb_paths) != len(msa_paths):
        sys.exit(f"Error: Number of PDB files ({len(pdb_paths)}) does not match number of MSA files ({len(msa_paths)})")
    
    # Run embedding extraction
    try:
        extract_embeddings(
            pdb_paths=pdb_paths,
            msa_paths=msa_paths,
            output_dir=args.output,
            weights_path=args.weights,
            device=args.gpu_id
        )
        
        print("\nSSEmb embedding extraction completed successfully.")
        
    except Exception as e:
        sys.exit(f"Error extracting SSEmb embeddings: {e}")


if __name__ == "__main__":
    main()