# external_calls/ssemb/_ssemb_score.py
'''
* Author: Kristoffer Enoe Johansson, Evan Komp
* Created: 4/2/2025
* License: MIT

SSEmb Wrapper Script for AIDE

This script runs the SSEmb model for protein variant effect prediction.
It processes protein structures and multiple sequence alignments to generate
scores for variant effects. The script supports both single mutations and
multiple mutations. Single mutations are rigorous masked marginal likelihood, 
while combinatorial mutants are additive over those single mutant scores.

The script requires:
1. A PDB file containing protein structure
2. An MSA file in A3M format
3. SSEmb model weights

Environment Setup:
- SSEMB_REPO: Path to the SSEmb repository
- SSEMB_ENV: Name of the conda environment with SSEmb dependencies

Example:
    python run_ssemb.py --pdb protein.pdb --msa protein.a3m --run ./output \
        --weights ./weights --variants variants.txt --gpu-id 0
'''

import os
import sys
import json
import random
import pickle
import re
import argparse
from pathlib import Path

import torch
import torch_geometric

from collections import OrderedDict
import pandas as pd
import numpy as np

# Add SSEmb source directory to path
SSEMB_REPO = os.environ.get('SSEMB_REPO', '../')
ssemb_src_dir = os.path.join(SSEMB_REPO, 'src/')
sys.path.append(ssemb_src_dir)

try:
    from helpers import remove_insertions, read_msa, forward, loop_pred
    import pdb_parser_scripts.clean_pdb as clean_pdb
    import pdb_parser_scripts.parse_pdbs as parse_pdbs
    from models.msa_transformer.model import MSATransformer
    from models.gvp.models import SSEmbGNN
    import models.gvp.data as gvp_data
except ImportError as e:
    sys.exit(f"Error importing SSEmb modules. Please check SSEMB_REPO environment variable: {e}")


def parse_mutation_string(mutation_string):
    """
    Parse a mutation string into a list of individual mutations.
    
    Args:
        mutation_string (str): String containing one or more mutations separated by semicolons.
            Each mutation should be in the format 'H20L' (wild-type, position, mutant).
        
    Returns:
        list: List of individual mutations as tuples [(wt, pos, mt), ...].
            Example: [('H', 20, 'L'), ('G', 70, 'Y')]
    
    Raises:
        ValueError: If the mutation string format is invalid.
    """
    mutations = []
    for mut in mutation_string.split(';'):
        # Extract wt, position, and mt using regex
        match = re.match(r'([A-Za-z])(\d+)([A-Za-z])', mut)
        if match:
            wt, pos, mt = match.groups()
            mutations.append((wt, int(pos), mt))
        else:
            raise ValueError(f"Invalid mutation format: {mut}. Expected format: 'H20L'")
    return mutations


def load_variant_file(variant_file_path):
    """
    Load a file containing variant specifications.
    
    The file should contain one variant per line, with multiple mutations 
    separated by semicolons (e.g., 'H20L;G70Y').
    
    Args:
        variant_file_path (str): Path to the variant file.
        
    Returns:
        list: List of variant entries, where each entry is a list of mutation tuples.
            Example: [[('H', 20, 'L')], [('G', 70, 'Y'), ('A', 30, 'V')]]
    
    Raises:
        ValueError: If the variant file contains invalid mutation specifications.
    """
    variants = []
    with open(variant_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                try:
                    mutations = parse_mutation_string(line)
                    variants.append(mutations)
                except ValueError as e:
                    print(f"Warning: {e}")
    return variants


def score_multi_mutations(df_single_mutations, protein_name, mutations):
    """
    Score a multi-mutation variant by averaging scores of individual mutations.
    
    Args:
        df_single_mutations (pd.DataFrame): DataFrame with single mutation scores,
            containing columns 'protein', 'variant', and 'score_ml'.
        protein_name (str): Name of the protein.
        mutations (list): List of mutation tuples [(wt, pos, mt), ...].
        
    Returns:
        float: Additive score for the multi-mutation variant.
    
    Raises:
        ValueError: If any of the specified mutations is not found in the
            single mutation scores for the given protein.
    """
    scores = []
    for wt, pos, mt in mutations:
        variant = f"{wt}{pos}{mt}"
        match = df_single_mutations[(df_single_mutations['protein'] == protein_name) & 
                                    (df_single_mutations['variant'] == variant)]
        if len(match) == 0:
            raise ValueError(f"Mutation {variant} not found in single mutation scores for protein {protein_name}")
        scores.append(match['score_ml'].values[0])
    
    # sum scores 
    return sum(scores)


def run_ssemb(pdb_paths, msa_paths, run_path, weights_path, variant_file=None, device=0):
    """
    Run the SSEmb model to predict variant effects.
    
    This function processes protein structures and MSAs, runs the SSEmb model,
    and generates scores for variants. It supports both single mutations and
    multi-mutations if a variant file is provided.
    
    Args:
        pdb_paths (list): List of paths to PDB files.
        msa_paths (list): List of paths to MSA files (in A3M format).
        run_path (str): Directory for output and intermediate files.
        weights_path (str): Directory containing model weights.
        variant_file (str, optional): Path to file containing multi-mutation variants.
        device (int, optional): GPU device ID to use. Defaults to 0.
    
    Returns:
        tuple: Paths to output files (single mutation scores, multi-mutation scores if applicable).
    
    Raises:
        ValueError: If inputs are invalid or processing fails.
    """
    # Check for same number of inputs
    if len(pdb_paths) != len(msa_paths):
        raise ValueError("Different number of PDB and MSA files. Input should be matched lists of structure, MSA and variant files")

    # Create run directory
    run_path = os.path.abspath(run_path)
    if os.path.isdir(run_path):
        print(f"Warning: Will overwrite run-directory: {run_path}")
    else:
        os.makedirs(run_path)

    # Create structure directory
    struc_dir = os.path.join(run_path, "structure")
    if os.path.isdir(struc_dir):
        if os.path.isfile(os.path.join(struc_dir, "coords.json")):
            os.remove(os.path.join(struc_dir, "coords.json"))
    else:
        os.mkdir(struc_dir)

    # Create cleaned directory for processed PDBs
    cleaned_dir = os.path.join(struc_dir, "cleaned")
    if os.path.isdir(cleaned_dir):
        # Remove input files from previous runs
        for file_or_dir in os.listdir(cleaned_dir):
            path = os.path.join(cleaned_dir, file_or_dir)
            if os.path.isfile(path) and file_or_dir.lower().endswith('.pdb'):
                os.remove(path)
    else:
        os.mkdir(cleaned_dir)

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

    # Create variant position dictionary
    variant_pos_dict = {}
    for entry in data:
        seq = entry["seq"]
        pos = [str(i + 1) for i in range(len(seq))]
        variant_wtpos_list = [[seq[i] + pos[i]] for i in range(len(seq))]
        variant_wtpos_list = [x for sublist in variant_wtpos_list for x in sublist]
        variant_pos_dict[entry["name"]] = variant_wtpos_list

    # Save data and variant positions dictionary
    with open(os.path.join(run_path, "data_with_msas.pkl"), "wb") as fp:
        pickle.dump(data, fp)

    with open(os.path.join(run_path, "variant_pos_dict.pkl"), "wb") as fp:
        pickle.dump(variant_pos_dict, fp)

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
    letter_to_num = testset.letter_to_num
    test_loader = torch_geometric.loader.DataLoader(testset, batch_size=1, shuffle=False)

    # Run model prediction
    print("Running SSEmb prediction...")
    model_msa.eval()
    model_gvp.eval()

    with torch.no_grad():
        pred_list, acc_mean = loop_pred(
            model_msa,
            model_gvp,
            msa_batch_converter,
            test_loader,
            variant_pos_dict,
            data,
            letter_to_num,
            device=device,
        )

    # Create position-level results dataframe
    df_ml = pd.DataFrame(pred_list, columns=["protein", "variant_pos", "score_ml_pos"])

    # Save position-level results
    pos_scores_path = os.path.join(run_path, "ssemb_pos.csv")
    df_ml.to_csv(pos_scores_path, index=False)
    print(f"Position-level scores saved to: {pos_scores_path}")

    # Compute single mutation scores from position-level scores
    pred_list_scores = []
    mt_list = [x for x in sorted(letter_to_num, key=letter_to_num.get)][:-1]  # Exclude special tokens

    for entry in data:
        protein = entry["name"]
        df_protein = df_ml[df_ml["protein"] == protein]

        # Create variants for all possible mutations
        wt = [[wt] * 20 for wt in entry["seq"]]
        pos = [[pos] * 20 for pos in list(df_protein["variant_pos"])]
        pos = [item for sublist in pos for item in sublist]
        mt = mt_list * len(wt)
        wt = [item for sublist in wt for item in sublist]
        score_ml = [
            item for sublist in list(df_protein["score_ml_pos"]) for item in sublist
        ]

        # Create rows for dataframe
        rows = [
            [protein, wt[i] + str(pos[i]) + mt[i], score_ml[i]] for i in range(len(mt))
        ]
        pred_list_scores += rows

    # Create single mutation scores dataframe
    df_ml_scores = pd.DataFrame(
        pred_list_scores, columns=["protein", "variant", "score_ml"]
    )

    # Save single mutation scores
    single_scores_path = os.path.join(run_path, "ssemb_scores.csv")
    df_ml_scores.to_csv(single_scores_path, index=False)
    print(f"Single mutation scores saved to: {single_scores_path}")

    multi_scores_path = None
    
    # Process multi-mutation variants if a variant file is provided
    if variant_file and os.path.isfile(variant_file):
        print(f"Processing multi-mutation variants from file: {variant_file}")
        
        multi_mutations = load_variant_file(variant_file)
        multi_mutation_scores = []
        
        for entry in data:
            protein = entry["name"]
            for mutations in multi_mutations:
                try:
                    # Verify that all mutations match the protein sequence
                    for wt, pos, mt in mutations:
                        # Convert position to 0-based indexing for sequence check
                        seq_pos = pos - 1
                        if seq_pos < 0 or seq_pos >= len(entry["seq"]):
                            raise ValueError(f"Position {pos} is out of range for protein {protein} with length {len(entry['seq'])}")
                        if entry["seq"][seq_pos] != wt:
                            raise ValueError(f"Wild-type amino acid '{wt}' at position {pos} does not match protein sequence '{entry['seq'][seq_pos]}'")
                    
                    # Score the multi-mutation
                    avg_score = score_multi_mutations(df_ml_scores, protein, mutations)
                    
                    # Create mutation string for output
                    mut_string = ';'.join([f"{wt}{pos}{mt}" for wt, pos, mt in mutations])
                    
                    multi_mutation_scores.append([protein, mut_string, len(mutations), avg_score])
                except ValueError as e:
                    print(f"Error processing multi-mutation variant: {e}")
        
        # Create and save DataFrame for multi-mutation scores
        if multi_mutation_scores:
            df_multi = pd.DataFrame(
                multi_mutation_scores, 
                columns=["protein", "variant", "num_mutations", "score_ml"]
            )
            multi_scores_path = os.path.join(run_path, "ssemb_multi_scores.csv")
            df_multi.to_csv(multi_scores_path, index=False)
            print(f"Multi-mutation scores saved to: {multi_scores_path}")
    
    return single_scores_path, multi_scores_path


def main():
    """
    Main function to parse command line arguments and run SSEmb.
    """
    # Parse commandline arguments
    arg_parser = argparse.ArgumentParser(description="SSEmb wrapper script")
    
    # Required arguments
    arg_parser.add_argument("--pdb", metavar="FILE", required=True,
                      help="Input structure in PDB format")
    arg_parser.add_argument("--msa", metavar="FILE", required=True,
                      help="Input MSA in A3M format")
    
    # Optional arguments
    arg_parser.add_argument("--run", metavar="PATH", default="ssemb",
                      help="Directory for output and intermediate files (default: ./ssemb)")
    arg_parser.add_argument("--weights", metavar="PATH", default=os.path.join(SSEMB_REPO, "weights"),
                      help="Directory with weight files for the SSEmb model")
    arg_parser.add_argument("--variants", metavar="FILE",
                      help="File containing list of variants to score (optional)")
    arg_parser.add_argument("--gpu-id", metavar="INT", type=int, default=0,
                      help="Index of GPU to be used (default: 0)")
    
    args = arg_parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.pdb):
        sys.exit(f"Error: PDB file '{args.pdb}' not found")
    if not os.path.exists(args.msa):
        sys.exit(f"Error: MSA file '{args.msa}' not found")
    if not os.path.exists(args.weights):
        sys.exit(f"Error: Weights directory '{args.weights}' not found")
    if args.variants and not os.path.exists(args.variants):
        sys.exit(f"Error: Variants file '{args.variants}' not found")
    
    # Run SSEmb
    try:
        single_scores_path, multi_scores_path = run_ssemb(
            pdb_paths=[args.pdb],
            msa_paths=[args.msa],
            run_path=args.run,
            weights_path=args.weights,
            variant_file=args.variants,
            device=args.gpu_id
        )
        
        print("\nSSEmb prediction completed successfully.")
        print(f"Single mutation scores: {single_scores_path}")
        if multi_scores_path:
            print(f"Multi-mutation scores: {multi_scores_path}")
        
    except Exception as e:
        sys.exit(f"Error running SSEmb: {e}")


if __name__ == "__main__":
    main()
