#!/usr/bin/env python
"""
01_embeddings.py — Precompute all embeddings for MDA HACL Round 3
==================================================================

Computes:
  - ESM2 per-residue embeddings for WT, V83P, and all unique mutants
  - SaProt wildtype-marginal zero-shot scores for all unique mutations
  - Substrate molecular embeddings: Morgan fingerprints, MACCS keys, Mordred 2D

Outputs to data/round3/processed/embeddings/.

Usage:
    # Default: compute everything for the training pipeline
    python 01_embeddings.py

    # Compute only specific embedding types
    python 01_embeddings.py --only esm2
    python 01_embeddings.py --only saprot
    python 01_embeddings.py --only substrates
    python 01_embeddings.py --only esm2,substrates

    # Custom input: embed new mutations from an arbitrary CSV
    python 01_embeddings.py --mutations-csv new_mutations.csv --output-dir output/

    # Custom input: embed new substrates from a JSON {name: smiles}
    python 01_embeddings.py --substrates-json new_substrates.json --output-dir output/

    # Override device / force recomputation
    python 01_embeddings.py --device cuda --force
"""

import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent        # code/round3/
PROJECT_ROOT = SCRIPT_DIR.parent.parent             # MDA_HACL/

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    """Load config.yaml."""
    path = Path(config_path) if config_path else SCRIPT_DIR / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {path}")
    return config


def load_wt_sequence(fasta_path: Path) -> str:
    """Parse a single-sequence FASTA file and return the sequence string."""
    lines = fasta_path.read_text().strip().split("\n")
    seq_lines = [line.strip() for line in lines if not line.startswith(">")]
    sequence = "".join(seq_lines)
    print(f"Loaded WT sequence ({len(sequence)} residues) from {fasta_path.name}")
    return sequence


def get_device(config: dict, override: Optional[str] = None) -> str:
    """Resolve device string. 'auto' selects best available."""
    import torch

    device = override or config.get("compute", {}).get("device", "cpu")

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")
    return device


def parse_mutation_string(mutation_string: str) -> Tuple[str, int, str]:
    """Parse 'V83A' into (wt_aa='V', position_1idx=83, mut_aa='A')."""
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation_string.strip())
    if m is None:
        raise ValueError(f"Cannot parse mutation string: '{mutation_string}'")
    return m.group(1), int(m.group(2)), m.group(3)


def position_1idx_to_0idx(position_1idx: int, position_offset: int = 10) -> int:
    """Convert 1-indexed experimental position to 0-indexed."""
    return position_1idx - position_offset - 1


def apply_mutation(wt_seq: str, mutation_string: str, position_offset: int = 10) -> str:
    """Apply a point mutation to the WT sequence, return full mutant sequence."""
    wt_aa, pos_1idx, mut_aa = parse_mutation_string(mutation_string)
    pos_0idx = position_1idx_to_0idx(pos_1idx, position_offset)

    if wt_seq[pos_0idx] != wt_aa:
        raise ValueError(
            f"Mutation {mutation_string}: expected {wt_aa} at position "
            f"{pos_0idx} but found {wt_seq[pos_0idx]}"
        )

    return wt_seq[:pos_0idx] + mut_aa + wt_seq[pos_0idx + 1:]


def collect_unique_mutations(
    processed_dir: Path,
) -> pd.DataFrame:
    """Load formaldehyde_ssm.csv and multi_substrate_ssm.csv, deduplicate mutations.

    Returns DataFrame with columns: mutation_string, position, wt_aa, mut_aa, variant
    """
    form_path = processed_dir / "formaldehyde_ssm.csv"
    multi_path = processed_dir / "multi_substrate_ssm.csv"

    dfs = []
    if form_path.exists():
        form_df = pd.read_csv(form_path)
        dfs.append(form_df[["mutation_string", "position", "wt_aa", "mut_aa", "variant"]])
    if multi_path.exists():
        multi_df = pd.read_csv(multi_path)
        dfs.append(multi_df[["mutation_string", "position", "wt_aa", "mut_aa", "variant"]])

    if not dfs:
        raise FileNotFoundError(
            f"No SSM data found in {processed_dir}. Run 00_data_wrangling.py first."
        )

    combined = pd.concat(dfs, ignore_index=True)
    unique = combined.drop_duplicates(subset=["mutation_string"]).reset_index(drop=True)
    print(f"Collected {len(unique)} unique mutations from {len(combined)} total rows")
    return unique


def load_mutations_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load a custom mutations CSV with at minimum mutation_string and variant columns."""
    df = pd.read_csv(csv_path)
    required = {"mutation_string", "variant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Custom mutations CSV missing columns: {missing}")

    # Parse position and AA info if not present
    if "position" not in df.columns or "wt_aa" not in df.columns:
        parsed = df["mutation_string"].apply(
            lambda s: pd.Series(parse_mutation_string(s), index=["wt_aa", "pos_1idx", "mut_aa"])
        )
        df["wt_aa"] = parsed["wt_aa"]
        df["mut_aa"] = parsed["mut_aa"]
        # We need a position_offset; assume 10 (MDA_HACL convention)
        df["position"] = parsed["pos_1idx"].apply(lambda p: position_1idx_to_0idx(p, 10))

    unique = df.drop_duplicates(subset=["mutation_string"]).reset_index(drop=True)
    print(f"Loaded {len(unique)} unique mutations from {csv_path.name}")
    return unique


def verify_mutation_positions(mutations_df: pd.DataFrame, wt_seq: str) -> None:
    """Verify that position, wt_aa, and mut_aa are consistent with variant sequences.

    For each mutation, checks:
      1. wt_seq[position] == wt_aa  (position maps to expected WT residue)
      2. variant[position] == mut_aa (variant actually has the mutation at that position)
      3. variant differs from WT only at position (single-point mutation)
    """
    n_errors = 0
    for _, row in mutations_df.iterrows():
        pos = row["position"]
        ms = row["mutation_string"]
        wt_aa = row["wt_aa"]
        mut_aa = row["mut_aa"]
        variant = row["variant"]

        # Check 1: WT sequence has expected AA at position
        if wt_seq[pos] != wt_aa:
            print(f"  ERROR {ms}: wt_seq[{pos}]={wt_seq[pos]} != expected wt_aa={wt_aa}")
            n_errors += 1
            continue

        # Check 2: variant has the mutant AA at position
        if variant[pos] != mut_aa:
            print(f"  ERROR {ms}: variant[{pos}]={variant[pos]} != expected mut_aa={mut_aa}")
            n_errors += 1
            continue

        # Check 3: variant differs from WT at exactly one position
        diffs = [i for i in range(len(wt_seq)) if wt_seq[i] != variant[i]]
        if wt_aa == mut_aa:
            # Synonymous: variant should equal WT
            if len(diffs) != 0:
                print(f"  ERROR {ms}: synonymous but variant differs at positions {diffs}")
                n_errors += 1
        else:
            if diffs != [pos]:
                print(f"  ERROR {ms}: expected diff at [{pos}] only, got {diffs}")
                n_errors += 1

    if n_errors > 0:
        raise ValueError(
            f"Position verification failed: {n_errors}/{len(mutations_df)} mutations "
            f"have inconsistent position/variant/wt_aa/mut_aa data"
        )
    print(f"  Position verification passed: all {len(mutations_df)} mutations consistent")


# ---------------------------------------------------------------------------
# ESM2 embeddings
# ---------------------------------------------------------------------------

def compute_esm2_sequence(
    sequence: str,
    embedder,
    label: str = "sequence",
) -> np.ndarray:
    """Embed a single sequence and return per-residue embeddings.

    Returns: (seq_len, embedding_dim) numpy array
    """
    from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences

    seqs = ProteinSequences([ProteinSequence(sequence)])
    emb = embedder.transform(seqs)  # (1, seq_len, dim)
    emb = emb.squeeze(0)  # (seq_len, dim)
    print(f"  ESM2 {label}: shape {emb.shape}")
    return emb


def compute_esm2_mutants(
    mutations_df: pd.DataFrame,
    embedder,
) -> Dict[str, np.ndarray]:
    """Compute ESM2 embedding at the mutated position for each unique mutation.

    Groups mutations by position for efficient batching.
    Returns: dict {mutation_string: (embedding_dim,) numpy array}
    """
    from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences

    result = {}
    grouped = mutations_df.groupby("position")
    n_positions = len(grouped)

    for i, (pos_0idx, group) in enumerate(grouped):
        print(f"  ESM2 mutants: position {pos_0idx} "
              f"({i + 1}/{n_positions}, {len(group)} variants)")

        sequences = ProteinSequences([
            ProteinSequence(row["variant"])
            for _, row in group.iterrows()
        ])

        emb = embedder.transform(sequences)  # (n_variants, seq_len, dim)
        # Extract embedding at the mutated position
        pos_emb = emb[:, pos_0idx, :]  # (n_variants, dim)

        for j, (_, row) in enumerate(group.iterrows()):
            result[row["mutation_string"]] = pos_emb[j]

    print(f"  ESM2 mutants: {len(result)} total embeddings, "
          f"dim={next(iter(result.values())).shape[0]}")
    return result


def run_esm2(
    wt_seq: str,
    mutations_df: pd.DataFrame,
    config: dict,
    device: str,
    output_dir: Path,
    force: bool = False,
    pyruvate_ref_mutation: str = "V83P",
    position_offset: int = 10,
):
    """Run all ESM2 embedding computations."""
    from aide_predict.bespoke_models import ESM2Embedding
    from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences

    model_name = config["embeddings"]["esm2"]["model_name"]
    batch_size = config.get("compute", {}).get("esm2_batch_size", 4)

    wt_path = output_dir / "esm2_wt_residues.npy"
    v83p_path = output_dir / "esm2_v83p_residues.npy"
    mutant_path = output_dir / "esm2_mutant_residues.npz"

    # Keep model on device across calls for efficiency
    os.environ["KEEP_MODEL_ON_DEVICE"] = "True"

    embedder = ESM2Embedding(
        model_checkpoint=model_name,
        pool=False,
        flatten=False,
        device=device,
        batch_size=batch_size,
    )
    # fit() is a no-op for ESM2 but required by sklearn API
    embedder.fit(ProteinSequences([ProteinSequence(wt_seq)]))

    # WT embeddings
    if force or not wt_path.exists():
        print("\nComputing ESM2 WT residue embeddings...")
        t0 = time.time()
        wt_emb = compute_esm2_sequence(wt_seq, embedder, label="WT")
        np.save(wt_path, wt_emb)
        print(f"  Saved to {wt_path.name} ({time.time() - t0:.1f}s)")
    else:
        print(f"\nSkipping ESM2 WT (exists: {wt_path.name})")
        wt_emb = np.load(wt_path)

    # V83P embeddings (pyruvate reference)
    if force or not v83p_path.exists():
        print("\nComputing ESM2 V83P residue embeddings...")
        t0 = time.time()
        v83p_seq = apply_mutation(wt_seq, pyruvate_ref_mutation, position_offset)
        v83p_emb = compute_esm2_sequence(v83p_seq, embedder, label="V83P")
        np.save(v83p_path, v83p_emb)
        print(f"  Saved to {v83p_path.name} ({time.time() - t0:.1f}s)")
    else:
        print(f"\nSkipping ESM2 V83P (exists: {v83p_path.name})")

    # Mutant embeddings
    if force or not mutant_path.exists():
        print(f"\nComputing ESM2 mutant embeddings ({len(mutations_df)} mutations)...")
        t0 = time.time()
        mutant_embs = compute_esm2_mutants(mutations_df, embedder)
        np.savez_compressed(mutant_path, **mutant_embs)
        print(f"  Saved to {mutant_path.name} ({time.time() - t0:.1f}s)")
    else:
        print(f"\nSkipping ESM2 mutants (exists: {mutant_path.name})")

    # Clean up env var
    os.environ.pop("KEEP_MODEL_ON_DEVICE", None)


# ---------------------------------------------------------------------------
# SaProt zero-shot scores
# ---------------------------------------------------------------------------

def run_saprot(
    wt_seq: str,
    mutations_df: pd.DataFrame,
    config: dict,
    device: str,
    output_dir: Path,
    force: bool = False,
):
    """Compute SaProt wildtype-marginal zero-shot scores."""
    from aide_predict.bespoke_models import SaProtLikelihoodWrapper
    from aide_predict.utils.data_structures import (
        ProteinSequence,
        ProteinSequences,
        ProteinStructure,
    )

    saprot_path = output_dir / "saprot_scores.json"

    if not force and saprot_path.exists():
        print(f"\nSkipping SaProt (exists: {saprot_path.name})")
        return

    # Load structure
    structure_file = PROJECT_ROOT / config["data"]["structure_file"]
    if not structure_file.exists():
        raise FileNotFoundError(
            f"Structure file not found: {structure_file}\n"
            "SaProt requires a PDB structure. Use --skip-saprot to skip."
        )

    print(f"\nComputing SaProt zero-shot scores ({len(mutations_df)} mutations)...")
    t0 = time.time()

    structure = ProteinStructure(structure_file=str(structure_file))
    wt_protein = ProteinSequence(wt_seq, structure=structure)

    marginal = config["embeddings"]["saprot"]["marginal"]
    marginal_method = f"{marginal}_marginal"  # "wildtype" -> "wildtype_marginal"
    batch_size = config.get("compute", {}).get("saprot_batch_size", 2)

    scorer = SaProtLikelihoodWrapper(
        model_checkpoint="westlake-repl/SaProt_650M_AF2",
        marginal_method=marginal_method,
        wt=wt_protein,
        pool=True,
        flatten=True,
        device=device,
        batch_size=batch_size,
    )

    # Build mutant ProteinSequences
    mutant_seqs = ProteinSequences([
        ProteinSequence(row["variant"])
        for _, row in mutations_df.iterrows()
    ])

    scorer.fit(mutant_seqs)
    scores = scorer.predict(mutant_seqs)  # (n_mutants,) numpy

    # Map to mutation_string
    score_dict = {}
    for i, (_, row) in enumerate(mutations_df.iterrows()):
        score_dict[row["mutation_string"]] = float(scores[i])

    with open(saprot_path, "w") as f:
        json.dump(score_dict, f, indent=2)

    elapsed = time.time() - t0
    print(f"  SaProt: {len(score_dict)} scores computed ({elapsed:.1f}s)")
    print(f"  Score range: [{min(score_dict.values()):.4f}, {max(score_dict.values()):.4f}]")
    print(f"  Saved to {saprot_path.name}")


# ---------------------------------------------------------------------------
# Substrate embeddings
# ---------------------------------------------------------------------------

def compute_substrate_morgan(
    substrate_smiles: Dict[str, str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprints for all substrates.

    Returns: (n_substrates, n_bits) numpy array
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fps = []
    for name, smiles in substrate_smiles.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for {name}: {smiles}")
        fp = gen.GetFingerprintAsNumPy(mol)
        fps.append(fp.astype(np.float32))

    result = np.stack(fps)
    print(f"  Morgan: {result.shape} (radius={radius}, bits={n_bits})")
    return result


def compute_substrate_maccs(substrate_smiles: Dict[str, str]) -> np.ndarray:
    """Compute MACCS keys for all substrates.

    Returns: (n_substrates, 166) numpy array
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys

    fps = []
    for name, smiles in substrate_smiles.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for {name}: {smiles}")
        maccs = MACCSkeys.GenMACCSKeys(mol)
        # Convert to numpy, then remove unused first bit
        arr = np.zeros(maccs.GetNumBits(), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(maccs, arr)
        fps.append(arr[1:])  # remove unused bit 0 → 166 dims

    result = np.stack(fps)
    print(f"  MACCS: {result.shape}")
    return result


def compute_substrate_mordred(
    substrate_smiles: Dict[str, str],
    existing_feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute Mordred 2D descriptors for all substrates.

    If existing_feature_names is provided (e.g., from training), only those
    features are kept (for consistency between training and inference).

    Returns: (array shape (n_substrates, D), list of kept feature names)
    """
    from mordred import Calculator, descriptors
    from rdkit import Chem

    calc = Calculator(descriptors, ignore_3D=True)

    all_results = []
    for name, smiles in substrate_smiles.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for {name}: {smiles}")
        result = calc(mol)
        all_results.append(result)

    # Get feature names from first result
    feature_names = [str(d) for d in calc.descriptors]

    # Convert to array
    raw_array = np.array(
        [[float(v) if not isinstance(v, Exception) else np.nan for v in r.values()]
         for r in all_results],
        dtype=np.float32,
    )

    if existing_feature_names is not None:
        # Select only the features from training
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        indices = [name_to_idx[n] for n in existing_feature_names if n in name_to_idx]
        result = raw_array[:, indices]
        kept_names = [feature_names[i] for i in indices]
        n_dropped = len(existing_feature_names) - len(kept_names)
        if n_dropped > 0:
            warnings.warn(f"Mordred: {n_dropped} training features not found in new data")
    else:
        # Remove columns with any NaN or non-finite values
        valid_mask = np.isfinite(raw_array).all(axis=0)
        result = raw_array[:, valid_mask]
        kept_names = [n for n, v in zip(feature_names, valid_mask) if v]
        n_dropped = (~valid_mask).sum()
        print(f"  Mordred: dropped {n_dropped}/{len(feature_names)} features with NaN/Inf")

    print(f"  Mordred: {result.shape} ({len(kept_names)} features kept)")
    return result, kept_names


def compute_substrate_molformer(
    substrate_smiles: Dict[str, str],
    device: str = "cpu",
) -> np.ndarray:
    """Compute MoLFormer-XL embeddings for all substrates.

    Uses the pretrained IBM MoLFormer-XL model (768-dim, trained on ~110M molecules).

    Returns: (n_substrates, 768) numpy array
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "ibm/MoLFormer-XL-both-10pct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, deterministic_eval=True, trust_remote_code=True
    )
    model.eval()
    model.to(device)

    smiles_list = list(substrate_smiles.values())
    inputs = tokenizer(smiles_list, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.pooler_output.cpu().numpy().astype(np.float32)

    print(f"  MoLFormer: {emb.shape} (model={model_name})")
    return emb


def run_substrates(
    config: dict,
    output_dir: Path,
    device: str = "cpu",
    force: bool = False,
    custom_substrates: Optional[Dict[str, str]] = None,
):
    """Run all substrate embedding computations."""
    morgan_path = output_dir / "substrate_morgan.npy"
    maccs_path = output_dir / "substrate_maccs.npy"
    mordred_path = output_dir / "substrate_mordred.npy"
    molformer_path = output_dir / "substrate_molformer.npy"
    names_path = output_dir / "substrate_names.json"
    mordred_names_path = output_dir / "mordred_feature_names.json"

    all_paths = [morgan_path, maccs_path, mordred_path, molformer_path]
    if not force and all(p.exists() for p in all_paths):
        print(f"\nSkipping substrates (all files exist)")
        return

    # Get substrate SMILES
    if custom_substrates:
        substrate_smiles = custom_substrates
    else:
        metadata_path = PROJECT_ROOT / config["data"]["output_dir"] / "substrate_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        substrate_smiles = {name: info["smiles"] for name, info in metadata.items()}

    substrate_names = list(substrate_smiles.keys())
    print(f"\nComputing substrate embeddings for {len(substrate_names)} substrates...")

    emb_config = config["embeddings"]["substrate"]

    # Morgan
    morgan = compute_substrate_morgan(
        substrate_smiles,
        radius=emb_config.get("morgan_radius", 2),
        n_bits=emb_config.get("morgan_bits", 2048),
    )
    np.save(morgan_path, morgan)

    # MACCS
    maccs = compute_substrate_maccs(substrate_smiles)
    np.save(maccs_path, maccs)

    # Mordred
    # If running on new substrates, try to load existing feature names for consistency
    existing_names = None
    if custom_substrates:
        training_mordred_names = (
            PROJECT_ROOT / config["data"]["output_dir"] / "embeddings" / "mordred_feature_names.json"
        )
        if training_mordred_names.exists():
            with open(training_mordred_names) as f:
                existing_names = json.load(f)
            print(f"  Using {len(existing_names)} feature names from training embeddings")

    mordred, feature_names = compute_substrate_mordred(substrate_smiles, existing_names)
    np.save(mordred_path, mordred)

    # MoLFormer
    molformer = compute_substrate_molformer(substrate_smiles, device=device)
    np.save(molformer_path, molformer)

    # Save names
    with open(names_path, "w") as f:
        json.dump(substrate_names, f, indent=2)
    with open(mordred_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"  Saved to {output_dir.name}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute embeddings for MDA HACL Round 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_embeddings.py                                  # compute everything
  python 01_embeddings.py --only esm2                      # ESM2 only
  python 01_embeddings.py --only substrates --force        # recompute substrates
  python 01_embeddings.py --substrates-json new.json --output-dir out/
  python 01_embeddings.py --mutations-csv new.csv --output-dir out/
  python 01_embeddings.py --device cuda                    # use GPU
        """,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: config.yaml in script dir)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Compute only specific types: esm2, saprot, substrates (comma-separated)",
    )
    parser.add_argument(
        "--mutations-csv", type=str, default=None,
        help="Custom mutations CSV (must have mutation_string, variant columns)",
    )
    parser.add_argument(
        "--substrates-json", type=str, default=None,
        help='Custom substrates JSON: {"name": "SMILES", ...}',
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device: cpu, cuda, mps, auto",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if output files exist",
    )
    parser.add_argument(
        "--skip-saprot", action="store_true",
        help="Skip SaProt computation (useful if foldseek unavailable)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t_start = time.time()

    # 1. Load config
    config = load_config(args.config)
    device = get_device(config, args.device)

    # 2. Resolve what to compute
    compute_all = args.only is None
    if args.only:
        steps = set(s.strip() for s in args.only.split(","))
    else:
        steps = {"esm2", "saprot", "substrates"}

    # 3. Resolve output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / config["data"]["output_dir"] / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 4. Load WT sequence
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    wt_fasta = processed_dir / "wt_sequence.fasta"
    wt_seq = load_wt_sequence(wt_fasta)

    position_offset = config["data"]["position_offset"]
    pyruvate_ref = config["data"]["pyruvate_ref_mutation"]

    # 5. Load mutations
    if args.mutations_csv:
        mutations_df = load_mutations_from_csv(Path(args.mutations_csv))
    else:
        mutations_df = collect_unique_mutations(processed_dir)

    # 5b. Verify position mapping is correct
    if "esm2" in steps or "saprot" in steps:
        print("\nVerifying mutation positions...")
        verify_mutation_positions(mutations_df, wt_seq)

    # 6. ESM2
    if "esm2" in steps:
        run_esm2(
            wt_seq=wt_seq,
            mutations_df=mutations_df,
            config=config,
            device=device,
            output_dir=output_dir,
            force=args.force,
            pyruvate_ref_mutation=pyruvate_ref,
            position_offset=position_offset,
        )

    # 7. SaProt
    if "saprot" in steps and not args.skip_saprot:
        run_saprot(
            wt_seq=wt_seq,
            mutations_df=mutations_df,
            config=config,
            device=device,
            output_dir=output_dir,
            force=args.force,
        )
    elif "saprot" in steps and args.skip_saprot:
        print("\nSkipping SaProt (--skip-saprot flag)")

    # 8. Substrates
    if "substrates" in steps:
        custom_substrates = None
        if args.substrates_json:
            with open(args.substrates_json) as f:
                custom_substrates = json.load(f)

        run_substrates(
            config=config,
            output_dir=output_dir,
            device=device,
            force=args.force,
            custom_substrates=custom_substrates,
        )

    # 9. Save metadata
    metadata_path = output_dir / "embedding_metadata.json"
    elapsed = time.time() - t_start

    metadata = {
        "device": device,
        "esm2_model": config["embeddings"]["esm2"]["model_name"],
        "n_unique_mutations": len(mutations_df),
        "wt_length": len(wt_seq),
        "steps_computed": sorted(steps),
        "force": args.force,
        "elapsed_seconds": round(elapsed, 1),
        "custom_mutations": args.mutations_csv is not None,
        "custom_substrates": args.substrates_json is not None,
    }

    # Add dimension info from saved files
    wt_path = output_dir / "esm2_wt_residues.npy"
    if wt_path.exists():
        wt_emb = np.load(wt_path)
        metadata["esm2_dim"] = int(wt_emb.shape[1])
        metadata["esm2_wt_shape"] = list(wt_emb.shape)

    mutant_path = output_dir / "esm2_mutant_residues.npz"
    if mutant_path.exists():
        with np.load(mutant_path) as data:
            metadata["n_mutant_embeddings"] = len(data.files)

    saprot_path = output_dir / "saprot_scores.json"
    if saprot_path.exists():
        with open(saprot_path) as f:
            scores = json.load(f)
        metadata["n_saprot_scores"] = len(scores)

    morgan_path = output_dir / "substrate_morgan.npy"
    if morgan_path.exists():
        metadata["substrate_morgan_shape"] = list(np.load(morgan_path).shape)
    maccs_path = output_dir / "substrate_maccs.npy"
    if maccs_path.exists():
        metadata["substrate_maccs_shape"] = list(np.load(maccs_path).shape)
    mordred_path = output_dir / "substrate_mordred.npy"
    if mordred_path.exists():
        metadata["substrate_mordred_shape"] = list(np.load(mordred_path).shape)
    molformer_path = output_dir / "substrate_molformer.npy"
    if molformer_path.exists():
        metadata["substrate_molformer_shape"] = list(np.load(molformer_path).shape)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # 10. Summary
    print(f"\n{'=' * 60}")
    print(f"Embedding computation complete ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    for key, val in sorted(metadata.items()):
        if key not in ("device", "elapsed_seconds", "custom_mutations", "custom_substrates"):
            print(f"  {key}: {val}")
    print()


if __name__ == "__main__":
    main()
