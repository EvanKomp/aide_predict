#!/usr/bin/env python
"""
00_data_wrangling.py — Data wrangling for MDA HACL Round 3
==========================================================

Loads raw experimental data:
  - Formaldehyde SSM (64 positions, ~1,271 mutations) from ssm_data.csv
  - Multi-substrate SSM (10 positions, 6 active + 3 inactive substrates) from all_ssm_data.xlsx
  - Supplemental multi-substrate SSM (5 additional positions) from all_data2.xlsx

Outputs clean, standardized CSVs to data/round3/processed/:
  - formaldehyde_ssm.csv
  - multi_substrate_ssm.csv
  - substrate_metadata.json
  - wt_sequence.fasta

Usage:
    python 00_data_wrangling.py
"""

import os
import re
import json
import hashlib
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Canonical SMILES from PubChem for all 9 substrates
SUBSTRATE_SMILES: Dict[str, Optional[str]] = {
    # Active
    "Formaldehyde":        "C=O",
    "Acetaldehyde":        "CC=O",
    "Acetone":             "CC(C)=O",
    "Glycoaldehyde":       "OCC=O",
    "Phenylacetaldehyde":  "O=CCc1ccccc1",
    "Pyruvate":            "CC(=O)C(=O)O",
    # Inactive
    "Methylglyoxal":       "CC(=O)C=O",
    "4-Hydroxybutan-2-one": "CC(=O)CCO",
    "Glyoxylic acid":      "OC(=O)C=O",
}

# Column name normalization for supplemental data files
SUPPLEMENTAL_SUBSTRATE_NAME_MAP: Dict[str, str] = {
    "Glyoxic Acid": "Glyoxylic acid",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config.yaml from the same directory as this script."""
    config_path = SCRIPT_DIR / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    return config


def parse_mutation_string(mutation_string: str) -> Tuple[str, int, str]:
    """Parse 'V83A' into (wt_aa='V', position_1idx=83, mut_aa='A')."""
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation_string.strip())
    if m is None:
        raise ValueError(f"Cannot parse mutation string: '{mutation_string}'")
    return m.group(1), int(m.group(2)), m.group(3)


def position_1idx_to_0idx(position_1idx: int, position_offset: int = 10) -> int:
    """Convert 1-indexed experimental position to 0-indexed sequence position.

    0-indexed = 1-indexed - offset - 1
    With offset=10: position 35 -> index 24, position 83 -> index 72.
    """
    return position_1idx - position_offset - 1


def position_0idx_to_1idx(position_0idx: int, position_offset: int = 10) -> int:
    """Inverse of position_1idx_to_0idx."""
    return position_0idx + position_offset + 1


def extract_wt_sequence(ssm_df: pd.DataFrame) -> str:
    """Extract WT sequence from synonymous mutations in ssm_data.csv.

    Synonymous mutations (e.g. V35V) carry the unmodified WT sequence
    in their 'variant' column.
    """
    synonymous = ssm_df[ssm_df["original_aa"] == ssm_df["mutation"]]
    if len(synonymous) == 0:
        raise ValueError("No synonymous mutations found in ssm_data.csv")

    wt_sequence = synonymous.iloc[0]["variant"]

    # Verify all synonymous mutations agree on the WT sequence
    mismatches = synonymous[synonymous["variant"] != wt_sequence]
    if len(mismatches) > 0:
        raise ValueError(
            f"{len(mismatches)} synonymous mutations have different variant "
            f"sequences — cannot reliably extract WT"
        )

    print(f"Extracted WT sequence ({len(wt_sequence)} residues) "
          f"from {len(synonymous)} synonymous mutations")
    return wt_sequence


def protein_hash(sequence: str) -> str:
    """MD5 hash of a protein sequence."""
    return hashlib.md5(sequence.encode()).hexdigest()


def apply_mutation(
    wt_sequence: str,
    mutation_string: str,
    position_offset: int = 10,
) -> str:
    """Apply a single mutation to the WT sequence, returning the full mutant sequence."""
    wt_aa, pos_1idx, mut_aa = parse_mutation_string(mutation_string)
    pos_0idx = position_1idx_to_0idx(pos_1idx, position_offset)

    if pos_0idx < 0 or pos_0idx >= len(wt_sequence):
        raise ValueError(
            f"Position {pos_1idx} (0-idx: {pos_0idx}) out of bounds "
            f"for sequence of length {len(wt_sequence)}"
        )

    actual_aa = wt_sequence[pos_0idx]
    if actual_aa != wt_aa:
        raise ValueError(
            f"Mutation {mutation_string}: expected {wt_aa} at position "
            f"{pos_1idx} (0-idx: {pos_0idx}), found {actual_aa}"
        )

    seq = list(wt_sequence)
    seq[pos_0idx] = mut_aa
    return "".join(seq)


def compute_log_fc(fold_change: pd.Series, epsilon: float = 0.01) -> pd.Series:
    """Compute log10(fold_change + epsilon)."""
    return np.log10(fold_change + epsilon)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_formaldehyde_ssm(config: dict, wt_sequence: str) -> pd.DataFrame:
    """Load and clean the full 64-position formaldehyde SSM dataset.

    Returns DataFrame with columns:
        position (0-indexed), wt_aa, mut_aa, mutation_string,
        fold_change, log_fc, variant, hash
    """
    csv_path = PROJECT_ROOT / config["data"]["formaldehyde_ssm_csv"]
    print(f"\nLoading formaldehyde SSM from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    epsilon = config["data"]["epsilon"]

    # Rename to standard schema
    # Drop the 1-indexed 'position' column first to avoid duplicate names
    df = df.drop(columns=["position"])
    df = df.rename(columns={
        "activity": "fold_change",
        "position_on_wt_0_indexed": "position",
        "original_aa": "wt_aa",
        "mutation": "mut_aa",
    })

    # Compute log target
    df["log_fc"] = compute_log_fc(df["fold_change"], epsilon)

    # Select and reorder
    df = df[["position", "wt_aa", "mut_aa", "mutation_string",
             "fold_change", "log_fc", "variant", "hash"]].copy()

    # Validate
    assert (df["fold_change"] >= 0).all(), "Negative fold_change values found"
    assert (df["variant"].str.len() == len(wt_sequence)).all(), \
        "Variant sequences have inconsistent lengths"
    assert df["position"].between(0, len(wt_sequence) - 1).all(), \
        "Position values out of bounds"

    n_zeros = (df["fold_change"] == 0).sum()
    n_positions = df["position"].nunique()
    print(f"  Cleaned: {len(df)} mutations at {n_positions} positions")
    print(f"  Fold-change: min={df['fold_change'].min():.3f}, "
          f"max={df['fold_change'].max():.3f}, "
          f"median={df['fold_change'].median():.3f}")
    print(f"  Zero-activity mutations: {n_zeros} ({100*n_zeros/len(df):.1f}%)")

    return df


def load_single_substrate_sheet(
    xlsx_path: Path,
    sheet_name: str,
    wt_sequence: str,
    position_offset: int,
    epsilon: float,
) -> pd.DataFrame:
    """Load one substrate sheet from all_ssm_data.xlsx.

    Returns DataFrame with columns:
        substrate, position (0-indexed), wt_aa, mut_aa, mutation_string,
        fold_change, log_fc, ref_type, variant, hash
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Standardize column names (first col = mutation, second col = activity)
    cols = df.columns.tolist()
    df = df.rename(columns={cols[0]: "mutation_string", cols[1]: "fold_change"})

    # Drop empty/NaN rows
    initial_len = len(df)
    df = df.dropna(subset=["mutation_string"])
    df["mutation_string"] = df["mutation_string"].astype(str).str.strip()
    df = df[df["mutation_string"] != ""]

    # Drop NaN activities (e.g. T561V, S569V in Formaldehyde) with warning
    nan_activity = df[df["fold_change"].isna()]
    if len(nan_activity) > 0:
        dropped = nan_activity["mutation_string"].tolist()
        warnings.warn(
            f"{sheet_name}: dropping {len(nan_activity)} mutations with NaN "
            f"activity: {dropped}"
        )
        df = df.dropna(subset=["fold_change"])

    dropped_total = initial_len - len(df)
    if dropped_total > 0:
        print(f"  {sheet_name}: dropped {dropped_total} rows "
              f"(NaN/empty), kept {len(df)}")

    # Parse mutations
    parsed = df["mutation_string"].apply(parse_mutation_string)
    df["wt_aa"] = parsed.apply(lambda x: x[0])
    df["position_1idx"] = parsed.apply(lambda x: x[1])
    df["mut_aa"] = parsed.apply(lambda x: x[2])
    df["position"] = df["position_1idx"].apply(
        lambda p: position_1idx_to_0idx(p, position_offset)
    )

    # Determine reference type
    ref_type = "v83p" if sheet_name == "Pyruvate" else "wt"
    df["ref_type"] = ref_type

    # Apply mutations to WT to get variant sequences
    df["variant"] = df["mutation_string"].apply(
        lambda ms: apply_mutation(wt_sequence, ms, position_offset)
    )
    df["hash"] = df["variant"].apply(protein_hash)

    # Substrate name
    df["substrate"] = sheet_name

    # Log transform
    df["log_fc"] = compute_log_fc(df["fold_change"], epsilon)

    # Select columns
    df = df[["substrate", "position", "wt_aa", "mut_aa", "mutation_string",
             "fold_change", "log_fc", "ref_type", "variant", "hash"]].copy()

    return df


def load_supplemental_wide_format_xlsx(
    xlsx_path: Path,
    wt_sequence: str,
    position_offset: int,
    epsilon: float,
) -> pd.DataFrame:
    """Load supplemental wide-format SSM data and convert to long format.

    The xlsx has a metadata row at index 0 and headers at row 1:
        Mutation, Formaldehyde, Acetaldehyde, ..., Glyoxic Acid

    Returns DataFrame with same schema as load_single_substrate_sheet().
    """
    df = pd.read_excel(xlsx_path, sheet_name=0, header=1)

    # Rename mutation column
    df = df.rename(columns={"Mutation": "mutation_string"})
    df["mutation_string"] = df["mutation_string"].astype(str).str.strip()
    df = df[df["mutation_string"] != ""]

    # Normalize substrate column names
    df = df.rename(columns=SUPPLEMENTAL_SUBSTRATE_NAME_MAP)

    # Identify substrate columns (everything except mutation_string)
    substrate_cols = [c for c in df.columns if c != "mutation_string"]

    # Melt wide -> long
    long_df = df.melt(
        id_vars=["mutation_string"],
        value_vars=substrate_cols,
        var_name="substrate",
        value_name="fold_change",
    )

    # Drop NaN fold_change rows
    n_before = len(long_df)
    long_df = long_df.dropna(subset=["fold_change"])
    n_dropped = n_before - len(long_df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with NaN fold_change")

    # Parse mutations
    parsed = long_df["mutation_string"].apply(parse_mutation_string)
    long_df["wt_aa"] = parsed.apply(lambda x: x[0])
    long_df["position"] = parsed.apply(
        lambda x: position_1idx_to_0idx(x[1], position_offset)
    )
    long_df["mut_aa"] = parsed.apply(lambda x: x[2])

    # Reference type
    long_df["ref_type"] = long_df["substrate"].apply(
        lambda s: "v83p" if s == "Pyruvate" else "wt"
    )

    # Variant sequences and hashes
    long_df["variant"] = long_df["mutation_string"].apply(
        lambda ms: apply_mutation(wt_sequence, ms, position_offset)
    )
    long_df["hash"] = long_df["variant"].apply(protein_hash)

    # Log transform
    long_df["log_fc"] = compute_log_fc(long_df["fold_change"], epsilon)

    # Select and reorder columns
    long_df = long_df[["substrate", "position", "wt_aa", "mut_aa", "mutation_string",
                        "fold_change", "log_fc", "ref_type", "variant", "hash"]].copy()

    return long_df


def create_inactive_substrate_data(
    substrate_name: str,
    multi_substrate_positions: List[int],
    wt_sequence: str,
    position_offset: int,
    epsilon: float,
) -> pd.DataFrame:
    """Create all-zero activity data for an inactive substrate.

    Generates len(multi_substrate_positions) * 20 rows with fold_change = 0,
    one per (position, AA) pair.
    """
    rows = []
    for pos_1idx in multi_substrate_positions:
        pos_0idx = position_1idx_to_0idx(pos_1idx, position_offset)
        wt_aa = wt_sequence[pos_0idx]

        for aa in STANDARD_AAS:
            mutation_string = f"{wt_aa}{pos_1idx}{aa}"
            variant = apply_mutation(wt_sequence, mutation_string, position_offset)
            rows.append({
                "substrate": substrate_name,
                "position": pos_0idx,
                "wt_aa": wt_aa,
                "mut_aa": aa,
                "mutation_string": mutation_string,
                "fold_change": 0.0,
                "log_fc": compute_log_fc(pd.Series([0.0]), epsilon).iloc[0],
                "ref_type": "wt",
                "variant": variant,
                "hash": protein_hash(variant),
            })

    df = pd.DataFrame(rows)
    return df


def load_multi_substrate_ssm(config: dict, wt_sequence: str) -> pd.DataFrame:
    """Load all multi-substrate SSM data (6 active + 3 inactive substrates).

    Returns long-format DataFrame with columns:
        substrate, position (0-indexed), wt_aa, mut_aa, mutation_string,
        fold_change, log_fc, is_active_substrate, ref_type, variant, hash
    """
    xlsx_path = PROJECT_ROOT / config["data"]["multi_substrate_ssm_xlsx"]
    position_offset = config["data"]["position_offset"]
    epsilon = config["data"]["epsilon"]
    config_positions = config["data"]["multi_substrate_positions"]

    print(f"\nLoading multi-substrate SSM from {xlsx_path}")

    dfs = []

    # Active substrates (from xlsx sheets)
    for sheet_name in config["data"]["active_substrate_sheets"]:
        print(f"  Loading sheet: {sheet_name}")
        df = load_single_substrate_sheet(
            xlsx_path, sheet_name, wt_sequence, position_offset, epsilon
        )
        df["is_active_substrate"] = True
        df["is_supplemental"] = False
        dfs.append(df)
        print(f"    {len(df)} mutations, "
              f"{(df['fold_change'] > 0).sum()} with activity > 0")

    # Supplemental data (wide-format xlsx with additional positions)
    supplemental_xlsx_key = config["data"].get("supplemental_ssm_xlsx")
    if supplemental_xlsx_key:
        supp_path = PROJECT_ROOT / supplemental_xlsx_key
        if supp_path.exists():
            print(f"\n  Loading supplemental data from {supp_path}")
            supp_df = load_supplemental_wide_format_xlsx(
                supp_path, wt_sequence, position_offset, epsilon
            )

            active_names = set(config["data"]["active_substrate_sheets"])
            inactive_names = set(config["data"]["inactive_substrates"])

            for substrate in supp_df["substrate"].unique():
                sub = supp_df[supp_df["substrate"] == substrate].copy()
                if substrate in active_names:
                    sub["is_active_substrate"] = True
                    sub["is_supplemental"] = True
                    dfs.append(sub)
                    print(f"    {substrate}: {len(sub)} supplemental mutations (active)")
                elif substrate in inactive_names:
                    sub["is_active_substrate"] = False
                    sub["is_supplemental"] = True
                    dfs.append(sub)
                    print(f"    {substrate}: {len(sub)} supplemental mutations (inactive)")
                else:
                    warnings.warn(
                        f"Supplemental substrate '{substrate}' not in active or "
                        f"inactive lists — skipping"
                    )
        else:
            warnings.warn(f"Supplemental xlsx not found: {supp_path}")

    # Derive the actual set of positions covered by the loaded active-substrate
    # data. Inactive substrates are generated synthetically on this set so the
    # dataset is consistent when supplemental data is toggled on/off.
    active_positions_0idx = sorted({
        int(p)
        for d in dfs if bool(d["is_active_substrate"].iloc[0])
        for p in d["position"].unique()
    })
    active_positions_1idx = [
        position_0idx_to_1idx(p, position_offset) for p in active_positions_0idx
    ]

    config_set = set(config_positions)
    active_set = set(active_positions_1idx)
    if config_set != active_set:
        missing = sorted(config_set - active_set)
        extra = sorted(active_set - config_set)
        print(
            f"  Note: config.multi_substrate_positions ({len(config_set)}) "
            f"differs from positions actually loaded ({len(active_set)})."
        )
        if missing:
            print(
                f"    In config but not loaded (skipped for inactive "
                f"substrates): {missing}"
            )
        if extra:
            print(f"    Loaded but not in config: {extra}")

    # Inactive substrates (synthetic all-zero data)
    if config["data"]["include_inactive_substrates"]:
        for substrate_name in config["data"]["inactive_substrates"]:
            print(f"  Creating inactive substrate: {substrate_name}")
            df = create_inactive_substrate_data(
                substrate_name, active_positions_1idx, wt_sequence,
                position_offset, epsilon
            )
            df["is_active_substrate"] = False
            df["is_supplemental"] = False
            dfs.append(df)
            print(f"    {len(df)} mutations (all FC=0)")

    multi_df = pd.concat(dfs, ignore_index=True)

    # Deduplicate: prefer real data (loaded first) over synthetic
    before_dedup = len(multi_df)
    multi_df = multi_df.drop_duplicates(
        subset=["substrate", "mutation_string"], keep="first"
    )
    n_deduped = before_dedup - len(multi_df)
    if n_deduped > 0:
        print(f"  Deduplicated {n_deduped} rows (real data preferred over synthetic)")

    # Summary
    print(f"\nMulti-substrate summary:")
    print(f"  Total rows: {len(multi_df)}")
    print(f"  Substrates: {multi_df['substrate'].nunique()}")
    for substrate in multi_df["substrate"].unique():
        sub = multi_df[multi_df["substrate"] == substrate]
        n_active = (sub["fold_change"] > 0).sum()
        active_flag = "ACTIVE" if sub["is_active_substrate"].iloc[0] else "INACTIVE"
        print(f"    {substrate:25s} [{active_flag:8s}]: "
              f"{len(sub):4d} mutations, {n_active:3d} with FC>0, "
              f"FC range [{sub['fold_change'].min():.3f}, {sub['fold_change'].max():.3f}]")

    return multi_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_formaldehyde_overlap(
    formaldehyde_df: pd.DataFrame,
    multi_df: pd.DataFrame,
) -> None:
    """Cross-validate formaldehyde data between ssm_data.csv and xlsx."""
    print("\nCross-validating formaldehyde overlap...")

    multi_form = multi_df[multi_df["substrate"] == "Formaldehyde"].copy()

    merged = pd.merge(
        formaldehyde_df[["mutation_string", "fold_change"]],
        multi_form[["mutation_string", "fold_change"]],
        on="mutation_string",
        suffixes=("_csv", "_xlsx"),
    )

    if len(merged) == 0:
        warnings.warn("No overlapping mutations found between sources!")
        return

    diff = (merged["fold_change_csv"] - merged["fold_change_xlsx"]).abs()
    corr = merged["fold_change_csv"].corr(merged["fold_change_xlsx"])

    print(f"  Overlapping mutations: {len(merged)}")
    print(f"  Max absolute difference: {diff.max():.6f}")
    print(f"  Mean absolute difference: {diff.mean():.6f}")
    print(f"  Pearson correlation: {corr:.6f}")

    if diff.max() > 0.1:
        warnings.warn(
            f"Large activity discrepancy between csv and xlsx sources: "
            f"max diff = {diff.max():.4f}"
        )
    else:
        print("  Validation passed: activities are consistent between sources")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def create_substrate_metadata(
    multi_df: pd.DataFrame,
    config: dict,
) -> Dict:
    """Create metadata dictionary for each substrate."""
    metadata = {}

    for substrate in multi_df["substrate"].unique():
        sub = multi_df[multi_df["substrate"] == substrate]
        is_active = bool(sub["is_active_substrate"].iloc[0])
        ref_type = sub["ref_type"].iloc[0]

        metadata[substrate] = {
            "name": substrate,
            "smiles": SUBSTRATE_SMILES.get(substrate),
            "is_active": is_active,
            "ref_type": ref_type,
            "ref_sequence_name": "V83P" if ref_type == "v83p" else "WT",
            "n_mutations": int(len(sub)),
            "n_active_mutations": int((sub["fold_change"] > 0).sum()),
            "positions_0idx": sorted(sub["position"].unique().tolist()),
        }

    return metadata


def save_wt_fasta(wt_sequence: str, output_path: Path) -> None:
    """Save WT sequence as FASTA file."""
    with open(output_path, "w") as f:
        f.write(">MDA_HACL_WT\n")
        # Wrap at 80 characters
        for i in range(0, len(wt_sequence), 80):
            f.write(wt_sequence[i:i+80] + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("00_data_wrangling.py — MDA HACL Round 3")
    print("=" * 60)

    # 1. Load config
    config = load_config()

    # 2. Create output directory
    output_dir = PROJECT_ROOT / config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 3. Extract WT sequence from ssm_data.csv
    csv_path = PROJECT_ROOT / config["data"]["formaldehyde_ssm_csv"]
    assert csv_path.exists(), f"Input not found: {csv_path}"
    raw_ssm = pd.read_csv(csv_path)
    wt_sequence = extract_wt_sequence(raw_ssm)

    # 4. Validate WT sequence length
    expected_len = config["data"]["wt_sequence_length"]
    assert len(wt_sequence) == expected_len, \
        f"WT sequence length {len(wt_sequence)} != expected {expected_len}"

    # 5. Save WT FASTA
    fasta_path = output_dir / "wt_sequence.fasta"
    save_wt_fasta(wt_sequence, fasta_path)
    print(f"Saved WT sequence to {fasta_path}")

    # 6. Process formaldehyde SSM (64 positions)
    formaldehyde_df = load_formaldehyde_ssm(config, wt_sequence)
    formaldehyde_path = output_dir / "formaldehyde_ssm.csv"
    formaldehyde_df.to_csv(formaldehyde_path, index=False)
    print(f"Saved {formaldehyde_path} ({len(formaldehyde_df)} rows)")

    # 7. Process multi-substrate SSM (9 substrates, 10 positions)
    xlsx_path = PROJECT_ROOT / config["data"]["multi_substrate_ssm_xlsx"]
    assert xlsx_path.exists(), f"Input not found: {xlsx_path}"
    multi_df = load_multi_substrate_ssm(config, wt_sequence)
    multi_path = output_dir / "multi_substrate_ssm.csv"
    multi_df.to_csv(multi_path, index=False)
    print(f"Saved {multi_path} ({len(multi_df)} rows)")

    # 8. Cross-validate formaldehyde overlap
    validate_formaldehyde_overlap(formaldehyde_df, multi_df)

    # 9. Save substrate metadata
    metadata = create_substrate_metadata(multi_df, config)
    meta_path = output_dir / "substrate_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {meta_path}")

    # 10. Final summary
    print("\n" + "=" * 60)
    print("Data Wrangling Complete")
    print("=" * 60)
    print(f"Formaldehyde SSM: {len(formaldehyde_df)} mutations "
          f"at {formaldehyde_df['position'].nunique()} positions")
    print(f"Multi-substrate SSM: {len(multi_df)} total rows")

    n_active = multi_df[multi_df["is_active_substrate"]]["substrate"].nunique()
    n_inactive = multi_df[~multi_df["is_active_substrate"]]["substrate"].nunique()
    print(f"  Active substrates: {n_active}")
    print(f"  Inactive substrates: {n_inactive}")
    print(f"  Unique positions: {sorted(multi_df['position'].unique().tolist())}")

    print(f"\nOutput files:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
