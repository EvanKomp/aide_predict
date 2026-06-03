#!/usr/bin/env python
"""
fetch_smiles.py — Fetch canonical SMILES from PubChem for extra substrates.

Reads CIDs from extra_substrates.xlsx, queries PubChem REST API,
and saves {name: SMILES} JSON for use with 06_predict.py --substrate-smiles.

Usage:
    python fetch_smiles.py
"""

import json
from pathlib import Path

import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
XLSX_PATH = PROJECT_ROOT / "data" / "round3" / "extra_substrates.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "round3" / "extra_substrates_smiles.json"


def fetch_smiles_from_pubchem(cids: list[int]) -> dict[int, str]:
    """Fetch canonical SMILES for a list of PubChem CIDs."""
    cid_str = ",".join(str(c) for c in cids)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/"
        f"{cid_str}/property/CanonicalSMILES/JSON"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    props = resp.json()["PropertyTable"]["Properties"]
    smiles_key = "CanonicalSMILES" if "CanonicalSMILES" in props[0] else "ConnectivitySMILES"
    return {p["CID"]: p[smiles_key] for p in props}


def main():
    df = pd.read_excel(XLSX_PATH)
    names = df["Substrate name"].tolist()
    cids = df["CID"].astype(int).tolist()

    print(f"Fetching SMILES for {len(cids)} substrates from PubChem...")
    cid_to_smiles = fetch_smiles_from_pubchem(cids)

    result = {}
    for name, cid in zip(names, cids):
        smiles = cid_to_smiles[cid]
        result[name] = smiles
        print(f"  {name:45s} CID={cid:>6d}  SMILES={smiles}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
