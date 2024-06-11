# aide_predict/io/bio_files.py
'''
* Author: Evan Komp
* Created: 5/22/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os

def read_fasta_like(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fasta file does not exist: {file_path}")
    
    outs = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                key = line[1:].strip()
                outs[key] = ''
            else:
                outs[key] += line.strip()
    return outs

def write_fasta_like(file_path: str, seqs: dict):
    with open(file_path, 'w') as f:
        for key, seq in seqs.items():
            f.write(f'>{key}\n{seq}\n')