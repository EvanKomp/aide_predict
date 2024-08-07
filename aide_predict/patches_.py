import numpy as np
import sys

# Check if np.str is not available
if not hasattr(np, 'str'):
    # Create a custom object that behaves like the old np.str
    class NumpyStrProxy:
        def __new__(cls, *args, **kwargs):
            return str(*args, **kwargs)

    # Add the proxy to the numpy module
    np.str = str

if not hasattr(np, 'int'):
    # Create a custom object that behaves like the old np.int
    # class NumpyIntProxy:
    #     def __new__(cls, *args, **kwargs):
    #         return np.int32(*args, **kwargs)

    # Add the proxy to the numpy module
    np.int = np.int32

# Apply the patch
sys.modules['numpy'] = np

import re
from evcouplings.couplings import tools
import pandas as pd

# Store the original function
original_parse_plmc_log = tools.parse_plmc_log

def patched_parse_plmc_log(log):
    """
    A patched version of parse_plmc_log that handles the new output format.
    """
    # Copy the original regular expressions
    res = {
        "focus": re.compile(r"Found focus (.+) as sequence (\d+)"),
        "seqs": re.compile(r"(\d+) valid sequences out of (\d+)"),
        "sites": re.compile(r"(\d+) sites out of (\d+)"),
        "region": re.compile(r"Region starts at (\d+)"),
        "samples": re.compile(r"Effective number of samples[^:]*: (\d+\.\d+)"),
        "optimization": re.compile(r"Gradient optimization: (.+)")
    }
    
    re_iter = re.compile(r"(\d+){}".format(
        "".join([r"\s+(\d+\.\d+)"] * 6)
    ))
    
    matches = {}
    fields = None
    iters = []
    
    for line in log.split("\n"):
        for (name, pattern) in res.items():
            m = re.search(pattern, line)
            if m:
                matches[name] = m.groups()
        if line.startswith("iter"):
            fields = line.split()
        m_iter = re.search(re_iter, line)
        if m_iter:
            iters.append(m_iter.groups())
    
    if fields is not None and iters:
        iter_df = pd.DataFrame(iters, columns=fields)
    else:
        iter_df = None
    
    # some output only defined in focus mode
    focus_index = None
    valid_sites, total_sites = None, None
    region_start = 1
    try:
        focus_index = int(matches["focus"][1])
        valid_sites, total_sites = map(int, matches["sites"])
        region_start = int(matches["region"][0])
    except KeyError:
        pass
    
    valid_seqs, total_seqs = map(int, matches["seqs"])
    eff_samples = float(matches["samples"][0])
    opt_status = matches["optimization"][0]
    
    return (
        iter_df,
        (
            focus_index, valid_seqs, total_seqs,
            valid_sites, total_sites, region_start,
            eff_samples, opt_status
        )
    )

# Replace the original function with our patched version
tools.parse_plmc_log = patched_parse_plmc_log

import pandas as pd

def patch_pandas_append():

    def patched_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        if isinstance(other, pd.DataFrame):
            start_index = len(self)
            for i, row in other.iterrows():
                self.loc[start_index + i] = row
        elif isinstance(other, pd.Series):
            self.loc[len(self)] = other
        else:
            raise TypeError("Unsupported type for 'other'. Expected DataFrame or Series.")
        
        if ignore_index:
            self.reset_index(drop=True, inplace=True)
        
        if verify_integrity:
            self.drop_duplicates(inplace=True)
        
        if sort:
            self.sort_index(inplace=True)
        
        return self

    pd.DataFrame.append = patched_append

# Apply the patch
patch_pandas_append()