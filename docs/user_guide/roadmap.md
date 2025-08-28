---
title: Roadmap
---

# Roadmaps

## Additional predictors from maintainers
__Quarter 2, 2026__

We have the following predictors planned to be wrapped:

1. ProteinMPNN (as a zero shot predictor)
  - RequiresStructureMixin
  - Does not require wildtype, eg. can do mutant marginal on variable length structures.
  - Type 3 dependencies (sub environment)
2. NOMELT (as a zero shot predictor)
  - Temperatute specific scores
3. ESM3 (Zero shot prediction and embedder)
  - Structure aware if available
  - For now, the annotation data mode will not be considered. This will require an additional attribute of the ProteinSequence class (in addition to sequence, structure, msa data types already supported).

## Contributions from the community
While we will maintain the codebase and address bugs identified by the community, the usefulness of the tool will ultimately require contributions from the community _a la_ higgingface scikit-learn, etc. When / If we add (2) community contributed models we will start developping the next major version (v2) 

## Major update v2
__Undetermined__

The component specification and software engineering exercise conducted in AIDE for different types of predictors would also be helpful for the variable and dispirate __generator__ methods available, eg. methods for producing producing new sequences. These broadly categorize into:
- Unconditional generators
- Conditional generators (eg. infilling, homolog aware, structure or other property conditioning)
    - ProteinMPNN
    - NOMELT
    - Tranception
    - 
- Score optimizers, eg. BADASS (already included in the package) which use a scoring function (AIDE predictor) to bias generation.
    - This type may even be able to be wrapped with Conditional Generators for tandem generation and filtering.
