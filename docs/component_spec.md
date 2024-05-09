<!-- docs/component_spec.md
<!--
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
-->

# Abstract base classes

## AquisitionFunction(Transformer)
Inherits from sklearn transformer. Takes in scores and optionally uncertainy and returns an aquisition score

## Selector
Meant to be wrapped by sklearn FunctionTransformer. This takes in aquisition scores and returns the next variant/s to test.
Options will for example TopN, or a Threshold.

## ZeroShotPredictor(Transformer)
Inherits from sklearn transformer. Takes in a sequence and returns a score.

## SequenceChecker(Transformer)
Inherits from sklearn transformer. Simply checks that all sequences are the same length, and also records the variable amino acid residues.

## Embedder(Transformer)
Inherits from sklearn transformer. Takes in a sequence and returns a feature vector. Options are Eg. one hot or esm1v.
Note that this may require information from the SequenceChecker. Expensive ones of these should write to disk so that embeddings can be reused.

## 

