# Protein Prediction

This project provides tools for ML aided prediction of protein mutation combinatorial libraries.

## Notes

This repo is build with DVC such that it can be blanket applied to any combo library by changing the input data.

## Project Structure

The project has the following directory structure:

```
protein-prediction
├── aide_predict
│   └── !TODO!
├── data
├── tests
│   └── __init__.py
├── docs
│   ├── component_spec.md # importable components for selecting next variants
├── dvc.yaml
├── params.yaml
├── setup.py
├── environment.yaml
└── README.md
```

## Methodology
Here, we scope only to the application of ML and covariation strategies to filter combinatorial library.
A la Hsu et al, predictor heads are trained on top of embeddings and/or zero shot predictive scores.
Indels ARE NOT supported, but are intended to be in future work as the aide ecosystem grows.

Current embeddings supported:
- EMS1v, add ./additional_dependancies/dependancies_esm1v.yaml to conda env
- One Hot Encoding

Current Zero Shot/Weak Supervised predictors supported:
- Tranception, add ./additional_dependancies/dependancies_tranception.yaml to conda env
- ESM1v, add ./additional_dependancies/dependancies_esm1v.yaml to conda env

Current supervised predictors supported:
- sklearn suite

## Usage

To use the tools for ML aided prediction of protein mutation combinatorial libraries, follow these steps:

1. Install the required dependencies by running the following command:

   ```
   conda env create -f environment.yaml
   ```

2. Prepare the data by placing it in the `data/` directory.

3. Configure the parameters for the ML models and prediction tools in the `params.yaml` file.

4. Run `dvc repro` to execute the ML models and prediction tools, spitting out what to select next


## License

This project is licensed under the [MIT License](LICENSE).