# Protein Prediction

This project provides tools for ML aided prediction of protein mutation combinatorial libraries.

## Notes

This repo is build with DVC such that it can be blanket applied to any combo library by changing the input data.

## TODO:
- Write embeddings classes
- Write EVcouplings wrapper
- Write Tranception wrapper * (low priority, PN did not provide a clear entry point so will require some finagling)
- Write MSATransformer wrapper. should be esasy if we enforce WT and fixed length. Maybe in the future extend to no WT compare by ensureing all sequences are in the passed are in the MSA
- Write "training" pipeline to init, potentially fit, and save all sklearn estimators and pipelines
- Write "predict" pipeline to load all sklearn estimators and pipelines, and predict on the passed data

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


## Citations
No software or code with viral licenses was used in the creation of this project.

Some coda from Tranception was adapted for MSA processing in this repo, and the model itself if supported as a wrapper:
> Notin, P., Dias, M., Frazer, J., Marchena-Hurtado, J., Gomez, A., Marks, D.S., Gal, Y. (2022). Tranception: Protein Fitness Prediction with Autoregressive Transformers and Inference-time Retrieval. ICML.

## License

This project is licensed under the [MIT License](LICENSE).