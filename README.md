# Buenos Aires Housing Price Prediction

Predicting apartment listing prices in Capital Federal from property size and location, and comparing linear regression with Ridge and Lasso regularization. Four notebooks cover data preparation, a first predictive model, regularization, and a final model comparison with explicit numerical and evaluation limitations.

## Objective

Estimate `price_aprox_usd` using covered area, latitude, longitude, and neighborhood. Compare prediction error against a training-mean baseline, inspect residuals, and assess what regularization changes in performance and coefficient sparsity. The target represents listing prices, not completed sale prices.

## Dataset and Preparation

The five CSV files in [data/](data/) contain 43,029 property listings before cleaning. Preparation retains Capital Federal apartments priced below USD 400,000, filters covered area between the 10th and 90th percentiles, extracts coordinates and neighborhood from string columns, removes unused and price-derived columns, and drops rows with missing retained values.

The result is **6,247 rows**, with target `price_aprox_usd` and four predictors: `surface_covered_in_m2`, `lat`, `lon`, and `neighborhood`. The area interval nominally covers the middle 80%, despite the cleaning function's “middle 90%” docstring. Its thresholds are computed before splitting.

The notebooks use an 80/20 split with `random_state=42`: 4,997 training rows and 1,250 test rows. [src/wrangle.py](src/wrangle.py) centralizes cleaning for notebooks 2–4; notebook 1 develops the cleaning steps directly.

## Project Walkthrough

| Notebook | Contribution |
|---|---|
| [01 — Data Cleaning and Feature Engineering](notebooks/01_Data_Cleaning_and_Feature_Engineering.ipynb) | Merge listings, examine missingness and correlations, clean data, extract features, and define the split. |
| [02 — First Predictive Model](notebooks/02_First_Predictive_Model.ipynb) | Fit covered-area-only linear regression; evaluate MAE, RMSE, and R², with prediction and residual plots. |
| [03 — Preventing Overfitting](notebooks/03_Preventing_Overfitting.ipynb) | Add location predictors, examine fixed penalties and alpha sweeps, perform fold-safe searches, and interpret coefficients. |
| [04 — Model Selection](notebooks/04_Model_Selection.ipynb) | Compare a mean baseline, multivariate OLS, Ridge, and Lasso on identical observations; report convergence, residuals, and sparsity. |

## Methodology

- **Baseline:** predict the training-target mean on both training and test observations.
- **Linear regression:** notebook 2 uses covered area alone; notebook 4 uses all four raw predictors. These are distinct fitted models.
- **Preprocessing:** multivariate pipelines one-hot encode neighborhood with `category_encoders` and standardize the encoded predictors.
- **Regularization:** Ridge applies an L2 penalty; Lasso applies an L1 penalty that can set coefficients exactly to zero.
- **Cross-validation:** corrected Ridge/Lasso searches fit the entire preprocessing-and-model pipeline inside five unshuffled folds. Ridge maximizes validation R²; Lasso minimizes validation MSE. Test data do not select alpha inside these searches.
- **Final comparison:** notebook 4 searches 100 logarithmically spaced Ridge alphas from 0.01 to 10,000 and Lasso alphas from 0.0001 to 100. These differ from notebook 3's grids. The selected pipelines are refitted on all training rows and evaluated using RMSE in USD.

## Results — Notebook 4

The following are saved results from the final multivariate comparison, not notebook 2's single-feature experiment.

| Model | Training RMSE | Test RMSE | Selected alpha |
|---|---:|---:|---:|
| Mean baseline | $59,700.25 | $57,576.13 | — |
| Multivariate OLS | $34,200.43 | $33,088.37 | — |
| Ridge | $34,205.01 | $33,069.57 | 49.770236 |
| Lasso | $34,202.52 | $33,071.30 | 28.480359 |

### Key Findings

All three multivariate linear models improve substantially on the baseline on this split. Ridge has the lowest observed test RMSE, but its **$1.73 advantage over Lasso** does not establish statistically reliable superiority. Regularization changes predictive error only slightly in this comparison.

The selected Lasso sets **1 of 59 encoded coefficients to zero (approximately 1.7%)**, indicating modest sparsity. Covered area has the largest absolute standardized coefficient, approximately USD 40,349.83 per training standard deviation, conditional on the other inputs. This is a fitted association, not causal importance. Neighborhood indicators have no explicitly omitted reference category, so standardized coefficients are not direct premiums over a reference neighborhood.

The prediction and residual plots show substantial errors and changing residual spread. They expose limitations of the linear predictions without establishing prediction intervals or deployment readiness.

## How to Run

The analysis dependencies are declared in [requirements.txt](requirements.txt): pandas, NumPy, Matplotlib, seaborn, scikit-learn, category-encoders, and missingno. Notebook tooling is separate from these project dependencies.

From the repository root, create and activate a Python environment, then install the dependencies and optional Jupyter tooling:

```sh
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```sh
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m ipykernel install --user --name buenos-aires-housing --display-name "Buenos Aires Housing"
```

Select the **Buenos Aires Housing** kernel in your notebook interface. Read the notebooks in order, 01 → 04; each reconstructs its data and models rather than loading another notebook's kernel state.

### Kernel Working Directories

| Notebooks | Required kernel working directory | Data path |
|---|---|---|
| 1–2 | Repository's `notebooks/` directory | `../data/` |
| 3–4 | Repository root | `data/` |

Starting Jupyter from the repository root does **not** by itself guarantee the kernel directory. Verify it with `Path.cwd()` from `pathlib` in your interface's console, and configure the directory before running cells. Notebooks 3–4 also import `src.wrangle` from the repository root.

For execution with explicitly controlled directories, start `python` in the repository root after the setup above and paste the following. It runs each notebook in a fresh kernel and refreshes its saved outputs; the Lasso search can take several minutes.

```python
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path.cwd()
for path in sorted((root / "notebooks").glob("*.ipynb")):
    cwd = path.parent if path.name.startswith(("01_", "02_")) else root
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        kernel_name="buenos-aires-housing",
        timeout=1800,
        resources={"metadata": {"path": str(cwd)}},
    ).execute()
    nbformat.write(notebook, path)
    print(f"Executed {path.name}")
```

`nbformat`, `nbclient`, and the kernel tooling used here are supplied by the Jupyter environment; they are not added to the analysis requirements. Version lower bounds do not pin an exact environment, so dependency changes may affect numerical results.

### Public Project Structure

```text
buenos-aires-housing-prediction/
├── data/                 # Five buenos-aires-real-estate-{1..5}.csv files
├── notebooks/            # Four portfolio notebooks linked above
├── src/
│   └── wrangle.py         # Shared cleaning functions
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Limitations

- **Test-set reuse:** the holdout is inspected across modeling stages and used to choose the leading model. It is not an independent final evaluation.
- **Lasso convergence:** notebook 4 records 293 convergence warnings at `max_iter=10000` and default tolerance `0.0001`. Some low-alpha candidates did not converge, making their CV scores uncertain. The selected alpha, both immediate neighbors, and the selected full-training refit did converge in focused diagnostics; the optimum over the complete grid remains unconfirmed.
- **Data-wide cleaning:** covered-area thresholds are learned before splitting. Fold-safe model preprocessing does not eliminate that upstream selection limitation.
- **Scope and stability:** results describe one filtered dataset and one train/test split. Price caps, area filtering, and complete-case selection limit generalization to other listings or market conditions.
- **Interpretation:** coefficient magnitudes depend on scaling, correlated predictors, category prevalence, and penalization. Zero coefficients do not prove that excluded predictors lack information. No statistical significance, coefficient confidence intervals, or prediction intervals are established.

## ⚠️ Academic Integrity & Disclaimer
This project is completed as part of the **Applied Data Science program at WorldQuant University (WQU)**.

**For Fellow Students:**
Please do not copy, fork, or submit any code from this repository as your own coursework. The purpose of this repo is to showcase my personal learning journey, critical thinking, and portfolio to potential employers.

## License and Data Provenance

The repository's code is offered under the [MIT License](LICENSE). This does not establish a license for third-party datasets or instructional materials.

The five CSVs were supplied for the learning project and contain `properati_url` listing links. The repository owner has confirmed that publication of the WQU portfolio work and the included datasets is permitted. This confirmation does not specify a dataset license or broader reuse terms; the MIT code license should not be treated as a license for the CSVs.
