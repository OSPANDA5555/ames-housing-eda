# Ames Housing Dataset — Exploratory Data Analysis (EDA)

A complete, reproducible Exploratory Data Analysis (no machine learning) of the **Ames Housing Dataset** (`train.csv`, 1,460 rows × 81 columns), structured step-by-step according to the project specification.

## Project Files

- `ames_housing_eda.ipynb` — Executed Jupyter notebook covering Sections 1–9 (Introduction, Data Loading, Variable Classification & Descriptive Statistics, Data Quality & Cleaning, Univariate Distributions, Correlation/Covariance & Scatter/Pair Plots, Correlation Heatmap, Outlier Analysis, 2×2 Summary Dashboard, and Conclusions).
- `ames_housing_eda.html` — Self-contained HTML report exported from the executed notebook (includes all tables, outputs, and embedded visualizations).
- `ames_housing_eda.py` — Standalone Python script version exported from the notebook.
- `eda_dashboard.png` — 2×2 executive summary dashboard saved in Step 10 (`House Price EDA Summary`).
- `train.csv` / `test.csv` / `sample_submission.csv` — Kaggle House Prices dataset files.
- `data_description.txt` — Official feature dictionary explaining `"NA"` (*feature absent*) encodings and ordinal scales.
- `requirements.txt` — Pinned Python dependencies for `.venv`.

## How to Run

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Re-execute the notebook end-to-end
jupyter nbconvert --to notebook --execute --inplace ames_housing_eda.ipynb

# Or run the standalone Python script
python ames_housing_eda.py
```
