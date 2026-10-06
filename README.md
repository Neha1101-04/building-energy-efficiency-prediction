# Predicting Building Energy Efficiency

Machine learning project that predicts a building's **heating load** and **cooling load** from its shape and glazing (window) features, using the Energy Efficiency (ENB2012) dataset.

## Dataset

768 simulated buildings, 8 input features and 2 targets (no missing values, no duplicates).
The notebook loads it directly from a public CSV URL, so no manual download is needed.

| Original | Column name | Meaning |
|---|---|---|
| X1 | `relative_compactness` | Relative compactness |
| X2 | `surface_area` | Surface area |
| X3 | `wall_area` | Wall area |
| X4 | `roof_area` | Roof area |
| X5 | `overall_height` | Overall height |
| X6 | `orientation` | Orientation |
| X7 | `glazing_area` | Glazing area |
| X8 | `glazing_area_distribution` | Glazing area distribution |
| Y1 | `heating_load` | Target: heating load |
| Y2 | `cooling_load` | Target: cooling load |

## Workflow

1. Load and clean the data
2. Exploratory data analysis (correlations, distributions, outliers)
3. Train/test split (80/20, `random_state=42`)
4. Linear Regression and Random Forest, with actual-vs-predicted and residual plots
5. 5-fold cross-validation
6. Feature selection experiment (8 vs 6 features)
7. Ridge and Lasso with scaling in a `Pipeline` (no data leakage)
8. Gradient Boosting and `GridSearchCV` tuning of Random Forest
9. Final comparison, saving the model, and conclusions

## Results (test set)

| Model | Heating R² | Heating RMSE | Cooling R² | Cooling RMSE |
|---|---|---|---|---|
| Linear Regression | 0.912 | 3.03 | 0.893 | 3.15 |
| Random Forest | 0.998 | 0.50 | 0.961 | 1.91 |
| **Gradient Boosting (best)** | **0.998** | **0.45** | **0.982** | **1.28** |

The tuned Random Forest improved its cross-validation score but not its test score. Run the notebook for the full table.

## Key findings

- `relative_compactness` and `overall_height` are the most important features (about 65% of Random Forest importance combined).
- `orientation` and `glazing_area_distribution` add almost nothing and can be dropped without losing accuracy.
- Linear models plateau around R² ≈ 0.90 because the relationships are non-linear; regularization (Ridge/Lasso) does not help.
- Cooling load is harder to predict than heating load for every model.

## Limitations

- The data is simulated and small, with a limited set of base shapes repeated across orientation and glazing. The same shape can appear in both train and test, so scores are probably optimistic for entirely new shapes.
- Only one climate and one building type are covered.
- Next step: `GroupKFold` so all rows of a building shape stay in the same fold.

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Predicting_Building_Energy_Efficiency_Completed.ipynb
```

An internet connection is needed on first run to fetch the dataset.

## Using the saved model

The notebook saves the final Gradient Boosting model as `energy_model.pkl`:

```python
import joblib, pandas as pd

model = joblib.load('energy_model.pkl')
building = pd.DataFrame([{
    'relative_compactness': 0.90, 'surface_area': 563.5, 'wall_area': 318.5,
    'roof_area': 122.5, 'overall_height': 7.0, 'orientation': 3,
    'glazing_area': 0.25, 'glazing_area_distribution': 2
}])
heating, cooling = model.predict(building)[0]
```

Note: pickle files should be loaded with the same scikit-learn version used to create them.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib

## Project structure

```
├── Predicting_Building_Energy_Efficiency_Completed.ipynb
├── README.md
└── requirements.txt
```
