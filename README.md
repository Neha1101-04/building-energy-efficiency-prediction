# Predicting Building Energy Efficiency

Machine learning models that estimate the **heating load** and **cooling load** of residential buildings from eight design characteristics such as compactness, height, surface area and glazing.

## Dataset
UCI Energy Efficiency dataset (`ENB2012_data.csv`): 768 simulated buildings, 8 input features, 2 target variables. The notebook loads the CSV directly from a URL, so no manual download is needed.

## Approach
1. Load the data and rename the columns to descriptive names
2. Check data quality (missing values, duplicates, data types, ranges)
3. Exploratory data analysis (correlations, distributions, feature vs target plots)
4. 80/20 train/test split
5. Baseline multi-output Linear Regression
6. Random Forest Regressor
7. Feature importance and predicted-vs-actual analysis

## Results (test set)

| Target | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Heating load | Linear Regression | 2.182 | 3.025 | 0.912 |
| Heating load | Random Forest | 0.345 | 0.494 | 0.998 |
| Cooling load | Linear Regression | 2.195 | 3.145 | 0.893 |
| Cooling load | Random Forest | 1.165 | 1.905 | 0.961 |

## Key findings
- Building shape drives energy demand: relative compactness, overall height and surface area make up about 80% of the random forest's feature importance.
- Orientation has almost no effect on either load.
- The relationship is partly non-linear, so the random forest clearly outperforms the linear baseline.
- Heating load is easier to predict than cooling load.

## Limitations
The data is simulated from a fixed grid of building designs, so a random split likely gives optimistic scores for entirely new designs. See the notebook for next steps (grouped cross-validation, hyperparameter tuning, gradient boosting, SHAP).

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Predicting_Building_Energy_Efficiency.ipynb
```
