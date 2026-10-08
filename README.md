# Forex Currency Forecasting

Forecasts daily exchange rates against the US dollar for **22 currencies**, comparing three model
families (XGBoost, LSTM, Prophet) and serving the chosen models in a Streamlit app.

## Data
Daily rates, 3 Jan 2000 to 31 Dec 2019 (5,015 rows after cleaning; 202 rows with missing/non-numeric values dropped).
The data file (`Foreign_Exchange_Rates.xlsx`) is not included in this repo.

## Notebooks
| Notebook | What it does |
|---|---|
| `analysis.ipynb` | Cleaning, EDA (trends, correlation between currencies), and the LSTM models |
| `XGBoost.ipynb` | One XGBoost model per currency, plus a naive baseline check |
| `prophet.ipynb` | One Prophet model per currency |

## Method
- **Split:** chronological 70% train / 10% validation / 20% test (Prophet: 70/30, so not directly comparable).
- **XGBoost:** previous 30 daily rates as features, early stopping on the validation set.
- **LSTM:** 60-day window, MinMax scaling, three stacked LSTM layers (100, 100, 50 units) + dense head, early stopping.
- **Prophet:** trend + weekly/yearly seasonality.
- **Metrics:** MAE, RMSE, MAPE on the test set.

## Results (test set)
- **XGBoost:** MAPE of **0.18-0.68% on 15 currencies**, but **3-24% on seven** (Brazil 3.4%, India 6.4%, UK 7.4%, Malaysia 8.4%, South Africa 15.9%, Sri Lanka 16.9%, Mexico 24.0%).
  These are the currencies whose rates trended outside the range seen in training; tree-based models cannot predict values beyond what they saw during training.
- **Prophet:** generally weaker, MAPE from 1.0% (Hong Kong) to 21.8% (India).
- **LSTM:** an example run had 0.65% MAPE on training data but 4.78% on the test set, showing the same difficulty with non-stationary series.
- **App routing:** XGBoost for 14 currencies; LSTM for 8 (Brazil, Hong Kong, India, Malaysia, Mexico, South Africa, Sri Lanka, UK), including all seven where XGBoost struggled.

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app needs `Foreign_Exchange_Rates.xlsx` next to `app.py` and a `models/` folder created by running the notebooks
(XGBoost: `models/<CODE>_XGB.pkl`; LSTM: `models/<CODE>_LSTM.keras` and a scaler `.pkl`). The app also accepts the
file names the notebooks write (e.g. `NZD_XGB.pkl`, `GBP_LSTM.keras`, `*_LSTM_scaler.pkl`).

## Limitations / next steps
- **Baseline:** exchange rates are close to a random walk, and the XGBoost test metrics are one-step-ahead (true previous rates as input),
  so a "tomorrow = today" forecast may score similarly. The last cell of `XGBoost.ipynb` computes this baseline;
  models should be compared against it.
- **Forecast horizon:** the app forecasts recursively (each prediction feeds the next), so errors compound over long horizons.
- **Evaluation:** a single train/test split, not rolling-origin cross-validation.
- Data ends in 2019; not financial advice.
