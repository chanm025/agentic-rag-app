import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf  # only needed for the LSTM models

st.set_page_config(page_title="Currency Forecast App")

DATA_FILE = "Foreign_Exchange_Rates.xlsx"

# Each currency is routed to the model that is used for it: XGBoost (saved with pickle) or an LSTM
# (saved as .keras, plus the MinMaxScaler it was trained with).
currency_model_files = {
    "AUSTRALIA - AUSTRALIAN DOLLAR/US$": {"type": "XGB", "model": "models/AUS_XGB.pkl"},
    "BRAZIL - REAL/US$": {"type": "LSTM", "model": "models/BRZ_LSTM.keras", "scaler": "models/BRZ_scaler.pkl"},
    "CANADA - CANADIAN DOLLAR/US$": {"type": "XGB", "model": "models/CND_XGB.pkl"},
    "CHINA - YUAN/US$": {"type": "XGB", "model": "models/CNY_XGB.pkl"},
    "DENMARK - DANISH KRONE/US$": {"type": "XGB", "model": "models/DNK_XGB.pkl"},
    "EURO AREA - EURO/US$": {"type": "XGB", "model": "models/EUR_XGB.pkl"},
    "HONG KONG - HONG KONG DOLLAR/US$": {"type": "LSTM", "model": "models/HKD_LSTM.keras", "scaler": "models/HKD_scaler.pkl"},
    "INDIA - INDIAN RUPEE/US$": {"type": "LSTM", "model": "models/IDR_LSTM.keras", "scaler": "models/IDR_scaler.pkl"},
    "JAPAN - YEN/US$": {"type": "XGB", "model": "models/JPY_XGB.pkl"},
    "KOREA - WON/US$": {"type": "XGB", "model": "models/KRW_XGB.pkl"},
    "MALAYSIA - RINGGIT/US$": {"type": "LSTM", "model": "models/MLR_LSTM.keras", "scaler": "models/MLR_scaler.pkl"},
    "MEXICO - MEXICAN PESO/US$": {"type": "LSTM", "model": "models/MXP_LSTM.keras", "scaler": "models/MXP_scaler.pkl"},
    "NORWAY - NORWEGIAN KRONE/US$": {"type": "XGB", "model": "models/NRK_XGB.pkl"},
    "NEW ZEALAND - NEW ZELAND DOLLAR/US$": {"type": "XGB", "model": "models/NZ_XGB.pkl"},
    "SOUTH AFRICA - RAND/US$": {"type": "LSTM", "model": "models/SAR_LSTM.keras", "scaler": "models/SAR_scaler.pkl"},
    "SINGAPORE - SINGAPORE DOLLAR/US$": {"type": "XGB", "model": "models/SGD_XGB.pkl"},
    "SRI LANKA - SRI LANKAN RUPEE/US$": {"type": "LSTM", "model": "models/SLR_LSTM.keras", "scaler": "models/SLR_scaler.pkl"},
    "SWITZERLAND - FRANC/US$": {"type": "XGB", "model": "models/SWF_XGB.pkl"},
    "SWEDEN - KRONA/US$": {"type": "XGB", "model": "models/SWK_XGB.pkl"},
    "THAILAND - BAHT/US$": {"type": "XGB", "model": "models/THB_XGB.pkl"},
    "TAIWAN - NEW TAIWAN DOLLAR/US$": {"type": "XGB", "model": "models/TWD_XGB.pkl"},
    "UNITED KINGDOM - UNITED KINGDOM POUND/US$": {"type": "LSTM", "model": "models/UK_LSTM.keras", "scaler": "models/UK_scaler.pkl"},
}
# The keys above are also the column names in the dataset.


# The notebooks save a few files under slightly different names than the ones listed above
# (e.g. NZD_XGB.pkl vs NZ_XGB.pkl, GBP_LSTM.keras vs UK_LSTM.keras, *_LSTM_scaler.pkl vs *_scaler.pkl),
# so each path is also tried under its alternative name.
ALT_NAMES = {
    "models/NZ_XGB.pkl": ["models/NZD_XGB.pkl"],
    "models/UK_LSTM.keras": ["models/GBP_LSTM.keras"],
}


def resolve(path):
    """Return the first existing file among a path and its alternative names, else None."""
    candidates = [path] + ALT_NAMES.get(path, [])
    if path.endswith("_scaler.pkl"):
        candidates.append(path.replace("_scaler.pkl", "_LSTM_scaler.pkl"))
    return next((p for p in candidates if os.path.exists(p)), None)


@st.cache_resource
def load_currency_model(currency_name):
    """Load the saved model (and scaler for LSTMs). Cached so it is only loaded once per currency."""
    if currency_name not in currency_model_files:
        raise ValueError(f"Currency {currency_name} not found in model files")
    info = currency_model_files[currency_name]

    model_path = resolve(info["model"])
    scaler_path = resolve(info["scaler"]) if "scaler" in info else None
    missing = [p for p, found in ((info["model"], model_path), (info.get("scaler"), scaler_path))
               if p and not found]
    if missing:
        raise FileNotFoundError(
            "Missing saved model file(s): " + ", ".join(missing) +
            ". Run the notebooks to create them (see README) or check the file names."
        )

    if info["type"] == "XGB":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model, "XGB", None

    model = tf.keras.models.load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, "LSTM", scaler


def forecast_xgb(model, df, currency_col, n_lags=30, horizon=30):
    """Recursive forecast: predict one step, append it to the history, repeat."""
    series = (
        pd.to_numeric(
            df[currency_col].astype(str).str.replace(",", "").str.strip(),
            errors="coerce",
        ).dropna().values
    )
    if len(series) < 10:  # defensive check
        raise ValueError(f"Not enough valid data points for {currency_col}")

    history = list(series)
    preds = []
    for _ in range(horizon):
        X_input = np.array(history[-n_lags:]).reshape(1, -1)
        y_pred = model.predict(X_input)[0]
        preds.append(y_pred)
        history.append(y_pred)

    return pd.DataFrame({
        # business days only: the training data has no weekends
        "Date": pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon),
        "Forecast": preds,
    })


def forecast_lstm(model, df, currency_col, scaler, n_input=60, horizon=30):
    """Recursive forecast on the scaled series, then convert back to the original units."""
    series = df[currency_col].values
    last_scaled = scaler.transform(series[-n_input:].reshape(-1, 1))

    preds_scaled = []
    for _ in range(horizon):
        X_input = last_scaled[-n_input:].reshape(1, n_input, 1)
        y_pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        preds_scaled.append(y_pred_scaled)
        last_scaled = np.append(last_scaled, y_pred_scaled).reshape(-1, 1)

    # inverse-transform once, after the loop
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return pd.DataFrame({
        "Date": pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon),
        "Forecast": preds,
    })


@st.cache_data
def load_data():
    """Load the exchange-rate sheet (it is stored as one comma-separated column)."""
    df = pd.read_excel(DATA_FILE, sheet_name=0, header=None, engine="openpyxl")
    df = df[0].str.split(",", expand=True)              # split into real columns
    df.columns = df.iloc[0]                             # first row becomes the header
    df = df.drop(0).reset_index(drop=True)
    df = df.rename(columns={"Time Serie": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.set_index("Date")
    return df.apply(pd.to_numeric, errors="coerce")     # all currency columns -> numbers


# ---------------- Streamlit UI ----------------
st.title("Currency Forecast App")
st.caption(
    "Recursive multi-day forecasts from XGBoost or LSTM models trained on 2000-2019 daily rates. "
    "Each step is predicted from earlier predictions, so errors compound: treat long horizons as a demo, not advice."
)

if not os.path.exists(DATA_FILE):
    st.error(f"Data file '{DATA_FILE}' not found. Place it next to app.py (see README).")
    st.stop()

df = load_data()

selected_currency = st.selectbox("Select Currency", options=list(currency_model_files.keys()))
forecast_days = st.number_input("Forecast Horizon (days)", min_value=1, max_value=365, value=30)

if st.button("Generate Forecast"):
    try:
        model, model_type, scaler = load_currency_model(selected_currency)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    st.write(f"Loaded model for {selected_currency}: {model_type}")

    if model_type == "XGB":
        forecast_df = forecast_xgb(model, df, selected_currency, horizon=forecast_days)
    else:
        forecast_df = forecast_lstm(model, df, selected_currency, scaler, horizon=forecast_days)

    st.subheader("Forecast Table")
    st.dataframe(forecast_df)

    st.subheader("Forecast Chart")
    fig, ax = plt.subplots()
    ax.plot(forecast_df["Date"], forecast_df["Forecast"])
    ax.set_title(f"Forecast: {selected_currency}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    plt.xticks(rotation=45)
    st.pyplot(fig)
