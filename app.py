# app.py
# -----------------------------------------
# FreshRetailNet-50K Sales Forecasting Dashboard
# -----------------------------------------

import os
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_data
from utils.visualization import plot_forecast


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="FreshRetail Sales Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 FreshRetailNet-50K Sales Forecasting Dashboard")
st.markdown(
    "Forecast product-level sales with **FreshRetailNet-50K** dataset. "
    "Use filters to explore store and product trends."
)


# -------------------------------
# Load Dataset (with local caching)
# -------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    if os.path.exists("data/train_cache.csv") and os.path.exists("data/eval_cache.csv"):
        train_df = pd.read_csv("data/train_cache.csv")
        eval_df = pd.read_csv("data/eval_cache.csv")
    else:
        from datasets import load_dataset
        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        train_df = pd.DataFrame(dataset["train"])
        eval_df = pd.DataFrame(dataset["eval"])
        os.makedirs("data", exist_ok=True)
        train_df.to_csv("data/train_cache.csv", index=False)
        eval_df.to_csv("data/eval_cache.csv", index=False)
    return train_df, eval_df


train_df, eval_df = load_data()


# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔍 Filters")

store_id = st.sidebar.selectbox("Select Store", sorted(train_df['store_id'].unique()))
product_id = st.sidebar.selectbox("Select Product", sorted(train_df['product_id'].unique()))
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 60, 30)


# -------------------------------
# Data Preprocessing
# -------------------------------
df_filtered = preprocess_data(train_df, store_id, product_id)

st.subheader(f"📈 Historical Sales Data (Store {store_id}, Product {product_id})")
st.dataframe(df_filtered.head(10))


# -------------------------------
# Train Prophet Model
# -------------------------------
if len(df_filtered) > 10:
    # Prepare data for Prophet
    df_prophet = df_filtered.rename(columns={'dt': 'ds', 'sale_amount': 'y'})

    model = Prophet()
    # Add external regressors if available
    for reg in ['discount', 'precpt', 'avg_temperature', 'avg_humidity']:
        if reg in df_prophet.columns:
            model.add_regressor(reg)

    model.fit(df_prophet)

    # Future DataFrame
    future = model.make_future_dataframe(periods=forecast_horizon, freq='D')

    # Fill future regressors with last known values
    for reg in ['discount', 'precpt', 'avg_temperature', 'avg_humidity']:
        if reg in df_prophet.columns:
            future[reg] = df_prophet[reg].iloc[-1]

    # Forecast
    forecast = model.predict(future)

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📊 Forecast Results")
    fig = plot_forecast(model, forecast, store_id, product_id)
    st.pyplot(fig)

    # Show forecast table
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

    # -------------------------------
    # Download Forecast
    # -------------------------------
    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"forecast_store{store_id}_product{product_id}.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ Not enough data for this store/product. Try another selection.")
