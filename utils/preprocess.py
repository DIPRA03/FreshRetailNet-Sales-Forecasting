# utils/preprocess.py
import pandas as pd

def preprocess_data(train_df: pd.DataFrame, store_id: int, product_id: int) -> pd.DataFrame:
    """
    Preprocess sales data for a given store & product.
    - Converts dates
    - Filters by store/product
    - Handles missing values
    """

    df_filtered = train_df[
        (train_df['store_id'] == store_id) & (train_df['product_id'] == product_id)
    ].copy()

    # Convert datetime
    df_filtered['dt'] = pd.to_datetime(df_filtered['dt'])

    # Handle missing sales
    if 'sale_amount' in df_filtered.columns:
        df_filtered['sale_amount'] = df_filtered['sale_amount'].fillna(method='ffill').fillna(0)

    # Handle missing external features if they exist
    for col in ['discount', 'precpt', 'avg_temperature', 'avg_humidity']:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].fillna(method='ffill').fillna(0)

    # Sort by date
    df_filtered = df_filtered.sort_values('dt').reset_index(drop=True)

    return df_filtered
