# utils/visualization.py
import matplotlib.pyplot as plt

def plot_forecast(model, forecast, store_id, product_id):
    """
    Plot Prophet forecast with labels.
    """
    fig = model.plot(forecast, xlabel="Date", ylabel="Sales Amount")
    plt.title(f"Sales Forecast - Store {store_id}, Product {product_id}")
    return fig
