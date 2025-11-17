# Time Series Forecasting Pipeline

This project provides a reproducible pipeline for forecasting monthly consumption of various products using four different time series models: ARIMA, SARIMAX, Prophet, and XGBoost.

## How to Run

1.  **Place the data file:** Ensure that the `HCkardexPesca.xlsx` file is in the root directory of the project.
2.  **Install dependencies:** Run the following command to install the necessary Python libraries:
    ```bash
    pip install pandas openpyxl statsmodels prophet xgboost scikit-learn matplotlib
    ```
3.  **Run the pipeline:** Execute the main script from the root directory:
    ```bash
    python3 src/forecasting_pipeline.py
    ```

## Outputs

*   **Filtered Data:** A filtered version of the input data is saved to `output/excel/filtered_data.xlsx`.
*   **Predictions and Metrics:** Detailed predictions and a summary of model performance metrics are saved in `output/excel/metrics_by_product.xlsx`.
*   **Plots:** Visualizations of the predictions for each product are saved as PNG images in the `output/plots/` directory.
