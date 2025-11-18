
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import matplotlib.pyplot as plt
import os
import itertools

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_FILE = 'output/excel/filtered_data.xlsx'
OUTPUT_DIR = 'output/excel/'
PLOTS_DIR = 'output/plots/'
START_DATE = '2020-01-01'
END_DATE = '2025-09-30'
FORECAST_END_DATE = '2026-12-31'

def load_and_complete_data(file_path):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return None
    df['ds'] = pd.to_datetime(df['ds'])
    completed_dfs = []
    full_date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    for sap_code in df['SAP'].unique():
        sap_df = df[df['SAP'] == sap_code]
        sap_df = sap_df.set_index('ds').reindex(full_date_range).reset_index().rename(columns={'index': 'ds'})
        sap_df['SAP'] = sap_code
        sap_df['y'] = sap_df['y'].fillna(0)
        sap_df['mes_picos'] = sap_df['ds'].dt.month.isin([2, 3, 9, 10]).astype(int)
        completed_dfs.append(sap_df[['ds', 'SAP', 'y', 'mes_picos']])
    completed_data = pd.concat(completed_dfs)
    print("Data loading and completion successful.")
    return completed_data

def train_arima(train_data):
    try:
        model = ARIMA(train_data['y'].values, order=(5,1,0))
        return model.fit()
    except Exception: return None

def train_sarimax(train_data, exog_train):
    try:
        p, d, q = range(0, 2), range(0, 2), range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
        best_aic = np.inf
        best_params = None
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = SARIMAX(train_data['y'], exog=exog_train, order=param, seasonal_order=param_seasonal)
                    results = mod.fit(disp=False)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = (param, param_seasonal)
                except Exception: continue
        if best_params:
            mod = SARIMAX(train_data['y'], exog=exog_train, order=best_params[0], seasonal_order=best_params[1])
            return mod.fit(disp=False)
    except Exception: return None
    return None

def create_features_for_rf(df):
    df['month'] = df['ds'].dt.month
    for i in range(1, 13):
        df[f'lag_{i}'] = df['y'].shift(i)
    return df

def train_random_forest(train_data):
    df = create_features_for_rf(train_data.copy()).dropna()
    if df.empty: return None
    features = ['mes_picos', 'month'] + [f'lag_{i}' for i in range(1, 13)]
    X_train, y_train = df[features], df['y']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_random_forest_recursive(model, train_data, steps, future_dates_df=None):
    full_data = train_data.copy()

    if future_dates_df is None:
        # This branch is for test set prediction
        dates = pd.date_range(start=full_data['ds'].max() + pd.DateOffset(months=1), periods=steps, freq='MS')
    else:
        # This branch is for future forecasting
        dates = future_dates_df['ds']

    for next_date in dates:
        next_pico = 1 if next_date.month in [2, 3, 9, 10] else 0

        temp_df = pd.concat([full_data, pd.DataFrame([{'ds': next_date, 'y': np.nan, 'mes_picos': next_pico}])], ignore_index=True)
        temp_df_featured = create_features_for_rf(temp_df)

        last_row = temp_df_featured.iloc[-1]
        features = ['mes_picos', 'month'] + [f'lag_{i}' for i in range(1, 13)]
        pred_input = last_row[features].values.reshape(1, -1)

        yhat = model.predict(pred_input)[0]

        full_data = pd.concat([full_data, pd.DataFrame([{'ds': next_date, 'y': yhat, 'mes_picos': next_pico}])], ignore_index=True)

    return full_data.iloc[-steps:]['y'].values


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return rmse, mae, mape

def select_best_model(metrics):
    return min(metrics, key=lambda k: metrics[k][0])

def plot_full_forecast(sap_code, historical, test_actual, test_pred, future_forecast):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(historical['ds'], historical['y'], label='Historical Consumption', color='black', linewidth=2)
    ax.plot(test_actual['ds'], test_actual['y'], label='Actual Test Data', color='blue', marker='o', linestyle='None')
    ax.plot(test_actual['ds'], test_pred, label='Test Set Forecast', color='red', linestyle='--', linewidth=2)
    ax.plot(future_forecast['Periodo'], future_forecast['Forecast'], label='Future Forecast', color='green', linestyle='--', linewidth=2)
    ax.set_title(f'Consumption Forecast for SAP: {sap_code}', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Consumption', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f'{PLOTS_DIR}forecast_{sap_code}.png')
    plt.close()

if __name__ == '__main__':
    completed_data = load_and_complete_data(DATA_FILE)
    all_metrics, all_future_forecasts = [], []

    if completed_data is not None:
        for sap_code in completed_data['SAP'].unique():
            print(f"\n--- Processing SAP: {sap_code} ---")
            sap_data = completed_data[completed_data['SAP'] == sap_code].copy().reset_index(drop=True)

            if len(sap_data) < 20: continue

            train_size = int(len(sap_data) * 0.8)
            train, test = sap_data.iloc[:train_size], sap_data.iloc[train_size:]

            models = {
                'ARIMA': train_arima(train),
                'SARIMAX': train_sarimax(train, train[['mes_picos']]),
                'RandomForest': train_random_forest(train)
            }

            metrics, test_preds = {}, {}
            if models['ARIMA']:
                preds = models['ARIMA'].forecast(steps=len(test))
                metrics['ARIMA'] = calculate_metrics(test['y'], preds)
                test_preds['ARIMA'] = preds
            if models['SARIMAX']:
                forecast = models['SARIMAX'].get_forecast(steps=len(test), exog=test[['mes_picos']])
                metrics['SARIMAX'] = calculate_metrics(test['y'], forecast.predicted_mean)
                test_preds['SARIMAX'] = forecast.predicted_mean.values
            if models['RandomForest']:
                preds = predict_random_forest_recursive(models['RandomForest'], train, len(test))
                metrics['RandomForest'] = calculate_metrics(test['y'], preds)
                test_preds['RandomForest'] = preds

            if not metrics: continue

            best_model_name = select_best_model(metrics)
            print(f"Best model for {sap_code}: {best_model_name}")

            summary = {'SAP': sap_code, 'Best_Model': best_model_name}
            for name, (rmse, mae, mape) in metrics.items():
                summary[f'RMSE_{name}'], summary[f'MAE_{name}'], summary[f'MAPE_{name}'] = rmse, mae, mape
            all_metrics.append(summary)

            future_dates = pd.date_range(start=END_DATE, end=FORECAST_END_DATE, freq='MS')
            forecast_len = len(future_dates)
            future_forecast_vals = None

            if best_model_name == 'ARIMA':
                full_model = train_arima(sap_data)
                if full_model: future_forecast_vals = full_model.forecast(steps=forecast_len)
            elif best_model_name == 'SARIMAX':
                full_model = train_sarimax(sap_data, sap_data[['mes_picos']])
                if full_model:
                    future_exog = pd.DataFrame({'ds': future_dates, 'mes_picos': [1 if d.month in [2,3,9,10] else 0 for d in future_dates]})
                    future_forecast_vals = full_model.get_forecast(steps=forecast_len, exog=future_exog[['mes_picos']]).predicted_mean
            elif best_model_name == 'RandomForest':
                full_model = train_random_forest(sap_data)
                if full_model: future_forecast_vals = predict_random_forest_recursive(full_model, sap_data, forecast_len)

            if future_forecast_vals is not None:
                future_df = pd.DataFrame({'SAP': sap_code, 'Periodo': future_dates, 'Forecast': future_forecast_vals})
                all_future_forecasts.append(future_df)
                plot_full_forecast(sap_code, sap_data, test, test_preds[best_model_name], future_df)
                print(f"Plot saved for SAP: {sap_code}")

    if all_metrics:
        pd.DataFrame(all_metrics).to_excel(f'{OUTPUT_DIR}evaluation_metrics.xlsx', index=False)
        print(f"\nEvaluation metrics saved to {OUTPUT_DIR}evaluation_metrics.xlsx")
    if all_future_forecasts:
        pd.concat(all_future_forecasts).to_excel(f'{OUTPUT_DIR}future_predictions.xlsx', index=False)
        print(f"Future predictions saved to {OUTPUT_DIR}future_predictions.xlsx")
