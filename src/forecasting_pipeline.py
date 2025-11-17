
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
import matplotlib.pyplot as plt
import os
import itertools

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_FILE = 'HCkardexPesca.xlsx'
OUTPUT_DIR = 'output/excel/'
PLOTS_DIR = 'output/plots/'
START_DATE = '2020-01-01'
END_DATE = '2025-09-30'
FORECAST_END_DATE = '2026-12-31'
SAP_CODES = [
    'A18110010517', 'A18130007812', 'A18130007814', 'A18130005688', 'A19000000276',
    'A18110009037', 'A18110002547', 'A18130007396', 'A18110009021', 'A18110010465',
    'A18110008863', 'A18110010460', 'A18130007393', 'A18130007399', 'A18130007380',
    'A18130006711', 'A18110001067'
]

def load_and_preprocess_data(file_path):
    # (Implementation is the same as before)
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return None
    df = df.rename(columns={'month_year': 'ds', 'Consumo Total': 'y'})
    df = df[df['SAP'].isin(SAP_CODES)]
    df['ds'] = pd.to_datetime(df['ds'])
    df = df[(df['ds'] >= START_DATE) & (df['ds'] <= END_DATE)]
    df['pico'] = df['ds'].dt.month.isin([2, 3, 9, 10]).astype(int)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_excel(f"{OUTPUT_DIR}filtered_data.xlsx", index=False)
    print(f"Filtered data saved to {OUTPUT_DIR}filtered_data.xlsx")
    return df

# --- Model Training ---
def train_arima(train_data):
    # (Implementation is the same as before)
    try:
        model = ARIMA(train_data['y'].values, order=(5,1,0))
        return model.fit()
    except Exception: return None

def train_sarimax(train_data, exog_train):
    # (Implementation is the same as before)
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    best_aic = np.inf
    best_params = None
    best_seasonal_params = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(train_data['y'],
                              exog=exog_train,
                              order=param,
                              seasonal_order=param_seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = mod.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = param
                    best_seasonal_params = param_seasonal
            except Exception:
                continue

    if best_params:
        mod = SARIMAX(train_data['y'], exog=exog_train, order=best_params, seasonal_order=best_seasonal_params)
        return mod.fit(disp=False)
    return None

def train_prophet(train_data):
    # (Implementation is the same as before)
    model = Prophet()
    model.add_regressor('pico')
    model.fit(train_data)
    return model

def create_features_for_xgb(df):
    df['month'] = df['ds'].dt.month
    df['rolling_mean_3'] = df['y'].shift(1).rolling(window=3).mean()
    df['rolling_mean_6'] = df['y'].shift(1).rolling(window=6).mean()
    for i in range(1, 13):
        df[f'lag_{i}'] = df['y'].shift(i)
    return df

def train_xgboost(train_data):
    df = create_features_for_xgb(train_data.copy())
    df = df.dropna()
    if df.empty: return None
    features = ['pico', 'month', 'rolling_mean_3', 'rolling_mean_6'] + [f'lag_{i}' for i in range(1, 13)]
    X_train, y_train = df[features], df['y']
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    return model

# (The rest of the script remains the same)
def predict_xgboost_recursive(model, train_data, steps):
    full_data = train_data.copy()
    for _ in range(steps):
        next_date = full_data['ds'].max() + pd.DateOffset(months=1)
        next_pico = 1 if next_date.month in [2, 3, 9, 10] else 0

        temp_df = pd.concat([full_data, pd.DataFrame([{'ds': next_date, 'y': np.nan, 'pico': next_pico}])], ignore_index=True)
        temp_df_featured = create_features_for_xgb(temp_df)

        last_row = temp_df_featured.iloc[-1]
        features = ['pico', 'month', 'rolling_mean_3', 'rolling_mean_6'] + [f'lag_{i}' for i in range(1, 13)]
        pred_input = last_row[features].values.reshape(1, -1)

        yhat = model.predict(pred_input)[0]

        full_data = pd.concat([full_data, pd.DataFrame([{'ds': next_date, 'y': yhat, 'pico': next_pico}])], ignore_index=True)

    return full_data.iloc[-steps:]['y'].values

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) if np.all(y_true != 0) else np.inf
    return rmse, mae, mape

def select_best_model(metrics):
    best_model = min(metrics, key=lambda k: metrics[k][0])
    best_rmse = metrics[best_model][0]
    reason = f"RMSE menor ({best_rmse:.2f})"

    for model, (rmse, mae, _) in metrics.items():
        if model != best_model and np.isclose(rmse, best_rmse, rtol=0.01) and mae < metrics[best_model][1]:
            best_model = model
            reason = f"RMSE empate -> MAE menor ({mae:.2f})"
            break
    return best_model, reason

def plot_predictions(df, sap_code, best_model):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Periodo'], df['Consumo_real'], label='Real', marker='o')
    for col in df.columns:
        if col.startswith('Pred_'):
            model_name = col.replace('Pred_', '')
            linestyle = '-' if model_name == best_model else '--'
            plt.plot(df['Periodo'], df[col], label=f'{model_name} Pred', linestyle=linestyle)
    plt.title(f'Test Set Predictions for SAP: {sap_code} (Best: {best_model})')
    plt.legend()
    plt.grid(True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f'{PLOTS_DIR}test_predictions_{sap_code}.png')
    plt.close()

if __name__ == '__main__':
    preprocessed_data = load_and_preprocess_data(DATA_FILE)
    all_metrics, all_test_predictions, all_future_forecasts = [], [], []

    if preprocessed_data is not None:
        for sap_code in SAP_CODES:
            print(f"\n--- Processing SAP: {sap_code} ---")
            sap_data = preprocessed_data[preprocessed_data['SAP'] == sap_code].copy().reset_index(drop=True)

            if len(sap_data) < 14:
                print(f"Not enough data for SAP: {sap_code}")
                continue

            train, test = train_test_split(sap_data, test_size=0.2, shuffle=False)

            models = {
                'ARIMA': train_arima(train),
                'SARIMAX': train_sarimax(train, train[['pico']]),
                'Prophet': train_prophet(train[['ds', 'y', 'pico']]),
                'XGBoost': train_xgboost(train)
            }

            test_preds = {'Periodo': test['ds'], 'Consumo_real': test['y']}
            metrics = {}

            if models['ARIMA']:
                preds = models['ARIMA'].forecast(steps=len(test))
                test_preds['Pred_ARIMA'] = preds
                metrics['ARIMA'] = calculate_metrics(test['y'], preds)

            if models['SARIMAX']:
                forecast = models['SARIMAX'].get_forecast(steps=len(test), exog=test[['pico']])
                test_preds['Pred_SARIMAX'] = forecast.predicted_mean
                conf_int = forecast.conf_int(alpha=0.05)
                test_preds['lower_95_SARIMAX'] = conf_int.iloc[:, 0].values
                test_preds['upper_95_SARIMAX'] = conf_int.iloc[:, 1].values
                metrics['SARIMAX'] = calculate_metrics(test['y'], forecast.predicted_mean)

            if models['Prophet']:
                future_df = test[['ds', 'pico']]
                forecast = models['Prophet'].predict(future_df)
                test_preds['Pred_Prophet'] = forecast['yhat'].values
                test_preds['lower_95_Prophet'] = forecast['yhat_lower'].values
                test_preds['upper_95_Prophet'] = forecast['yhat_upper'].values
                metrics['Prophet'] = calculate_metrics(test['y'], forecast['yhat'])

            if models['XGBoost']:
                preds = predict_xgboost_recursive(models['XGBoost'], train, len(test))
                test_preds['Pred_XGBoost'] = preds
                metrics['XGBoost'] = calculate_metrics(test['y'], preds)

            test_preds_df = pd.DataFrame(test_preds)
            all_test_predictions.append(test_preds_df.assign(SAP=sap_code))

            if metrics:
                best_model_name, reason = select_best_model(metrics)
                print(f"Best model for {sap_code}: {best_model_name} ({reason})")
                plot_predictions(test_preds_df, sap_code, best_model_name)

                full_model = None
                if best_model_name == 'ARIMA': full_model = train_arima(sap_data)
                elif best_model_name == 'SARIMAX': full_model = train_sarimax(sap_data, sap_data[['pico']])
                elif best_model_name == 'Prophet': full_model = train_prophet(sap_data[['ds', 'y', 'pico']])
                elif best_model_name == 'XGBoost': full_model = train_xgboost(sap_data)

                future_dates = pd.date_range(start=sap_data['ds'].max() + pd.DateOffset(months=1), end=FORECAST_END_DATE, freq='MS')
                forecast_len = len(future_dates)
                future_forecast_vals = None

                if full_model:
                    if best_model_name == 'ARIMA':
                        future_forecast_vals = full_model.forecast(steps=forecast_len)
                    elif best_model_name == 'SARIMAX':
                        future_pico = pd.DataFrame({'ds': future_dates, 'pico': [1 if d.month in [2,3,9,10] else 0 for d in future_dates]})
                        future_forecast_vals = full_model.get_forecast(steps=forecast_len, exog=future_pico[['pico']]).predicted_mean
                    elif best_model_name == 'Prophet':
                        future_df = pd.DataFrame({'ds': future_dates, 'pico': [1 if d.month in [2,3,9,10] else 0 for d in future_dates]})
                        forecast = full_model.predict(future_df)
                        future_forecast_vals = forecast['yhat'].values
                    elif best_model_name == 'XGBoost':
                        future_forecast_vals = predict_xgboost_recursive(full_model, sap_data, forecast_len)

                if future_forecast_vals is not None:
                    all_future_forecasts.append(pd.DataFrame({'SAP': sap_code, 'Periodo': future_dates, 'Forecast': future_forecast_vals}))

                summary = {'SAP': sap_code, 'Best_model': best_model_name, 'Reason_best_model': reason}
                for model, (rmse, mae, mape) in metrics.items():
                    summary[f'RMSE_{model}'], summary[f'MAE_{model}'], summary[f'MAPE_{model}'] = rmse, mae, mape
                all_metrics.append(summary)

    if all_test_predictions:
        with pd.ExcelWriter(f'{OUTPUT_DIR}metrics_by_product.xlsx') as writer:
            pd.concat(all_test_predictions).to_excel(writer, sheet_name='test_predictions', index=False)
            pd.DataFrame(all_metrics).to_excel(writer, sheet_name='summary_metrics', index=False)
            if all_future_forecasts:
                pd.concat(all_future_forecasts).to_excel(writer, sheet_name='future_forecasts', index=False)
        print(f"\nResults saved to {OUTPUT_DIR}metrics_by_product.xlsx")
