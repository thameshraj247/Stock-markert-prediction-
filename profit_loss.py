import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model # type: ignore
import math
import matplotlib.pyplot as plt

def prepare_time_series_data(data, window_size=60):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    return mape, accuracy

def show_prediction_ui():
    st.title("LSTM Stock Price Predictor")

    uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Preprocessing - replicate your notebook steps
        df['Date'] = df['date'].str.split(' ').str.get(0)
        df.drop(columns=['date', 'symbol'], inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        df_used = df[['close', 'high', 'low', 'open']]

        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_used), columns=df_used.columns, index=df_used.index)

        # Train/test split same as notebook (80/20)
        train_size = int(len(df_scaled) * 0.8)
        train_data = df_scaled.iloc[:train_size]
        test_data = df_scaled.iloc[train_size:]

        # Prepare test sequences (window=60)
        X_test = prepare_time_series_data(test_data.values, 60)

        # Load your pre-trained models (make sure you have these saved)
        lstm1 = load_model('LSTM1_model.keras')
        lstm2 = load_model('LSTM2_model.keras')

        # Predict
        preds_lstm1 = lstm1.predict(X_test)
        preds_lstm2 = lstm2.predict(X_test)

        # Inverse scale predictions
        preds_lstm1_inv = scaler.inverse_transform(preds_lstm1)
        preds_lstm2_inv = scaler.inverse_transform(preds_lstm2)

        # Inverse scale true values
        y_true = test_data.iloc[60:].copy()  # Because first 60 are used as input window
        y_true_inv = scaler.inverse_transform(y_true)

        # Build DataFrame for comparison
        df_results = pd.DataFrame({
            'Actual Close': y_true_inv[:, 0],
            'Predicted Close LSTM1': preds_lstm1_inv[:, 0],
            'Predicted Close LSTM2': preds_lstm2_inv[:, 0],
        }, index=y_true.index)

        st.subheader("Actual vs Predicted Close Prices")
        st.line_chart(df_results)

        # Calculate accuracy
        mape1, acc1 = calculate_mape(df_results['Actual Close'], df_results['Predicted Close LSTM1'])
        mape2, acc2 = calculate_mape(df_results['Actual Close'], df_results['Predicted Close LSTM2'])

        st.write(f"LSTM1 Model Accuracy: {acc1:.2f}% (MAPE: {mape1:.2f}%)")
        st.write(f"LSTM2 Model Accuracy: {acc2:.2f}% (MAPE: {mape2:.2f}%)")

        # Show raw data preview
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

if __name__ == "__main__":
    show_prediction_ui()
