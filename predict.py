import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import warnings
warnings.simplefilter(action='ignore')

def prepare_time_series_data(Data, window_size):
    sequences = []
    labels = []
    for i in range(len(Data) - window_size):
        sequence = Data[i : i + window_size]
        label = Data.iloc[i + window_size]
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def predict_and_inverse_transform(DF, X_test, model, scaler):
    test = DF.iloc[-len(X_test):].copy()
    predictions = model.predict(X_test)
    inverse_predictions = scaler.inverse_transform(predictions)
    inverse_predictions = pd.DataFrame(inverse_predictions,
                                       columns=['Predicted Close', 'Predicted High', 'Predicted Low', 'Predicted Open'],
                                       index=DF.iloc[-len(X_test):].index)
    test_df = pd.concat([test.copy(), inverse_predictions], axis=1)
    test_df[['close', 'high', 'low', 'open']] = scaler.inverse_transform(test_df[['close', 'high', 'low', 'open']])
    return test_df

def calculate_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    return mape, accuracy

def show_prediction_ui():
    st.title("📊 Full Stock Price LSTM Model Training & Prediction")

    uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])
    if not uploaded_file:
        st.warning("Please upload a stock CSV file to begin.")
        return

    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    try:
        df['Date'] = df['date'].str.split(' ').str.get(0)
        df.drop(columns=['date','symbol'], inplace=True)
    except KeyError:
        st.error("CSV must contain 'date' and 'symbol' columns.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    DF = df[['close','high','low','open']].copy()
    scaler = MinMaxScaler()
    DF[DF.columns] = scaler.fit_transform(DF)

    st.write(f"Dataset shape after scaling: {DF.shape}")

    training_size = round(len(DF) * 0.80)
    train_data = DF.iloc[:training_size, :]
    test_data = DF.iloc[training_size:, :]

    st.write(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

    window_size = 60
    X_train, y_train = prepare_time_series_data(train_data, window_size)
    X_test, y_test = prepare_time_series_data(test_data, window_size)

    st.write(f"Prepared training data shape: {X_train.shape}, labels shape: {y_train.shape}")
    st.write(f"Prepared testing data shape: {X_test.shape}, labels shape: {y_test.shape}")

    # Reshape for LSTM input (samples, time_steps, features) - Already done as numpy array from DataFrame slices
    # Just ensure dtype float32 for keras compatibility
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    early_stop = EarlyStopping(monitor='loss', patience=5)

    # LSTM Model 1
    st.header("Training LSTM Model 1")
    LSTM1 = Sequential()
    LSTM1.add(LSTM(100, return_sequences=True, input_shape=(window_size, X_train.shape[2])))
    LSTM1.add(Dropout(0.2))
    LSTM1.add(LSTM(100, return_sequences=False))
    LSTM1.add(Dropout(0.2))
    LSTM1.add(Dense(X_train.shape[2]))

    LSTM1.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    st.write(LSTM1.summary())

    with st.spinner("Training LSTM1... This may take a while."):
        history1 = LSTM1.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test),
                             batch_size=32, callbacks=[early_stop], verbose=0)

    st.success("LSTM Model 1 training completed!")

    # LSTM Model 2
    st.header("Training LSTM Model 2")
    LSTM2 = Sequential()
    LSTM2.add(LSTM(150, input_shape=(window_size, X_train.shape[2]), return_sequences=True))
    LSTM2.add(Dropout(0.2))
    LSTM2.add(LSTM(100, return_sequences=True))
    LSTM2.add(Dropout(0.2))
    LSTM2.add(LSTM(100, return_sequences=False))
    LSTM2.add(Dropout(0.2))
    LSTM2.add(Dense(units=50))
    LSTM2.add(Dense(units=5))
    LSTM2.add(Dense(X_train.shape[2]))

    LSTM2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    st.write(LSTM2.summary())

    with st.spinner("Training LSTM2... This may take a while."):
        history2 = LSTM2.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test),
                             batch_size=32, callbacks=[early_stop], verbose=0)

    st.success("LSTM Model 2 training completed!")

    # Plot losses for LSTM2
    st.subheader("LSTM2 Training Loss and MAE")
    fig, ax = plt.subplots(figsize=(10,6))
    pd.DataFrame(history2.history).plot(ax=ax)
    ax.set_title("Loss and Mean Absolute Error vs. Epochs")
    ax.set_xlabel("Epochs")
    st.pyplot(fig)

    # Predictions and inverse transform
    test_df1 = predict_and_inverse_transform(DF, X_test, LSTM1, scaler)
    test_df2 = predict_and_inverse_transform(DF, X_test, LSTM2, scaler)

    # Plot comparison of actual vs predicted close price (LSTM2)
    st.subheader("Actual vs Predicted Close Prices (LSTM2)")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    test_df2['close'].plot(label='Close Price', ax=ax2)
    test_df2['Predicted Close'].plot(label='Predicted Close Price', ax=ax2)
    ax2.set_title('Comparison of Actual and Predicted Close Prices')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # Add error columns
    test_df1['Close Price Error (LSTM1)'] = test_df1['close'] - test_df1['Predicted Close']
    test_df2['Close Price Error (LSTM2)'] = test_df2['close'] - test_df2['Predicted Close']

    st.subheader("Sample Close Price Differences")
    st.write("LSTM1 Close Price Differences:")
    st.dataframe(test_df1[['close', 'Predicted Close', 'Close Price Error (LSTM1)']].head())

    st.write("LSTM2 Close Price Differences:")
    st.dataframe(test_df2[['close', 'Predicted Close', 'Close Price Error (LSTM2)']].head())

    # Accuracy calculations
    mape1, accuracy1 = calculate_accuracy(test_df1['close'], test_df1['Predicted Close'])
    mape2, accuracy2 = calculate_accuracy(test_df2['close'], test_df2['Predicted Close'])

    st.subheader("Model Accuracy")
    st.write(f"LSTM1 Accuracy: {accuracy1:.2f}%")
    st.write(f"LSTM2 Accuracy: {accuracy2:.2f}%")

    # Accuracy comparison bar chart
    st.subheader("Accuracy Comparison")
    fig3, ax3 = plt.subplots(figsize=(6,4))
    bars = ax3.bar(['LSTM1', 'LSTM2'], [accuracy1, accuracy2], color=['skyblue', 'lightgreen'])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Model Accuracy Comparison')
    ax3.grid(axis='y')

    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    st.pyplot(fig3)

if __name__ == "__main__":
    show_prediction_ui()
