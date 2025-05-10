
# lstm_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# --- Load and prepare Tata Motors data ---
tata = pd.read_csv("Tata Motors Limited.csv")
tata['Date'] = pd.to_datetime(tata['Date'])
tata = tata.sort_values('Date')
tata.set_index('Date', inplace=True)
tata['Close'] = tata['Close'].astype(str).str.replace(',', '').astype(float)
tata_close = tata[['Close']]

# --- Scale Tata Motors data ---
tata_scaler = MinMaxScaler()
tata_scaled = tata_scaler.fit_transform(tata_close)

# --- Create sequences ---
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_len = 60
X_tata, y_tata = create_sequences(tata_scaled, seq_len)
X_tata = X_tata.reshape((X_tata.shape[0], X_tata.shape[1], 1))

# --- Train Tata Motors model ---
tata_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_tata.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
tata_model.compile(optimizer='adam', loss='mean_squared_error')
tata_model.fit(X_tata, y_tata, epochs=10, batch_size=32)
tata_model.save("tata_model.h5")
tata_last_seq = tata_scaled[-60:].reshape(1, 60, 1)
tata_pred = tata_model.predict(tata_last_seq)
print("Predicted Tata Motors Close:", tata_scaler.inverse_transform(tata_pred)[0][0])

# --- Load and prepare Infosys data ---
infosys = pd.read_csv("Infosys Limited.csv")
infosys['Date'] = pd.to_datetime(infosys['Date'], format="%d-%b-%y", errors='coerce')
infosys.dropna(subset=['Date'], inplace=True)
infosys = infosys.sort_values('Date')
infosys.set_index('Date', inplace=True)
infosys['Close'] = infosys['Close'].astype(str).str.replace(',', '').astype(float)
infosys_close = infosys[['Close']]

# --- Scale Infosys data ---
infosys_scaler = MinMaxScaler()
infosys_scaled = infosys_scaler.fit_transform(infosys_close)
X_infosys, y_infosys = create_sequences(infosys_scaled, seq_len)
X_infosys = X_infosys.reshape((X_infosys.shape[0], X_infosys.shape[1], 1))

# --- Train Infosys model ---
infosys_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_infosys.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
infosys_model.compile(optimizer='adam', loss='mean_squared_error')
infosys_model.fit(X_infosys, y_infosys, epochs=10, batch_size=32)
infosys_model.save("infosys_model.h5")
infosys_last_seq = infosys_scaled[-60:].reshape(1, 60, 1)
infosys_pred = infosys_model.predict(infosys_last_seq)
print("Predicted Infosys Close:", infosys_scaler.inverse_transform(infosys_pred)[0][0])
