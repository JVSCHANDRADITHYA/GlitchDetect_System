import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

# -------------------------
# 1. Load Data
# -------------------------
df = pd.read_csv("sensor_data.csv")   # replace with your file

# If it has multiple columns, pick one or use them all
values = df.values.astype(float)

# -------------------------
# 2. Normalize
# -------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# -------------------------
# 3. Create Sequences
# -------------------------
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i+seq_len)]
        y = data[i:(i+seq_len)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 50
X, y = create_sequences(scaled, SEQ_LEN)

# Train/test split
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -------------------------
# 4. LSTM Autoencoder
# -------------------------
model = Sequential([
    LSTM(64, activation='relu', input_shape=(SEQ_LEN, X.shape[2]), return_sequences=False),
    RepeatVector(SEQ_LEN),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X.shape[2]))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# -------------------------
# 5. Train
# -------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)

# -------------------------
# 6. Reconstruction Error
# -------------------------
X_pred = model.predict(X_test)
reconstruction_error = np.mean(np.power(X_test - X_pred, 2), axis=(1,2))

# -------------------------
# 7. Threshold for anomaly
# -------------------------
threshold = np.mean(reconstruction_error) + 3*np.std(reconstruction_error)
print("Threshold: ", threshold)

anomalies = reconstruction_error > threshold

# -------------------------
# 8. Plot
# -------------------------
plt.figure(figsize=(15,6))
plt.plot(reconstruction_error, label="Reconstruction error")
plt.hlines(threshold, xmin=0, xmax=len(reconstruction_error), colors='r', label="Threshold")
plt.legend()
plt.show()

# Plot anomalies on original signal
plt.figure(figsize=(15,6))
plt.plot(scaled[train_size+SEQ_LEN:], label="Sensor signal")
plt.scatter(np.where(anomalies)[0], scaled[train_size+SEQ_LEN:][anomalies], color='r', label="Anomaly")
plt.legend()
plt.show()
