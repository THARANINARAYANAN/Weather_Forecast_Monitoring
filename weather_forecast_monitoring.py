import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

data_path= r'/content/drive/MyDrive/weather_data.csv'
df = pd.read_csv(data_path)

df.columns = df.columns.str.lstrip()
df.columns = df.columns.str.rstrip()
df.head()

plt.figure(figsize=(15, 10))
sns.histplot(df._tempm, bins = [i for i in range(0, 61, 5)], kde=False)
plt.title('Distributiom of Temperature')
plt.grid()
plt.show()

df.index = pd.to_datetime(df.datetime_utc)
df.head()

required_cols = ['_dewptm','_fog','_hail','_hum','_rain','_snow','_tempm','_thunder','_tornado']
df = df[required_cols]
df.head()

df.isna().sum()
df = df.fillna(method = 'ffill')
df.head()

df.isna().sum()
df_final = df.resample('D').mean()
df_final.head()

df_final.isna().sum()
df_final = df_final.fillna(method = 'ffill')
df_final.head()

df_final.head(12)

from sklearn.preprocessing import MinMaxScaler

#Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_final)

#Define sequence length and features
sequence_length = 10 #Number of time steps in each sequence
num_features = len(df_final.columns)

# Create sequences and corresponding labels
sequences = []
labels = []
for i in range(len(scaled_data) - sequence_length):
    seq = scaled_data[i:i+sequence_length]
    label = scaled_data[i+sequence_length][6] # tempm" column index
    sequences.append(seq)
    labels.append(label)

# Convert sequences and labels to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

#Split into train and test sets
train_size = int(0.8 * len(sequences))
train_x, test_x = sequences[:train_size], sequences [train_size:]
train_y, test_y =  labels[:train_size], labels [train_size:]

print("Train X shape:", train_x.shape)
print("Train Y shape:", train_y.shape)
print("Test x shape:", test_x.shape)
print("Test y shape:", test_y.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Create the LSTM model
model = Sequential()

# Add LSTM layers with dropout
model.add(LSTM(units=128, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dropout (0.2))

model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout (0.2))

model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout (0.2))

# Add a dense output layer
model.add(Dense (units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

early_stopping = EarlyStopping (monitor= 'val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/best_model_weights.keras', monitor='val_loss', save_best_only=True)
# Train the model
history = model.fit (
    train_x, train_y,
    epochs=70,
    batch_size=32,
    validation_split=0.2, # Use part of the training data as validation
    callbacks=[early_stopping, model_checkpoint]

import tensorflow as tf

# Load the model from the .keras file (saved during training)
best_model = tf.keras.models.load_model('/content/drive/MyDrive/best_model_weights.keras')

# Evaluate the best model on the test set
test_loss = best_model.evaluate(test_x, test_y)
print("Test Loss:", test_loss)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt. legend (['Train', 'Validation'], loc='upper right')
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
# Assuming you have trained the model and have the 'best_model' object
# Also, 'test_x' and 'test_y' should be available
# Predict temperatures using the trained model
predictions = best_model.predict(test_x)
# Calculate evaluation metrics
mae = mean_absolute_error(test_y, predictions)
mse = mean_squared_error(test_y, predictions)
rmse = np.sqrt(mse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse) # Changed 'rase' to 'rmse'

# y_true values
test_y_copies = np.repeat(test_y.reshape(-1, 1), test_x.shape[-1], axis=-1)
true_temp = scaler.inverse_transform (test_y_copies) [:,6]
# predicted values
prediction = best_model.predict(test_x)
prediction_copies = np.repeat(prediction, 9, axis=-1)
predicted_temp = scaler.inverse_transform(prediction_copies)[:, 6]

# Plotting predicted and actual temperatures
plt.figure(figsize=(10, 6))
plt.plot(df_final.index [-100:], true_temp[-100:], label='Actual')
plt.plot(df_final. index [-100:], predicted_temp [-100: ], label='Predicted')
plt.title('Temperature Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()

!pip install tensorflow

import tensorflow as tf

# Assuming 'test_y' and 'predictions' are your true and predicted values
mape = tf.keras.metrics.MeanAbsolutePercentageError()
mape.update_state(test_y, predictions)  # Update the metric state
mape_result = mape.result().numpy()  # Get the result as a NumPy value

accuracy = 100 - mape_result

print(f"MAPE: {mape_result}")
print(f"Accuracy: {accuracy}%")

