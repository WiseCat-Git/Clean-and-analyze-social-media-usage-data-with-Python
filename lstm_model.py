import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the cleaned data
df_cleaned = pd.read_csv("cleaned_data_with_date.csv")

# Inspect the unique values in the target column
print("Unique target values before conversion:", df_cleaned['target'].unique())

# Convert to binary classification (e.g., map 0-2 to 0 (negative) and 3-4 to 1 (positive))
df_cleaned['target'] = df_cleaned['target'].apply(lambda x: 0 if x < 2 else 1)

# Verify the conversion
print("Unique target values after conversion:", df_cleaned['target'].unique())

# Split the data into features (X) and target (y)
X = df_cleaned['text']
y = df_cleaned['target']

# Tokenization
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(X)
X_tokenized = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_tokenized, maxlen=100)

# Convert target to categorical (binary classification)
y = to_categorical(y, num_classes=2)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Building the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=X_padded.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))  # Adjust the output layer for 2 classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
score, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {acc:.2f}")

# Plot accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()
