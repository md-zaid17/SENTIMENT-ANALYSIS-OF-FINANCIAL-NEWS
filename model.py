import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
df = pd.read_csv(r"C:\Users\bhatt\OneDrive\Desktop\SENTI NEWZZ PROJECT\Financial Market News.csv", encoding="ISO-8859-1")

df.columns = ["Sentiment", "News"]  # Rename columns

# Encode sentiment labels
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Sentiment"])

# Save label encoder
with open("label_encoder.pickle", "wb") as le_file:
    pickle.dump(label_encoder, le_file)

# Tokenization
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 250
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df["News"])

# Save tokenizer
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df["News"])
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
labels = np.array(df["Label"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels)

# Compute class weights correctly
class_weights_array = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels)
class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}  # Convert to dictionary format

# Build Bi-LSTM model
model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128, input_length=MAX_SEQUENCE_LENGTH),
    Bidirectional(LSTM(128, return_sequences=True)),  # Bi-LSTM for better learning
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model with class weights
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights)

# Save model
model.save("financial_sentiment_lstm.h5")
print("Model training complete and saved as 'financial_sentiment_lstm.h5'.")
