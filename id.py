import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping

# Load your combined dataset (English and Hindi words)
# Assuming you have a DataFrame with columns 'Word' and 'Language'
# Replace 'your_dataset.csv' with your actual file path
df = pd.read_excel('english_words.xlsx')

# Convert the 'English' and 'Hindi' columns to strings
df['English'] = df['English'].astype(str)
df['Hindi'] = df['Hindi'].astype(str)

# Preprocess text: convert to lowercase, remove punctuation and special characters
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply preprocessing to English and Hindi columns
df['English'] = df['English'].apply(preprocess_text)
df['Hindi'] = df['Hindi'].apply(preprocess_text)

# Combine English and Hindi words into a single list
all_words = df['English'].tolist() + df['Hindi'].tolist()

# Tokenize words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_words)
vocab_size = len(tokenizer.word_index) + 1

# Convert words to sequences
english_sequences = tokenizer.texts_to_sequences(df['English'])
hindi_sequences = tokenizer.texts_to_sequences(df['Hindi'])

# Pad sequences to a consistent length
max_sequence_length = max(len(seq) for seq in english_sequences + hindi_sequences)
padded_english = pad_sequences(english_sequences, maxlen=max_sequence_length, padding='post')
padded_hindi = pad_sequences(hindi_sequences, maxlen=max_sequence_length, padding='post')

# Create labels (0 for English, 1 for Hindi)
labels = np.array([0] * len(english_sequences) + [1] * len(hindi_sequences))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(np.vstack((padded_english, padded_hindi)), labels, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))  # Return sequences for the next LSTM layer
model.add(LSTM(64))  # Second LSTM layer
model.add(Dense(64, activation='relu'))  # Dense layer
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(2, activation='softmax'))  # Output layer

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[early_stopping])

# Save the model
model.save('language_classification_model.h5')

# Load the model (if needed)
model = tf.keras.models.load_model('language_classification_model.h5')

# Evaluation
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_val, y_pred_classes))

# Prompt the user for input
test_sentence = input("Enter a sentence: ")
test_sentence = preprocess_text(test_sentence)  # Preprocess the input sentence
test_words = test_sentence.split()

# Tokenize and pad each word separately
test_sequences = [tokenizer.texts_to_sequences([word]) for word in test_words]
padded_test_sequences = [pad_sequences(seq, maxlen=max_sequence_length, padding='post') for seq in test_sequences]

# Predict the language of each word
for word, seq in zip(test_words, padded_test_sequences):
    prediction = model.predict(seq)
    predicted_language = 'English' if np.argmax(prediction) == 0 else 'Hindi'
    print(f"{word}: {predicted_language}")