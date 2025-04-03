# from flask import Flask, request, jsonify, render_template
# import requests
# from bs4 import BeautifulSoup
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import pickle

# app = Flask(__name__)

# # Load trained model
# model = tf.keras.models.load_model("best_model.h5")

# # Load the tokenizer that was used during training
# with open("tokenizer.pickle", "rb") as handle:
#     tokenizer = pickle.load(handle)

# # Tokenization parameters
# MAX_SEQUENCE_LENGTH = 250  # Ensure this matches the training script

# def fetch_news_content(url):
#     """Fetch news content from a given URL."""
#     headers = {'User-Agent': 'Mozilla/5.0'}
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, "html.parser")
#         paragraphs = soup.find_all('p')
#         return ' '.join([p.get_text() for p in paragraphs])
#     return ""

# def preprocess_text(text):
#     """Preprocess the text data."""
#     if not text or text.strip() == "":  # Check for empty text
#         return np.zeros((1, MAX_SEQUENCE_LENGTH))  # Return padded zero sequence
    
#     sequences = tokenizer.texts_to_sequences([text])
    
#     if not sequences or sequences == [[]]:  # Check if sequence is empty
#         return np.zeros((1, MAX_SEQUENCE_LENGTH))  # Return padded zero sequence
    
#     padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
#     return padded_sequences

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
#     news_url = data.get("news_url")
    
#     if not news_url:
#         return jsonify({"error": "No URL provided"}), 400

#     news_content = fetch_news_content(news_url)
    
#     if not news_content:
#         return jsonify({"error": "Could not fetch news content"}), 400

#     processed_text = preprocess_text(news_content)
    
#     prediction = model.predict(processed_text)[0][0]
#     sentiment = "Positive" if prediction > 0.5 else "Negative"
    
#     return jsonify({"sentiment": sentiment})

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("financial_sentiment_lstm.h5")

# Load the tokenizer that was used during training
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Tokenization parameters
MAX_SEQUENCE_LENGTH = 250  # Ensure this matches the training script

def fetch_news_content(url):
    """Fetch news content from a given URL."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    return ""

def preprocess_text(text):
    """Preprocess the text data."""
    if not text or text.strip() == "":  # Check for empty text
        return np.zeros((1, MAX_SEQUENCE_LENGTH))  # Return padded zero sequence
    
    sequences = tokenizer.texts_to_sequences([text])
    
    if not sequences or sequences == [[]]:  # Check if sequence is empty
        return np.zeros((1, MAX_SEQUENCE_LENGTH))  # Return padded zero sequence
    
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    return padded_sequences

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    news_url = data.get("news_url")
    
    if not news_url:
        return jsonify({"error": "No URL provided"}), 400

    news_content = fetch_news_content(news_url)
    
    if not news_content:
        return jsonify({"error": "Could not fetch news content"}), 400

    processed_text = preprocess_text(news_content)
    
    prediction = model.predict(processed_text)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)