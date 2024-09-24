import pickle
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing import sequence

from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_keras_model():
    global model
    # Load the model using the SavedModel format for better compatibility
    model = load_model('models/uci_sentimentanalysis.keras')

def load_tokenizer():
    global tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

@app.before_first_request
def before_first_request():
    load_keras_model()
    load_tokenizer()

def sentiment_analysis(input):
    user_sequences = tokenizer.texts_to_sequences([input])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]), 2)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        text = request.form.get("user_text")

        # VADER sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)  # VADER results

        # Custom model sentiment analysis
        sentiment["custom model positive"] = sentiment_analysis(text)

    return render_template('form.html', sentiment=sentiment)

if __name__ == "__main__":
    app.run()