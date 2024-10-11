import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

model = None
tokenizer = None

def load_keras_model():
    global model
    model = load_model('models/uci_sentimentanalysis.h5')

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
    return round(float(prediction[0][0]) * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = dict()
    text = ""
    if request.method == "POST":
        text = request.form.get("user_text")
        sentiment = analyzer.polarity_scores(text)
        sentiment["custom model positive"] = sentiment_analysis(text)
    return render_template('form.html', sentiment=sentiment, user_text=text)

if __name__ == "__main__":
    app.run()
