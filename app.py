from flask import Flask, render_template, request, redirect
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import re
import nltk


nltk.download('stopwords')

app = Flask(__name__)
CLEANING_REGEX = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def setup():
    global tokenizer
    global model
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model('product-rev-sent-v1.2.h5')

def clean(text):
  stop_words = stopwords.words('english')
  text = text.lower()
  text = re.sub(CLEANING_REGEX, ' ', text)
  text = " ".join(i for i in text.split() if i not in stop_words)
  return text

def decode_sentiment(score):
    return "Positive 😁" if score>0.5 else "Negative 😡"

def pre_process(reviews):
    reviews['cleaned'] = reviews['content'].apply(lambda text:  clean(text))
    reviews_sequences = pad_sequences(tokenizer.texts_to_sequences(reviews.cleaned),
                        maxlen = 30)
    return reviews_sequences

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text_frame = [request.form['review']]
        text_frame = pad_sequences(tokenizer.texts_to_sequences(text_frame),
                            maxlen = 30)

        score = model.predict(text_frame, verbose=2, batch_size=1)
        response = dict()
        response['prediction'] = decode_sentiment(score) 
        response['score'] = str(score[0][0]) 
        return render_template('index.html', prediction=response['prediction'], confidence = response['score'])
    else:
        return render_template('index.html')
    

if __name__ == "__main__":
    setup()
    app.run(debug=True)