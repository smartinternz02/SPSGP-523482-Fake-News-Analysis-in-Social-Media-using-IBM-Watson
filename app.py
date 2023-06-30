# Importing the Libraries
import nltk
nltk.download("punkt")
import flask
from flask import Flask, request, render_template
import pickle
import os
from newspaper import Article
import urllib
from flask_cors import CORS



app = Flask(__name__,)
CORS(app)
app = flask.Flask(__name__, template_folder='template', static_url_path='/static')
with open(r"C:\Users\Jigyas Boruah\Downloads\Fake news in social media using IBM watson\fake_news (2).pkl", 'rb') as handle:
    model = pickle.load(handle)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    pred = model.predict([news])
    return render_template('main.html', prediction_text='The news is "{}"'.format(pred[0]))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=False, threaded=True, use_reloader=False)
