from flask import Flask, render_template, request
import re
import pandas as pd
import langid

app = Flask(__name__, static_folder='static')

# Load English and Hindi words
words_df = pd.read_excel('english_words.xlsx')
english_words = words_df['English'].tolist()
hindi_words = words_df['Hindi'].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_language', methods=['POST'])
def detect_language():
    sentence = request.form['sentence']
    words = re.findall(r'\w+', sentence)
    results = []

    def detect_language(word):
        if word.lower() in english_words:
            return 'English'
        elif word.lower() in hindi_words:
            return 'Hindi'
        else:
            lang, _ = langid.classify(word)
            if lang == 'hi':
                return 'Hindi'
            else:
                return 'Other'

    for word in words:
        language = detect_language(word)
        results.append({'word': word, 'language': language})

    return render_template('index.html', results=results, sentence=sentence)

if __name__ == '__main__':
    app.run(debug=True)
