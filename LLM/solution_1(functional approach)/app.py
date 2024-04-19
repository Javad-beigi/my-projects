from flask import Flask, request, jsonify
from transformers import pipeline

#create a flask application instance
app = Flask(__name__)


# initialze pipelines for translation, idiom detection, idiom extraction, and sentiment analysis
translator = pipeline('translation',model='Helsinki-NLP/opus-mt-en-es')
idiom_detector = pipeline('text-classification',model='joeddav/xlm-roberta-large-xnli-anli')
idiom_extractor = pipeline('text-classification',model='anton-l/roberta-large-sentiment')
sentiment_analyzer = pipeline('sentiment-analysis',model='nlptown/bert-base-multilingual-uncased-sentiment')


# define route for translating text
@app.route('/translate' , methods=['POST'])
def translate_text():
    data = request.json  # extract JSON data from request
    text = data['text']
    target_language = data['es']


    #translate text using specified source and target languages
    translated_text = translator(text,es=target_language)[0]['translation_text']

    #return translated text as JSON response

    return jsonify({'translated_text': translated_text})



# define route for detecting idioms in text
@app.route('/detect_idioms', methods=['POST'])
def detect_idioms():
    data = request.json
    text = data['text']

    #detect idioms in text using idiom detection pipeline
    idiom_prediction = idiom_detector(text)[0]
    # check if the text contains an idiom
    is_idiom = idiom_prediction['label'] == 'LABEL_1'
    # GET confidence score of idiom detection
    confidence = idiom_prediction['score']

    return jsonify({'is_idiom': is_idiom, 'confidence': confidence})



# define route for extracting idioms from text
@app.route('/extract_idioms', methods=['POST'])
def extract_idioms():
    data = request.json
    text = data['text']

    # extract idioms from text using idiom extraction pipeline
    idiom_extraction = idiom_extractor(text)[0]
    #get extracted idiom
    idiom = idiom_extraction['label']
    confidence = idiom_extraction['score']

    return jsonify({'idiom': idiom, 'confidence' : confidence})



#define route for analyzing sentiment of text
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data['text']

    #analyze sentiment of text using sentiment analysis pipeline
    sentiment = sentiment_analyzer(text)[0]['label'] # get sentiment label

    return jsonify({'sentiment': sentiment})


# Start the Flask Application

if __name__ == '__main__':
    app.run(debug=True)





