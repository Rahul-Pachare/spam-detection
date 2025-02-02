from flask import Flask, request, jsonify
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model and vectorizer
model = load_model('spam_model.h5')
tfidf = joblib.load('tfidf_vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get('text', '')
    if not email_text:
        return jsonify({'error': 'No text provided'}), 400

    # Predict spam probability
    email_tfidf = tfidf.transform([email_text]).toarray()
    spam_probability = model.predict(email_tfidf)[0][0] * 100
    return jsonify({'spam_probability': round(spam_probability, 2)})

if __name__ == '__main__':
    app.run(debug=True)