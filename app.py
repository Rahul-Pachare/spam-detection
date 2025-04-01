from flask import Flask, request, jsonify
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model and vectorizer
model = load_model('spam_classifier_model.h5')
tfidf = joblib.load('tfidf_vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json
    email_text = data.get('text', '')
    print(email_text)
    # Check if text is provided
    if not email_text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Transform the input text using the TF-IDF vectorizer
    email_tfidf = tfidf.transform([email_text]).toarray()
    
    # Predict spam probability using the model
    spam_probability = model.predict(email_tfidf)[0][0] * 100
    
    # Convert float32 to native Python float
    spam_probability = float(spam_probability)
    print(spam_probability)
    # Return the result as JSON
    return jsonify({'spam_probability': round(spam_probability, 2)})

if __name__ == '__main__':
     app.run(debug=True, host='0.0.0.0', port=8000)