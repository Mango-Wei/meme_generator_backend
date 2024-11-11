from flask import Flask, request, jsonify
import joblib
import os
from openai import OpenAI

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model_path = 'best_random_forest_model.joblib'
loaded_model = joblib.load(model_path)

# OpenAI setup
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# Define routes for interaction
@app.route('/generate_meme', methods=['POST'])
def generate_meme():
    data = request.json
    user_text = data['text']
    
    # Your model and OpenAI code logic here
    # Perform model inference, create a meme template, etc.
    
    response = {
        "message": "Meme generated successfully",
        "meme_url": "path/to/generated_meme.jpg"  # Return the URL or path to the generated meme
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
