import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from pssp_model.pssp_model import preprocess_sequences, decode_structures
import numpy as np


# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for the templates and static folders
template_dir = os.path.join(current_dir, '../public/templates')
static_dir = os.path.join(current_dir, '../public/static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Load the pre-trained model
model = load_model(os.path.join(current_dir, '../best_model_v2.keras'))

@app.route('/')
def home():
    return render_template('pssp.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['input_data']
    
    # Assuming input_data is a single amino acid sequence provided as a string
    if isinstance(input_data, str):
        # Preprocess the sequence using the preprocess_sequences function
        processed_input = preprocess_sequences(input_data)
        
        # Reshape the processed input to match the model's expected input shape
        processed_input = np.expand_dims(processed_input, axis=0)  # Add batch dimension
        
        # Make predictions using the model
        prediction = model.predict(processed_input)

        # Trim the prediction to match the input sequence length
        trimmed_prediction = prediction[0][:len(input_data)]

        # # Debugging: Print input length and prediction shape
        # print(f"Input length: {len(input_data)}, Prediction shape: {prediction.shape}")
        
        # Decode the predicted output back to the secondary structure string
        decoded_prediction = decode_structures(trimmed_prediction)

        # # Debugging: Print the decoded prediction length
        # print(f"Decoded prediction length: {len(decoded_prediction)}")
        
        # Return the decoded prediction as a JSON response
        return jsonify({'prediction': decoded_prediction})
    
    else:
        return jsonify({'error': 'Invalid input data. Please provide a valid amino acid sequence.'}), 400
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)