from flask import Flask, send_from_directory, request, jsonify
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Load the pre-trained model
model = load_model('best_model_v2.keras')

@app.route('/')
def home():
    return send_from_directory('../public', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.json['input_data']
    
    # Preprocess the input data as needed
    # (This will depend on how your model expects the data)
    
    # Convert the input data into the format your model expects
    # For example, you may need to reshape it or convert it to a numpy array
    # processed_input = ...

    # Make predictions using the model
    # prediction = model.predict(processed_input)
    
    # Process the output (if needed) and return it as a response
    # return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)