# python_service/app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('path/to/your/model.h5')

def one_hot_encode(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = {aa: i for i, aa in enumerate(amino_acids)}
    encoding = np.zeros((1000, len(amino_acids)), dtype=np.float32)
    for i, aa in enumerate(sequence[:1000]):
        encoding[i, aa_dict[aa]] = 1
    return encoding

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sequence = data['sequence']
    encoded_sequence = one_hot_encode(sequence)
    prediction = model.predict(np.array([encoded_sequence]))
    result = prediction.argmax(axis=-1).tolist()  # Assuming categorical output
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)