from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pickle


app = Flask(__name__)

# Load the trained model
model = joblib.load('fish_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    species = data['species']
    length1 = float(data['length1'])
    length2 = float(data['length2'])
    length3 = float(data['length3'])
    height = float(data['height'])
    width = float(data['width'])

    # One-hot encode the species
    species_dict = {'Bream': 0, 'Roach': 1, 'Pike': 2, 'Smelt': 3, 'Perch': 4, 'Parkki': 5, 'Whitefish': 6}
    species_encoded = [0] * 6
    species_encoded[species_dict[species]] = 1

    features = [length1, length2, length3, height, width] + species_encoded
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)
    return jsonify({'weight': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
