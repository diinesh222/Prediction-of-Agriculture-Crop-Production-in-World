import os
from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

print(sklearn.__version__)

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the absolute paths to the model files
dtr_path = os.path.join(script_dir, 'dtr.pkl')
preprocessor_path = os.path.join(script_dir, 'models', 'preprocessor.pkl')

# Verify the existence of the model files
if not os.path.exists(dtr_path):
    print(f"File not found: {dtr_path}")
if not os.path.exists(preprocessor_path):
    print(f"File not found: {preprocessor_path}")

# List the contents of the models directory to verify file presence
models_dir = os.path.join(script_dir, 'models')
print(f"Contents of {models_dir}:")
print(os.listdir(models_dir))

# Load models
dtr = pickle.load(open(dtr_path, 'rb'))
preprocessor = pickle.load(open(preprocessor_path, 'rb'))
print(os.listdir(os.path.join(script_dir, 'models')))

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', prediction=prediction[0][0])

if __name__ == "__main__":
    app.run(debug=True)
