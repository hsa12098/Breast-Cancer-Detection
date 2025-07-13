import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import os
import shutil
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow Cross-Origin Resource Sharing (CORS)

# Load the trained model
model = load_model('final_model_using_class_weights.h5')

# Ensure 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Preprocess the image for prediction
def preprocess_image(image_file):
    img = load_img(image_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Predict class and confidence
def prediction(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = "Malignant" if prediction[0][0] > 0.5 else "Benign"
    confidence = prediction[0][0] * 100
    return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def doSomething():
    image = request.files['image']
    if not image:
        return jsonify({"error": "No image provided"}), 400

    # Save uploaded image as 'image.jpg' in static/
    temp_path = 'static/image.jpg'
    image.save(temp_path)

    # Make prediction
    pred_class, confidence = prediction(temp_path)

    return jsonify({
        'prediction': pred_class,
        #'confidence': f'{confidence:.2f}%',
        'contoured_image_url': f'http://localhost:5000/static/image.jpg'
    })

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)