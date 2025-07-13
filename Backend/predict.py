from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
from PIL import Image
import os




def preprocess_image(image_file):
    img = load_img(image_file, target_size=(224, 224))  # Resize to match model input size
    img_array = img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match batch format
    img_array /= 255.0  # Normalize pixel values (if required)

    return img_array

def predict():
    model = load_model('final_model_using_class_weights.h5')

    img_path = "Benign.jpg"
    img_array = preprocess_image(img_path)

    prediction = model.predict(img_array)

    class_names = ["Benign", "Malignant"]  # adjust if you have more classes
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]

    print(f"Predicted Class: {predicted_class}")                            