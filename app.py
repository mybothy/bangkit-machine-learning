import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

# Load the TensorFlow Lite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load labels from labels.txt
def load_labels(labels_file_path):
    with open(labels_file_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Preprocess the input image
def preprocess_image(image, target_size=(224, 224)):
    try:
        # Open image and convert to RGB format
        img = Image.open(image).convert('RGB')
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        raise

# Predict the image
def predict_image(interpreter, image_array):
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Convert predictions to a serializable format
        predictions_list = predictions.tolist()  # Convert numpy array to list
        return predictions_list
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

# Flask application setup
app = Flask(__name__)

model_path = "FINALBGSBGT/my_model_lite_with_metadata.tflite"  # Path to your TFLite model
labels_path = "FINALBGSBGT/labels.txt"                         # Path to your labels.txt file

# Load model and labels
interpreter = load_tflite_model(model_path)
labels = load_labels(labels_path)

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image_path = file.stream

    # Load model and preprocess image
    interpreter = load_tflite_model(model_path)
    image_array = preprocess_image(image_path)
    
    try:
        # Predict the image
        predictions = predict_image(interpreter, image_array)
        
        # Convert predictions to JSON
        predicted_class = labels[np.argmax(predictions)]
        confidence_score = float(np.max(predictions))  # Ensure it is serializable

        return jsonify({
            "predicted_class": predicted_class,
            "confidence_score": confidence_score
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
