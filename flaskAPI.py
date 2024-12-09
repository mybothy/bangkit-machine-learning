from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load labels from a labels.txt file
def load_labels(labels_file_path):
    with open(labels_file_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Preprocess the input image
def preprocess_image(file_stream, target_size=(224, 224)):
    image = Image.open(file_stream).convert("RGB").resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)

# Predict the image
def predict_image(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

# Flask application setup
app = Flask(__name__)

model_path = "FINALBGSBGT/my_model_lite_with_metadata.tflite"  # Path to your TFLite model
labels_path = "FINALBGSBGT/labels.txt"                         # Path to your labels.txt file

# Load labels
labels = load_labels(labels_path)

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        interpreter = load_tflite_model(model_path)
        image_array = preprocess_image(file.stream)
        predictions = predict_image(interpreter, image_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = labels[predicted_class_index]
        confidence_score = np.max(predictions)
        print(f"Predicted Class: {predicted_class_name}, Confidence Score: {confidence_score}")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "predicted_class": predicted_class_name,
        "confidence_score": confidence_score
    })

if __name__ == '__main__':
    app.run(debug=True)
