
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load your custom YOLO model
model = YOLO('best.pt')  # Update this path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Load image
    try:
        img = Image.open(io.BytesIO(file.read()))
        print("Image successfully loaded.")
    except Exception as e:
        return jsonify({'error': f'Error loading image: {str(e)}'})

    # Make predictions
    try:
        results = model.predict(source=img, conf=0.25)
        print("Predictions made.")
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'})

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        predictions = []
        for i in range(len(results[0].boxes.conf)):
            confidence = results[0].boxes.conf[i].item()
            class_id = int(results[0].boxes.cls[i])
            class_name = results[0].names[class_id]
            predictions.append({'class': class_name, 'confidence': confidence})

        # Sort predictions by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        # Format predictions with confidence percentages
        formatted_predictions = [
            {'class': pred['class'], 'confidence': f"{pred['confidence'] * 100:.2f}%" }
            for pred in predictions
        ]
    else:
        formatted_predictions = [{'class': 'No predictions', 'confidence': '0.00%'}]

    return jsonify({'predictions': formatted_predictions})

if __name__ == '__main__':
    app.run(debug=True)
