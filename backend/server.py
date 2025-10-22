from flask import Flask, request, jsonify
from ultralytics import YOLO
from datetime import datetime
import os
import cv2
import base64
import numpy as np

app = Flask(__name__, static_folder="../frontend", static_url_path="/")


model = YOLO("best.pt")  

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/detect-page')
def detect_page():
    return app.send_static_file('detect.html')

@app.route('/realtime-page')
def realtime_page():
    return app.send_static_file('realtime.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    os.makedirs("uploads", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_path = os.path.join("uploads", f"input_{timestamp}.jpg")
    file.save(input_path)

    img = cv2.imread(input_path)
    if img is None:
        return jsonify({"error": "Image cannot be read"}), 400

    results = model.predict(source=input_path, conf=0.5, save=False)
    result = results[0]

    detections = []
    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[cls]
            confidence = float(score)
            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2]
            })
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"detections": detections, "image_base64": img_base64})

if __name__ == '__main__':
    app.run(debug=True)
