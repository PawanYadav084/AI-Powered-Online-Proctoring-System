# app.py (Updated with CORS)

from flask import Flask, request, jsonify
from flask_cors import CORS # <-- 1. YEH LINE ADD KAREIN
import torch
import cv2
import numpy as np
import base64

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # <-- 2. YEH LINE ADD KAREIN

# --- Load Models (only once when the server starts) ---
print("Loading models, please wait...")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.conf = 0.40
print("Models loaded successfully!")

# --- Core AI Logic in a Function ---
def analyze_image_frame(frame):
    warnings = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        warnings.append("No Face Detected!")
    elif len(faces) > 1:
        warnings.append(f"{len(faces)} Faces Detected!")
    results = yolo_model(frame)
    df = results.pandas().xyxy[0]
    prohibited_items = ['cell phone', 'book']
    for index, row in df.iterrows():
        item_name = row['name']
        if item_name in prohibited_items:
            warnings.append(f"{item_name.title()} Detected!")
    return warnings

# --- API Endpoint ---
@app.route('/process_frame', methods=['POST'])
def process_frame():
    json_data = request.get_json()
    image_data_base64 = json_data['image_data'].split(',')[1]
    image_bytes = base64.b64decode(image_data_base64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    warnings = analyze_image_frame(frame)
    return jsonify({'warnings': warnings})

# --- Main entry point ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
























# # app.py

# from flask import Flask, request, jsonify
# import torch
# import cv2
# import numpy as np
# import base64

# # --- Initialize Flask App ---
# app = Flask(__name__)

# # --- Load Models (only once when the server starts) ---
# print("Loading models, please wait...")
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# yolo_model.conf = 0.40
# print("Models loaded successfully!")

# # --- Core AI Logic in a Function ---
# def analyze_image_frame(frame):
#     """
#     This function takes a single image frame (as a numpy array)
#     and returns a list of warnings.
#     """
#     warnings = []

#     # 1. Face Detection Analysis
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 5)
#     if len(faces) == 0:
#         warnings.append("No Face Detected!")
#     elif len(faces) > 1:
#         warnings.append(f"{len(faces)} Faces Detected!")

#     # 2. Object Detection Analysis
#     results = yolo_model(frame)
#     df = results.pandas().xyxy[0]
    
#     prohibited_items = ['cell phone', 'book']
#     for index, row in df.iterrows():
#         item_name = row['name']
#         if item_name in prohibited_items:
#             warnings.append(f"{item_name.title()} Detected!")
            
#     return warnings

# # --- API Endpoint ---
# @app.route('/process_frame', methods=['POST'])
# def process_frame():
#     # Get the image data from the POST request
#     json_data = request.get_json()
#     image_data_base64 = json_data['image_data'].split(',')[1] # Remove the "data:image/jpeg;base64," part

#     # Decode the Base64 string to bytes
#     image_bytes = base64.b64decode(image_data_base64)
    
#     # Convert bytes to a numpy array for OpenCV
#     np_arr = np.frombuffer(image_bytes, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
#     # Analyze the frame using our AI function
#     warnings = analyze_image_frame(frame)
    
#     # Return the warnings as a JSON response
#     return jsonify({'warnings': warnings})

# # --- Main entry point ---
# if __name__ == '__main__':
#     # Use host='0.0.0.0' to make it accessible on your network
#     app.run(host='0.0.0.0', port=5001, debug=True)