# face_counter.py

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more natural mirror-like view
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Get the number of faces detected
    face_count = len(faces)

    # Display a warning if face count is not equal to 1
    if face_count == 0:
        warning_text = "WARNING: No face detected!"
        cv2.putText(frame, warning_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif face_count > 1:
        warning_text = f"WARNING: {face_count} faces detected!"
        cv2.putText(frame, warning_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # If exactly one face is detected, you can show a confirmation message
        status_text = "Status: OK"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('AI Proctor - Face Count', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
























































































# import cv2

# # Haar Cascade classifier को लोड करें (यह फाइल आपको OpenCV के साथ मिलती है या ऑनलाइन डाउनलोड कर सकते हैं)
# # Make sure you have this file in the same directory or provide the full path.
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # वेबकैम शुरू करें (0 का मतलब है डिफ़ॉल्ट वेबकैम)
# cap = cv2.VideoCapture(0)

# while True:
#     # वेबकैम से एक-एक फ्रेम पढ़ें
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # चेहरे को पहचानने के लिए फ्रेम को ग्रेस्केल में बदलें
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # फ्रेम में चेहरों का पता लगाएं
#     # detectMultiScale function returns a list of rectangles (x, y, w, h)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # पहचाने गए हर चेहरे के चारों ओर एक आयत (rectangle) बनाएं
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # परिणामी फ्रेम दिखाएं
#     cv2.imshow('Online Proctoring System - Face Detection', frame)

#     # 'q' बटन दबाने पर लूप से बाहर निकलें
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # सब कुछ हो जाने के बाद, कैप्चर को रिलीज़ करें और विंडो बंद कर दें
# cap.release()
# cv2.destroyAllWindows()







# import torch
# import cv2

# # YOLOv5 मॉडल को PyTorch Hub से लोड करें
# # 'yolov5s' एक छोटा और तेज मॉडल है
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # जिस इमेज का परीक्षण करना है, उसका नाम
# img_path = 'test_image.jpg' # अपनी इमेज का पाथ यहाँ डालें
# img = cv2.imread(img_path)

# # मॉडल से अनुमान लगाएं
# results = model(img)

# # परिणामों को प्रोसेस करें और दिखाएं
# results.print()  # कंसोल पर परिणाम प्रिंट करें (जैसे: 1 person, 1 cell phone)
# results.show()   # इमेज को बाउंडिंग बॉक्स के साथ दिखाएं

# # अगर आप बाउंडिंग बॉक्स के कोऑर्डिनेट्स चाहते हैं:
# df = results.pandas().xyxy[0]
# print(df)

# # सिर्फ सेल फोन का पता लगाने के लिए
# cell_phones = df[df['name'] == 'cell phone']
# if not cell_phones.empty:
#     print(f"Cell phone detected! Details:\n{cell_phones}")








#     from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Backend Server is running!"

# # यह वह API एंडपॉइंट होगा जहां Frontend फ्रेम भेजेगा
# @app.route('/analyze_frame', methods=['POST'])
# def analyze_frame():
#     # अभी के लिए, हम सिर्फ एक डमी रिस्पॉन्स भेज रहे हैं
#     # असली प्रोजेक्ट में, आप यहाँ इमेज डेटा प्राप्त करेंगे और AI मॉडल चलाएंगे
    
#     # मान लीजिए, विश्लेषण के बाद यह परिणाम आया
#     suspicious_activity = {
#         'face_count': 1,
#         'phone_detected': False,
#         'looking_away': True
#     }
    
#     return jsonify(suspicious_activity)

# if __name__ == '__main__':
#     app.run(debug=True)